"""Record finished Jobs as MLflow runs through the REST API (no mlflow package on the host).

Job containers have no network, so the controller logs after a Job ends: tags that tie the run to
the Job, the flattened configuration as params, every training metric row (step = episode) and
evaluation win rates per opponent. The result is written to ``<run>/mlflow.json``; a Job is logged
once. MLflow being down never fails a Job: the next controller start retries unlogged Jobs.
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml

log = logging.getLogger('quoridor.jobs.mlflow')
STATUS = dict(succeeded='FINISHED', failed='FAILED', cancelled='KILLED')
SKIP = ('episode', 'step', 'timestamp', 'type')
BATCH = 1000  # MLflow log-batch limit: 1000 metrics, params and tags together (at most 100 params) per request


def flatten(value, prefix=''):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flatten(item, f'{prefix}{key}.').items()}
    return {prefix[:-1]: value}


def _metric(key, value, step, stamp):
    return dict(key=key[:250], value=float(value), step=int(step), timestamp=stamp)


def rows_to_metrics(train_rows, eval_rows, stamp):
    """MLflow metric records; non-numeric and non-finite values are left out. Each row gets its own timestamp
    (start + row index in ms), so a file that repeats an episode still yields distinct (key, step, timestamp) records."""
    out = []
    for index, row in enumerate(train_rows):
        step = row.get('episode', index)
        for key, value in row.items():
            if key not in SKIP and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                out.append(_metric(key, value, step, stamp + index))
    for index, row in enumerate(eval_rows):
        name = row.get('opponent_name') or row.get('opponent') or row.get('opponent_id')
        if name and isinstance(row.get('win_rate'), (int, float)) and row.get('episode') is not None:
            out.append(_metric(f'eval/{name}/win_rate', row['win_rate'], row['episode'], stamp + index))
    return out


class MlflowLog:
    def __init__(self, uri, timeout=15.0):
        self.uri, self.timeout = uri.rstrip('/'), timeout

    def _call(self, method, path, payload=None, query=None):
        url = f'{self.uri}/api/2.0/mlflow/{path}' + (f'?{urllib.parse.urlencode(query)}' if query else '')
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b'{}')
        except urllib.error.HTTPError as exc:
            exc.msg = f'{exc.msg}: {exc.read()[:500].decode(errors="replace")}'
            raise

    def experiment(self, name):
        try:
            return self._call('GET', 'experiments/get-by-name', query=dict(experiment_name=name))['experiment']['experiment_id']
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        return self._call('POST', 'experiments/create', dict(name=name))['experiment_id']

    def log(self, job, config_text, train_rows, eval_rows):
        project = job['spec'].get('project_id', 'quoridor-legacy')
        prefix = 'platform' if project == 'quoridor-legacy' else f'platform/projects/{project}'
        experiment = f"{prefix}/{job['experiment'] or 'jobs'}"
        experiment_id = self.experiment(experiment)
        start = int((job.get('started') or job['created']) * 1000)
        tags = {'mlflow.runName': job['name'], 'quoridor.job_id': job['id'], 'quoridor.entrypoint': job['entrypoint'],
                'quoridor.source_commit': job['spec']['source']['commit'], 'quoridor.status': job['status'],
                'quoridor.experiment': job['experiment'] or '', 'quoridor.registered': ','.join(job.get('registered') or [])[:5000]}
        tags['platform.project_id'] = project
        if job['spec'].get('game_ref'):
            tags['platform.game_ref'] = json.dumps(job['spec']['game_ref'], sort_keys=True)
        run_id = self._call('POST', 'runs/create', dict(experiment_id=experiment_id, start_time=start, run_name=job['name'],
                                                       tags=[dict(key=k, value=str(v)) for k, v in tags.items()]))['run']['info']['run_id']
        try:
            return self._fill(run_id, experiment_id, experiment, job, config_text, train_rows, eval_rows, start)
        except Exception:
            try:  # leave no half-written run behind; the Job is retried at the next controller start
                self._call('POST', 'runs/delete', dict(run_id=run_id))
            except Exception:
                log.exception('could not delete incomplete MLflow run %s', run_id)
            raise

    def _fill(self, run_id, experiment_id, experiment, job, config_text, train_rows, eval_rows, start):
        try:
            config = yaml.safe_load(config_text) if isinstance(config_text, str) else config_text
        except yaml.YAMLError:
            config = None
        params = {'entrypoint': job['entrypoint'], **{k: v for k, v in flatten(config).items() if k}} if isinstance(config, dict) else {}
        params = [dict(key=k[:250], value=json.dumps(v) if isinstance(v, (list, dict)) else str(v)[:500]) for k, v in list(params.items())[:100]]
        metrics = rows_to_metrics(train_rows, eval_rows, start)
        if params:
            self._call('POST', 'runs/log-batch', dict(run_id=run_id, params=params))
        for index in range(0, len(metrics), BATCH):
            self._call('POST', 'runs/log-batch', dict(run_id=run_id, metrics=metrics[index:index + BATCH]))
        end = int((job.get('finished') or time.time()) * 1000)
        self._call('POST', 'runs/update', dict(run_id=run_id, status=STATUS.get(job['status'], 'FINISHED'), end_time=end))
        return dict(run_id=run_id, experiment_id=experiment_id, experiment=experiment, metrics=len(metrics))
