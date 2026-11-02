"""Job scheduling: one GPU Job at a time, bounded CPU Jobs, restart reconciliation.

Layout under ``<data root>/jobs``:
  jobs.sqlite              Job records
  <id>/job.json            container-facing specification (mounted read-only)
  <id>/source/             read-only snapshot of the submitted commit
  <id>/stdout.log          container output, saved when the container ends
  runs/<id>/               the Job run directory (training artifact contract; a Training artifact root)
Registered checkpoints are copied to ``<data root>/models/<sha>.pt`` and listed in
``<data root>/catalog-jobs.json``, which the model index discovers. Registration
never changes champions, rosters or the MLflow database.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time

import yaml

from agent_system.training.jobs import ENTRYPOINTS
from .health import assess
from .store import ACTIVE, Store

log = logging.getLogger('quoridor.jobs')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z')
TEMPLATE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}(/[A-Za-z0-9][A-Za-z0-9._ -]{0,79})?\Z')
# Metric files in preference order: the training contract, then the CNN research runner.
METRIC_FILES = ('metrics.jsonl', 'experiment/metrics.jsonl', 'experiment/research-metrics.jsonl')
CHECKPOINT_NAME = re.compile(r'ep(\d+)(?:_step(\d+))?')
LIMITS = dict(cpus=(1, 12), memory_gb=(2, 48), timeout_h=(0.05, 72))
DEFAULTS = dict(gpu=dict(cpus=6, memory_gb=24, timeout_h=12), cpu=dict(cpus=4, memory_gb=8, timeout_h=6))


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class Controller:
    def __init__(self, *, data_root, git, docker, image, source_ref, uid_gid='1000:1000',
                 max_cpu_jobs=2, clock=time.time, mlflow=None, model_store=None):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root / 'jobs'
        self.runs = self.root / 'runs'
        self.runs.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root / 'jobs.sqlite')
        self.git, self.docker, self.image, self.source_ref = git, docker, image, source_ref
        self.uid_gid, self.max_cpu_jobs, self.clock = uid_gid, max_cpu_jobs, clock
        self.lock = threading.RLock()
        self.catalog_lock = threading.Lock()
        self.mlflow, self.mlflow_lock, self.mlflow_busy = mlflow, threading.Lock(), set()
        self.model_store = model_store  # verification ledger and merged index (model_store.ModelStore)

    # ---- submission -------------------------------------------------------------------------
    def entrypoints(self):
        commit = self.git.resolve(self.source_ref)
        rows = {}
        for name, meta in ENTRYPOINTS.items():
            row = dict(meta)
            if 'templates' in meta:
                pattern = re.compile(meta.get('template_pattern', r'[^/]+\.ya?ml'))
                row['templates'] = [f for f in self.git.files(commit, meta['templates']) if pattern.fullmatch(f)]
            rows[name] = row
        return dict(source_ref=self.source_ref, source_commit=commit, image=self.image, entrypoints=rows)

    def template(self, entrypoint, name):
        meta = ENTRYPOINTS[entrypoint]
        if 'templates' not in meta or not TEMPLATE.fullmatch(name) or '..' in name:
            raise KeyError(name)
        return self.git.show(self.git.resolve(self.source_ref), f"{meta['templates']}/{name}")

    def submit(self, payload):
        entrypoint = payload.get('entrypoint')
        if entrypoint not in ENTRYPOINTS:
            raise ValueError(f'Unknown entrypoint: {entrypoint}')
        meta = ENTRYPOINTS[entrypoint]
        config = payload.get('config')
        if meta['format'] == 'yaml':
            if not isinstance(config, str) or not isinstance(yaml.safe_load(config), dict):
                raise ValueError('Config must be a YAML mapping')
        else:
            parsed = json.loads(config) if isinstance(config, str) else config
            if not isinstance(parsed, dict):
                raise ValueError('Config must be a JSON object')
            config = json.dumps(parsed)
        name = payload.get('name') or entrypoint
        experiment = payload.get('experiment') or None
        for value in (name, experiment):
            if value is not None and not NAME.fullmatch(value):
                raise ValueError(f'Invalid name: {value!r}')
        gpu = bool(meta['gpu'])
        resources = dict(DEFAULTS['gpu' if gpu else 'cpu'], gpu=gpu)
        for key, (low, high) in LIMITS.items():
            if key in (payload.get('resources') or {}):
                value = float(payload['resources'][key])
                if not low <= value <= high:
                    raise ValueError(f'{key} must be within [{low}, {high}]')
                resources[key] = value
        follow_up = payload.get('follow_up') or {}
        if set(follow_up) - {'gate'} or (follow_up.get('gate') and meta['kind'] != 'train'):
            raise ValueError('follow_up only supports {"gate": "<gate id>"} for training Jobs')
        commit = self.git.resolve(payload.get('source') or self.source_ref)
        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime(self.clock()))
        identifier = f'job-{stamp}-{secrets.token_hex(3)}'
        spec = dict(entrypoint=entrypoint, config=config, resources=resources, image=self.image,
                    source=dict(ref=payload.get('source') or self.source_ref, commit=commit),
                    follow_up={k: v for k, v in follow_up.items() if v})
        (self.root / identifier).mkdir()
        (self.runs / identifier).mkdir()
        self.store.insert(dict(id=identifier, name=name, entrypoint=entrypoint, experiment=experiment, spec=spec,
                               status='queued', created=self.clock()))
        return self.store.get(identifier)

    def cancel(self, identifier):
        with self.lock:
            job = self.store.get(identifier)
            if job is None:
                raise KeyError(identifier)
            if job['status'] == 'queued':
                self.store.update(identifier, status='cancelled', finished=self.clock(), error='cancelled before start')
            elif job['status'] in ('preparing', 'running'):
                self.store.update(identifier, cancel_requested=1)
                if job['container']:
                    threading.Thread(target=self.docker.stop, args=(job['container'],), daemon=True).start()
            return self.store.get(identifier)

    # ---- scheduling -------------------------------------------------------------------------
    def tick(self):
        with self.lock:
            for job in self.store.listing(('preparing', 'running')):
                self._reconcile(job)
            active = self.store.listing(('preparing', 'running'))
            gpu_busy = any(j['spec']['resources']['gpu'] for j in active)
            cpu_running = sum(1 for j in active if not j['spec']['resources']['gpu'])
            for job in reversed(self.store.listing(('queued',))):  # oldest first
                if job['spec']['resources']['gpu']:
                    if gpu_busy:
                        continue
                    gpu_busy = True
                else:
                    if cpu_running >= self.max_cpu_jobs:
                        continue
                    cpu_running += 1
                self._start(job)

    def _container_args(self, job):
        r = job['spec']['resources']
        control, run = self.root / job['id'], self.runs / job['id']
        # --init forwards the stop signal to the Job process, so cancellation ends it gracefully.
        args = ['--init', '--name', f"qjob-{job['id']}", '--label', f"org.quoridor.job={job['id']}", '--user', self.uid_gid,
                '--cpus', str(r['cpus']), '--memory', f"{r['memory_gb']}g", '--shm-size', '2g',
                '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,size=2g',
                '-e', f"OMP_NUM_THREADS={int(r['cpus'])}", '-e', 'HOME=/tmp',
                # The host uid has no passwd entry in the image; getpass and torch caches read these instead.
                '-e', 'USER=quoridor-job', '-e', 'LOGNAME=quoridor-job', '-e', 'TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor',
                '-e', f"QUORIDOR_SOURCE_COMMIT={job['spec']['source']['commit']}",
                '-v', f'{self.data_root}:{self.data_root}:ro',
                '-v', f'{control / "job.json"}:/job/job.json:ro', '-v', f'{control / "source"}:/job/source:ro',
                '-v', f'{run}:/job/run:rw']
        if r['gpu']:
            args += ['--device', 'nvidia.com/gpu=all']
        return args + [job['spec']['image']]

    def _start(self, job):
        identifier = job['id']
        self.store.update(identifier, status='preparing', started=self.clock())
        try:
            source = self.root / identifier / 'source'
            if not source.exists():
                partial = self.root / identifier / 'source.partial'
                shutil.rmtree(partial, ignore_errors=True)
                partial.mkdir()
                self.git.export(job['spec']['source']['commit'], partial)
                partial.rename(source)
            spec = dict(job['spec'], id=identifier, name=job['name'], experiment=job['experiment'],
                        run_dir='/job/run', data_root=str(self.data_root))
            (self.root / identifier / 'job.json').write_text(json.dumps(spec, indent=2))
            container = self.docker.run(self._container_args(job))
            self.store.update(identifier, status='running', container=f'qjob-{identifier}')
            log.info('started %s in %s', identifier, container[:12])
        except Exception as exc:
            log.exception('start failed for %s', identifier)
            self.store.update(identifier, status='failed', finished=self.clock(), error=f'start failed: {exc}')

    def _reconcile(self, job):
        if job['status'] == 'preparing' and not job['container']:
            # The controller stopped while exporting source; nothing ran, so fail visibly rather than rerun.
            self.store.update(job['id'], status='failed', finished=self.clock(), error='controller restarted during preparation')
            return
        state = self.docker.state(job['container'])
        if state is None:
            self.store.update(job['id'], status='failed', finished=self.clock(), error='container lost')
            return
        if state.get('Running'):
            limit = job['spec']['resources']['timeout_h'] * 3600
            if job['started'] and self.clock() - job['started'] > limit and not job['cancel_requested']:
                self.store.update(job['id'], cancel_requested=1, error='timeout')
                threading.Thread(target=self.docker.stop, args=(job['container'],), daemon=True).start()
            return
        self._finalize(job, int(state.get('ExitCode', -1)))

    def _finalize(self, job, exit_code):
        identifier = job['id']
        (self.root / identifier / 'stdout.log').write_text(self.docker.logs(job['container']))
        fields = dict(exit_code=exit_code, finished=self.clock())
        if job['cancel_requested']:
            fields.update(status='cancelled', error=job['error'] or 'cancelled')
        elif exit_code == 0:
            try:
                fields.update(status='succeeded', registered=self.register(job))
                self.publish_season(job)
            except Exception as exc:
                log.exception('registration failed for %s', identifier)
                fields.update(status='failed', error=f'registration failed: {exc}')
        else:
            fields.update(status='failed', error=f'exit code {exit_code}')
        self.store.update(identifier, **fields)
        self.docker.remove(job['container'])
        if self.mlflow:
            threading.Thread(target=self.log_mlflow, args=(identifier,), daemon=True).start()
        if self.model_store and fields.get('registered'):
            threading.Thread(target=self.model_store.refresh, daemon=True).start()
        gate = job['spec'].get('follow_up', {}).get('gate')
        if gate and fields.get('status') == 'succeeded' and fields.get('registered'):
            # Gate the final registered checkpoint; the follow-up Job keeps the source commit and experiment.
            follow = self.submit(dict(entrypoint='run_arena_matrix', name=f"{job['name']}-gate"[:80], experiment=job['experiment'],
                                      source=job['spec']['source']['commit'],
                                      config=dict(candidates=[fields['registered'][-1]], names={fields['registered'][-1]: job['name']},
                                                  gate=gate, workers=4)))
            log.info('queued gate %s after %s', follow['id'], identifier)

    def publish_season(self, job):
        """Copy a complete Arena result to <data root>/arena/<job id>, where the ladder lists it."""
        source = self.runs / job['id'] / 'arena'
        report = source / 'report.json'
        if not report.exists() or json.loads(report.read_text()).get('status') != 'complete':
            return None
        target = self.data_root / 'arena' / job['id']
        if not target.exists():
            partial = target.with_name(target.name + '.partial')
            shutil.rmtree(partial, ignore_errors=True)
            shutil.copytree(source, partial, ignore=shutil.ignore_patterns('.lock'))
            partial.rename(target)
        return job['id']

    # ---- registration -----------------------------------------------------------------------
    def register(self, job):
        run = self.runs / job['id']
        manifest = run / 'manifest.json'
        if not manifest.is_file():
            return []
        entries, registered = [], []
        models = self.data_root / 'models'
        models.mkdir(exist_ok=True)
        # Only runs that follow the training artifact contract are listed by the Training workspace.
        readable = any((run / name).exists() for name in ('config.yaml', 'metrics.jsonl'))
        for item in json.loads(manifest.read_text())['checkpoints']:
            source = (run / item['file']).resolve()
            if not source.is_relative_to(run) or _sha256(source) != item['sha256']:
                raise ValueError(f"Checkpoint {item['file']} does not match its manifest SHA")
            target = models / f"{item['sha256']}.pt"
            if target.exists():
                if _sha256(target) != item['sha256']:
                    raise ValueError(f'Immutable store corruption: {target}')
            else:
                partial = target.with_suffix('.partial')
                shutil.copyfile(source, partial)
                partial.chmod(0o444)
                partial.rename(target)
            entries.append(dict(id=f"{job['name']}_{source.stem}", sha256=item['sha256'], checkpoint=str(target),
                                source_path=str(source), training_job=job['id'], training_run=str(run),
                                experiment=job['experiment'],
                                # The Job run directory is a Training artifact root entry named by the Job id.
                                **(dict(run_ref=dict(source='legacy', run_id=job['id'])) if readable else {}),
                                output_semantics=item.get('output_semantics'),
                                source_commit=job['spec']['source']['commit'],
                                selection_reason='registered by platform Job', metadata=item.get('metadata') or {}))
            registered.append(item['sha256'])
        self._append_catalog(entries)
        return registered

    def _append_catalog(self, entries):
        if not entries:
            return
        path = self.data_root / 'catalog-jobs.json'
        with self.catalog_lock:
            catalog = json.loads(path.read_text()) if path.exists() else dict(models=[], description='Checkpoints registered by platform Jobs')
            known = {(m['sha256'], m['training_job']) for m in catalog['models']}
            catalog['models'] += [e for e in entries if (e['sha256'], e['training_job']) not in known]
            partial = path.with_suffix('.partial')
            partial.write_text(json.dumps(catalog, indent=2))
            os.replace(partial, path)

    # ---- MLflow -------------------------------------------------------------------------------
    def log_mlflow(self, identifier):
        """Record a finished Job as an MLflow run once (``<run>/mlflow.json``); errors are logged, never raised."""
        record = self.runs / identifier / 'mlflow.json'
        with self.mlflow_lock:
            if record.exists() or identifier in self.mlflow_busy:
                return None
            self.mlflow_busy.add(identifier)
        try:
            job = self.store.get(identifier)
            if job is None or job['status'] in ACTIVE:
                return None
            data = self.metrics(identifier)
            train = [r for r in data['metrics'] if r.get('type', 'train') == 'train']
            result = self.mlflow.log(job, job['spec']['config'], train, data['evaluations'])
            partial = record.with_suffix('.partial')
            partial.write_text(json.dumps(result, indent=2))
            os.replace(partial, record)
            log.info('logged %s to MLflow run %s', identifier, result['run_id'])
            return result
        except Exception:
            log.exception('MLflow logging failed for %s', identifier)
            return None
        finally:
            with self.mlflow_lock:
                self.mlflow_busy.discard(identifier)

    def backfill_mlflow(self):
        """Log every finished Job that has no MLflow run yet (oldest first)."""
        if self.mlflow:
            for job in reversed(self.store.listing(('succeeded', 'failed', 'cancelled'))):
                self.log_mlflow(job['id'])

    # ---- reads ------------------------------------------------------------------------------
    def detail(self, identifier):
        job = self.store.get(identifier)
        if job is None:
            raise KeyError(identifier)
        run = self.runs / identifier
        events = run / 'events.jsonl'
        job['events'] = [json.loads(x) for x in events.read_text().splitlines() if x.strip()] if events.exists() else []
        job['run_id'] = identifier
        job['run_files'] = sorted(p.name for p in run.iterdir()) if run.exists() else []
        job['season'] = identifier if (self.data_root / 'arena' / identifier / 'report.json').exists() else None
        diagnoses = next((p for p in (run / 'diagnoses.json', run / 'experiment' / 'diagnoses.json') if p.exists()), run / 'diagnoses.json')
        try:
            job['diagnoses'] = json.loads(diagnoses.read_text()) if diagnoses.exists() else []
        except ValueError:
            job['diagnoses'] = []  # rewritten after every checkpoint; may be mid-write
        job['checkpoints'] = self._checkpoints(run)
        try:
            job['mlflow'] = json.loads((run / 'mlflow.json').read_text()) if (run / 'mlflow.json').exists() else None
        except ValueError:
            job['mlflow'] = None
        return job

    def _checkpoints(self, run):
        """Saved checkpoints in training order: the manifest (with SHA-256) once written, the files before that.
        Episode and environment step come from the file name (``ep00050_step14841.pt``), as in the evaluation rows."""
        listed = None
        manifest = run / 'manifest.json'
        if manifest.is_file():
            try:
                listed = [(c['file'], c['sha256'], c.get('metadata') or {}) for c in json.loads(manifest.read_text())['checkpoints']]
            except (ValueError, KeyError):
                listed = None
        if listed is None:
            listed = [(str(p.relative_to(run)), None, {}) for d in ('checkpoints', 'experiment/checkpoints')
                      for p in sorted((run / d).glob('*.pt'))]
        rows = []
        for file, sha, metadata in listed:
            match = CHECKPOINT_NAME.search(Path(file).stem)
            episode = int(match[1]) if match else metadata.get('episode_count')
            rows.append(dict(file=file, name=Path(file).name, sha256=sha, episode=episode,
                             step=int(match[2]) if match and match[2] else None))
        return sorted(rows, key=lambda r: (r['episode'] is None, r['episode'] or 0, r['name']))

    def log_text(self, identifier, tail=400):
        job = self.store.get(identifier)
        if job is None:
            raise KeyError(identifier)
        saved = self.root / identifier / 'stdout.log'
        if job['status'] == 'running' and job['container']:
            return self.docker.logs(job['container'], tail=tail)
        if saved.exists():
            return '\n'.join(saved.read_text(errors='replace').splitlines()[-tail:])
        return ''

    def metrics(self, identifier, limit=5000, after=0):
        """Metric rows from line ``after`` on (the stream sends only new lines), evaluations and health alerts."""
        job = self.store.get(identifier)
        if job is None:
            raise KeyError(identifier)
        run = self.runs / identifier

        def flat(value, prefix=''):
            if isinstance(value, dict):
                return {k: v for key, item in value.items() for k, v in flat(item, f'{prefix}{key}.').items()}
            return {prefix[:-1]: value} if isinstance(value, (int, float, str)) and not isinstance(value, bool) else {}

        def rows(name):
            """(line index, row) pairs of the last ``limit`` lines, and the file's line count."""
            path = run / name
            if not path.exists():
                return [], 0
            lines = path.read_text(errors='replace').splitlines()
            values = []
            for index in range(max(0, len(lines) - limit), len(lines)):
                try:
                    values.append((index, flat(json.loads(lines[index]))))
                except ValueError:
                    continue  # a line may be partially written while the Job runs
            return values, len(lines)

        def finite(row):  # JSON responses cannot carry NaN or Inf; the alerts already reported them
            return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
        source = next((name for name in METRIC_FILES if (run / name).exists()), None)
        metrics, total = rows(source) if source else ([], 0)
        evaluations = [row for _, row in rows('eval_results.jsonl')[0]]
        replays = [row for _, row in rows('eval_replays.jsonl')[0]]
        alerts = assess([row for _, row in metrics], evaluations, running=job['status'] == 'running',
                        last_write=(run / source).stat().st_mtime if source else job['started'], now=self.clock())
        start = max(after, total - limit)
        return dict(metrics=[finite(row) for index, row in metrics if index >= start], metric_source=source,
                    offset=start, total=total, evaluations=[finite(row) for row in evaluations], alerts=alerts,
                    replays=[finite(row) for row in replays])

    def loop(self, stop: threading.Event, interval=2.0):
        while not stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception('scheduler tick failed')
            stop.wait(interval)
