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
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time

import yaml

from agent_system.training.jobs import ENTRYPOINTS
from .store import ACTIVE, Store

log = logging.getLogger('quoridor.jobs')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z')
LIMITS = dict(cpus=(1, 12), memory_gb=(2, 48), timeout_h=(0.05, 72))
DEFAULTS = dict(gpu=dict(cpus=6, memory_gb=24, timeout_h=12), cpu=dict(cpus=4, memory_gb=8, timeout_h=6))


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class Controller:
    def __init__(self, *, data_root, git, docker, image, source_ref, uid_gid='1000:1000',
                 max_cpu_jobs=2, clock=time.time):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root / 'jobs'
        self.runs = self.root / 'runs'
        self.runs.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root / 'jobs.sqlite')
        self.git, self.docker, self.image, self.source_ref = git, docker, image, source_ref
        self.uid_gid, self.max_cpu_jobs, self.clock = uid_gid, max_cpu_jobs, clock
        self.lock = threading.RLock()
        self.catalog_lock = threading.Lock()

    # ---- submission -------------------------------------------------------------------------
    def entrypoints(self):
        commit = self.git.resolve(self.source_ref)
        rows = {}
        for name, meta in ENTRYPOINTS.items():
            row = dict(meta)
            if 'templates' in meta:
                row['templates'] = [f for f in self.git.files(commit, meta['templates']) if f.endswith(('.yaml', '.yml'))]
            rows[name] = row
        return dict(source_ref=self.source_ref, source_commit=commit, image=self.image, entrypoints=rows)

    def template(self, entrypoint, name):
        meta = ENTRYPOINTS[entrypoint]
        if 'templates' not in meta or not NAME.fullmatch(name):
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
        commit = self.git.resolve(payload.get('source') or self.source_ref)
        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime(self.clock()))
        identifier = f'job-{stamp}-{secrets.token_hex(3)}'
        spec = dict(entrypoint=entrypoint, config=config, resources=resources, image=self.image,
                    source=dict(ref=payload.get('source') or self.source_ref, commit=commit))
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
            except Exception as exc:
                log.exception('registration failed for %s', identifier)
                fields.update(status='failed', error=f'registration failed: {exc}')
        else:
            fields.update(status='failed', error=f'exit code {exit_code}')
        self.store.update(identifier, **fields)
        self.docker.remove(job['container'])

    # ---- registration -----------------------------------------------------------------------
    def register(self, job):
        run = self.runs / job['id']
        manifest = run / 'manifest.json'
        if not manifest.is_file():
            return []
        entries, registered = [], []
        models = self.data_root / 'models'
        models.mkdir(exist_ok=True)
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
                                # The Job run directory is a Training artifact root entry named by the Job id.
                                run_ref=dict(source='legacy', run_id=job['id']),
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
        return job

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

    def metrics(self, identifier, limit=5000):
        if self.store.get(identifier) is None:
            raise KeyError(identifier)
        run = self.runs / identifier

        def rows(name):
            path = run / name
            if not path.exists():
                return []
            values = []
            for line in path.read_text(errors='replace').splitlines()[-limit:]:
                try:
                    values.append(json.loads(line))
                except ValueError:
                    continue  # a line may be partially written while the Job runs
            return values
        return dict(metrics=rows('metrics.jsonl'), evaluations=rows('eval_results.jsonl'))

    def loop(self, stop: threading.Event, interval=2.0):
        while not stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception('scheduler tick failed')
            stop.wait(interval)
