"""The immutable model store and the single generated model index, maintained by the Job controller.

* ``models/verified.json`` - verification ledger: SHA-256 -> size, mtime_ns, verified_at for every
  ``models/<sha>.pt`` whose content hash equals its name. Only new or changed files are hashed. The
  platform reads the ledger instead of hashing (a store file is verified while size and mtime match).
* ``platform-p1-p2/catalogs/research-catalog-current.json`` - the merged model index (every catalog,
  registry, Job registration and AgentSpec) exported by ``platform_entry.py --generate-lab-catalog``
  and published atomically under a stable name, so the Lab sees new checkpoints without a restart.
* ``sync_command`` (optional) runs afterwards, e.g. to start the Dev evidence mirror so a new checkpoint is
  playable on Dev at once instead of after the next timer run.
Source catalogs are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

log = logging.getLogger('quoridor.jobs.models')
SOURCE = Path(__file__).resolve().parents[2]
CURRENT = 'research-catalog-current.json'


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _publish(path, text):
    partial = path.with_name(path.name + '.partial')
    partial.write_text(text)
    os.replace(partial, path)


class ModelStore:
    def __init__(self, data_root, mlflow_uri=None, clock=time.time, sync_command=None):
        self.data_root = Path(data_root).resolve()
        self.models = self.data_root / 'models'
        self.ledger = self.models / 'verified.json'
        self.catalogs = self.data_root / 'platform-p1-p2' / 'catalogs'
        self.mlflow_uri, self.clock, self.lock = mlflow_uri, clock, threading.Lock()
        self.sync_command = shlex.split(sync_command) if sync_command else None

    def verify(self):
        """Hash store files that are new or changed since the ledger; returns (verified, hashed, mismatched)."""
        ledger = json.loads(self.ledger.read_text()).get('models', {}) if self.ledger.exists() else {}
        verified, hashed, mismatched = {}, 0, []
        for path in sorted(self.models.glob('*.pt')):
            name, stat = path.stem, path.stat()
            entry = ledger.get(name)
            if entry and entry['size'] == stat.st_size and entry['mtime_ns'] == stat.st_mtime_ns:
                verified[name] = entry
                continue
            hashed += 1
            if _sha256(path) == name:
                verified[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, verified_at=self.clock())
            else:
                mismatched.append(path.name)
        if verified != ledger or not self.ledger.exists():
            _publish(self.ledger, json.dumps(dict(schema_version=1, models=verified, mismatched=mismatched), indent=1, sort_keys=True))
        return len(verified), hashed, mismatched

    def export_index(self):
        """Regenerate the merged research catalog and publish it as research-catalog-current.json."""
        env = dict(os.environ, QUORIDOR_DATA_ROOT=str(self.data_root), PLATFORM_MODEL_CATALOGS='discover',
                   PYTHONPATH=os.pathsep.join(str(SOURCE / p) for p in ('training-dashboard', '.', 'backend-server', 'engine')))
        if self.mlflow_uri:
            env['MLFLOW_TRACKING_URI'] = self.mlflow_uri
        out = subprocess.run([sys.executable, str(SOURCE / 'training-dashboard' / 'platform_entry.py'),
                              '--generate-lab-catalog', str(self.catalogs)], env=env, cwd=SOURCE,
                             capture_output=True, text=True, timeout=600, check=True).stdout.strip().splitlines()[-1]
        generated = Path(out)
        if generated.parent != self.catalogs or not generated.is_file():
            raise RuntimeError(f'unexpected catalog path: {out}')
        current = self.catalogs / CURRENT
        if not current.exists() or current.read_bytes() != generated.read_bytes():
            partial = current.with_name(CURRENT + '.partial')
            shutil.copyfile(generated, partial)
            os.replace(partial, current)
        return generated

    def refresh(self):
        """Verify, then export; logged, never raised (a stale index is better than a stopped controller)."""
        with self.lock:
            try:
                count, hashed, mismatched = self.verify()
                generated = self.export_index()
                log.info('model store: %d verified (%d hashed, %d mismatched); index %s', count, hashed, len(mismatched), generated.name)
            except Exception:
                log.exception('model store refresh failed')
            if self.sync_command:
                try:
                    subprocess.run(self.sync_command, timeout=300, check=True, capture_output=True)
                    log.info('sync command finished')
                except Exception:
                    log.exception('sync command failed')
