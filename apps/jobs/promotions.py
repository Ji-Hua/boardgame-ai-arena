"""Promotion: the human-confirmed step that marks a checkpoint deployed, with an append-only audit log.

``<data root>/promotions.jsonl`` holds one JSON object per decision (``promote`` or ``withdraw``); the
deployed set is the replay of that log. A promotion needs a verified store checkpoint that was a candidate
of a complete gate season published by a platform gate Job, plus a written reason; the gate evidence for
that checkpoint is copied into the record. Rosters, champion aliases and seasons are never changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time

SHA = re.compile(r'[0-9a-f]{64}\Z')
OUTCOME = ('games', 'wins', 'draws', 'losses', 'score')


class Promotions:
    def __init__(self, data_root, store, clock=time.time):
        self.data_root, self.store, self.clock = Path(data_root), store, clock
        self.path = self.data_root / 'promotions.jsonl'
        self.lock = threading.Lock()

    def history(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def current(self):
        deployed = {}
        for record in self.history():
            if record['action'] == 'promote':
                deployed[record['sha256']] = record
            else:
                deployed.pop(record['sha256'], None)
        return deployed

    def gates(self, sha):
        """Complete gate seasons (published by run_arena_matrix Jobs) in which ``sha`` was a candidate."""
        rows = []
        for job in self.store.listing(('succeeded',)):
            if job['entrypoint'] != 'run_arena_matrix':
                continue
            config = json.loads(job['spec']['config']) if isinstance(job['spec']['config'], str) else job['spec']['config']
            season = self.data_root / 'arena' / job['id']
            if sha not in (config.get('candidates') or []) or not (season / 'report.json').is_file():
                continue
            report, manifest = (json.loads((season / name).read_text()) for name in ('report.json', 'manifest.json'))
            agent = next((a['id'] for a in manifest['roster']['agents']
                          if ((a.get('algo') or {}).get('params') or {}).get('sha256') == sha), None)
            ranking = next((r for r in report.get('ranking', []) if r['id'] == agent), None)
            if report.get('status') != 'complete' or agent is None or ranking is None:
                continue
            opponents = {other: {k: cell.get(k) for k in OUTCOME}
                         for other, cell in (report.get('matrix', {}).get(agent) or {}).items() if cell and cell.get('games')}
            rows.append(dict(season=job['id'], gate=config.get('gate'), protocol=report.get('protocol'), agent=agent,
                             manifest_sha256=report.get('manifest_sha256'), rank=ranking.get('rank'), elo=ranking.get('elo'),
                             ci95=ranking.get('ci95'), roster_size=len(report.get('ranking', [])),
                             **{k: ranking.get(k) for k in OUTCOME}, opponents=opponents, published=job['finished']))
        return rows

    def _append(self, record):
        with self.lock:
            with self.path.open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return record

    def _check(self, sha, note):
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError('sha256 must be a full lowercase SHA-256')
        if not isinstance(note, str) or not note.strip() or len(note) > 2000:
            raise ValueError('A written reason (1-2000 characters) is required')

    def promote(self, sha, season, note):
        self._check(sha, note)
        ledger = self.data_root / 'models' / 'verified.json'
        verified = json.loads(ledger.read_text()).get('models', {}) if ledger.exists() else {}
        if sha not in verified:
            raise ValueError('Only a verified checkpoint in the model store can be promoted')
        if sha in self.current():
            raise ValueError('Already deployed; withdraw it first to record a new decision')
        evidence = next((g for g in self.gates(sha) if g['season'] == season), None)
        if evidence is None:
            raise ValueError('The season is not a complete gate season with this checkpoint as a candidate')
        return self._append(dict(time=self.clock(), action='promote', sha256=sha, season=season,
                                 gate=evidence['gate'], note=note.strip(), evidence=evidence))

    def withdraw(self, sha, note):
        self._check(sha, note)
        if sha not in self.current():
            raise ValueError('This checkpoint is not deployed')
        return self._append(dict(time=self.clock(), action='withdraw', sha256=sha, note=note.strip()))
