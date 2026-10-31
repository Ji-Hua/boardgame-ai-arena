"""SQLite record of platform Jobs. Status changes are the only mutable state."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading

ACTIVE = ('queued', 'preparing', 'running')
FINAL = ('succeeded', 'failed', 'cancelled')
SCHEMA = '''create table if not exists jobs(
  id text primary key, name text, entrypoint text not null, experiment text, spec text not null,
  status text not null, created real not null, started real, finished real, exit_code integer,
  container text, error text, registered text, cancel_requested integer not null default 0)'''


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute('pragma journal_mode=wal')
            self.db.execute(SCHEMA)

    @staticmethod
    def _row(row):
        if row is None:
            return None
        job = dict(row)
        job['spec'] = json.loads(job['spec'])
        job['registered'] = json.loads(job['registered']) if job['registered'] else []
        job['cancel_requested'] = bool(job['cancel_requested'])
        return job

    def insert(self, job):
        with self.lock:
            self.db.execute('insert into jobs(id,name,entrypoint,experiment,spec,status,created) values(?,?,?,?,?,?,?)',
                            (job['id'], job['name'], job['entrypoint'], job['experiment'], json.dumps(job['spec']),
                             job['status'], job['created']))

    def update(self, identifier, **fields):
        if 'registered' in fields:
            fields['registered'] = json.dumps(fields['registered'])
        columns = ', '.join(f'{k}=?' for k in fields)
        with self.lock:
            self.db.execute(f'update jobs set {columns} where id=?', (*fields.values(), identifier))

    def get(self, identifier):
        with self.lock:
            return self._row(self.db.execute('select * from jobs where id=?', (identifier,)).fetchone())

    def listing(self, statuses=None, limit=200, experiment=None):
        clauses, args = [], []
        if statuses:
            clauses.append(f'status in ({",".join("?" * len(statuses))})')
            args += list(statuses)
        if experiment:
            clauses.append('experiment=?')
            args.append(experiment)
        query = 'select * from jobs' + (' where ' + ' and '.join(clauses) if clauses else '')
        with self.lock:
            rows = self.db.execute(query + ' order by created desc, rowid desc limit ?', (*args, limit)).fetchall()
        return [self._row(r) for r in rows]
