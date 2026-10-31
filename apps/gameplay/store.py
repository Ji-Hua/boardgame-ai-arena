"""SQLite record of platform games. Moves are append-only; positions are recomputed by the engine."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading

SCHEMA = '''create table if not exists games(
  id text primary key, play_hash text not null, watch_hash text not null, season text not null,
  identities text not null, actions text not null, revision integer not null, status text not null,
  winner integer, listed integer not null default 0, created real not null, updated real not null)'''
STATUSES = ('active', 'finished', 'closed', 'abandoned')


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
        game = dict(row)
        game['identities'] = json.loads(game['identities'])
        game['actions'] = json.loads(game['actions'])
        game['listed'] = bool(game['listed'])
        return game

    def insert(self, game):
        with self.lock:
            self.db.execute('insert into games(id,play_hash,watch_hash,season,identities,actions,revision,status,'
                            'created,updated) values(?,?,?,?,?,?,0,?,?,?)',
                            (game['id'], game['play_hash'], game['watch_hash'], game['season'],
                             json.dumps(game['identities']), '[]', 'active', game['created'], game['created']))

    def get(self, identifier):
        with self.lock:
            return self._row(self.db.execute('select * from games where id=?', (identifier,)).fetchone())

    def append(self, identifier, revision, actions, status, winner, now):
        """Record one move; the revision guard makes a concurrent writer fail instead of overwrite."""
        with self.lock:
            changed = self.db.execute('update games set actions=?,revision=?,status=?,winner=?,updated=? '
                                      'where id=? and revision=? and status=?',
                                      (json.dumps(actions), revision + 1, status, winner, now,
                                       identifier, revision, 'active')).rowcount
        return changed == 1

    def update(self, identifier, now, **fields):
        assert set(fields) <= {'status', 'listed'}
        with self.lock:
            self.db.execute(f'update games set {",".join(k + "=?" for k in fields)},updated=? where id=?',
                            (*fields.values(), now, identifier))

    def count(self, status):
        with self.lock:
            return self.db.execute('select count(*) from games where status=?', (status,)).fetchone()[0]

    def abandon(self, before, now):
        with self.lock:
            return self.db.execute("update games set status='abandoned',updated=? where status='active' and updated<?",
                                   (now, before)).rowcount

    def listing(self, limit=100):
        with self.lock:
            rows = self.db.execute("select * from games where listed=1 and status='finished' "
                                   'order by updated desc, rowid desc limit ?', (limit,)).fetchall()
        return [self._row(r) for r in rows]
