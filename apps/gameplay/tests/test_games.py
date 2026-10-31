import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.gameplay.app import create_app
from apps.gameplay.games import Conflict, CreateGame, Denied, Games, GameStep, IDLE_SECONDS
from apps.gameplay.store import Store
from backend.replay import _dependencies


class Evidence:
    def spec(self, season, identifier):
        if identifier == 'missing':
            raise StopIteration
        spec = {'id': identifier, 'algo': {'type': identifier}, 'policy': {'type': 'top_k', 'k': 1}}
        return dict(id=identifier, name=identifier, kind='heuristic', spec=spec, sha256=None,
                    playable=identifier != 'locked')


class GameTests(unittest.TestCase):
    serialize = staticmethod(_dependencies()[0]._serialize_state)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'games.sqlite'
        self.now = [1000.0]
        self.games = self.open()

    def tearDown(self):
        self.tmp.cleanup()

    def open(self, **kw):
        return Games(Store(self.db), Evidence(), self.serialize, clock=lambda: self.now[0], **kw)

    def create(self, p1='human', p2='greedy', games=None):
        return (games or self.games).create(CreateGame(season='s', p1=p1, p2=p2))

    def test_moves_survive_restart_and_agents_reload(self):
        g = self.create()
        self.games.step(g['id'], g['token'], GameStep(revision=0, action={'player': 1, 'type': 'pawn', 'target': [4, 1]}))
        reopened = self.open()
        view = reopened.view(g['id'], g['token'])
        self.assertEqual((view['revision'], view['position']['pawns'][0]), (1, [4, 1]))
        after = reopened.step(g['id'], g['token'], GameStep(revision=1))
        self.assertEqual(after['revision'], 2)
        self.assertEqual(after['actions'][1]['player'], 2)
        self.assertEqual(len(after['positions']), 2)
        # Positions are recomputed by the engine from the stored actions only.
        stored = sqlite3.connect(self.db).execute('select actions, play_hash from games').fetchone()
        self.assertNotIn('positions', stored[0])
        self.assertNotEqual(stored[1], g['token'])

    def test_watch_token_reads_only(self):
        g = self.create()
        watch = self.games.view(g['id'], g['watch_token'])
        self.assertEqual((watch['role'], watch['legal_actions']), ('watcher', []))
        with self.assertRaises(Denied):
            self.games.step(g['id'], g['watch_token'], GameStep(revision=0, action={'player': 1, 'type': 'pawn', 'target': [4, 1]}))
        with self.assertRaises(Denied):
            self.games.close(g['id'], g['watch_token'])
        with self.assertRaises(Denied):
            self.games.view(g['id'], 'wrong')

    def test_revision_illegal_and_agent_rules(self):
        g = self.create()
        with self.assertRaises(ValueError):
            self.games.step(g['id'], g['token'], GameStep(revision=0, action={'player': 1, 'type': 'pawn', 'target': [0, 8]}))
        with self.assertRaises(ValueError):
            self.games.step(g['id'], g['token'], GameStep(revision=0))
        self.assertEqual(self.games.view(g['id'], g['token'])['revision'], 0)
        self.games.step(g['id'], g['token'], GameStep(revision=0, action={'player': 1, 'type': 'pawn', 'target': [4, 1]}))
        with self.assertRaises(Conflict):
            self.games.step(g['id'], g['token'], GameStep(revision=0))
        with self.assertRaises(ValueError):
            self.games.step(g['id'], g['token'], GameStep(revision=1, action={'player': 2, 'type': 'pawn', 'target': [4, 7]}))
        with self.assertRaises(ValueError):
            self.create(p2='locked')

    def test_finished_listing_close_and_idle_abandon(self):
        g = self.create('greedy', 'greedy')
        view = g
        while view['status'] == 'active':
            view = self.games.step(g['id'], g['token'], GameStep(revision=view['revision']))
        self.assertIn(view['winner'], (1, 2))
        with self.assertRaises(Conflict):
            self.games.step(g['id'], g['token'], GameStep(revision=view['revision']))
        with self.assertRaises(Denied):
            self.games.view(g['id'], '')
        self.games.set_listed(g['id'], g['token'], True)
        public = self.games.view(g['id'], '')
        self.assertEqual((public['role'], public['winner']), ('public', view['winner']))
        self.assertEqual([x['id'] for x in self.games.listing()['games']], [g['id']])
        self.games.close(g['id'], g['token'])
        self.assertEqual(self.games.listing()['games'], [])
        with self.assertRaises(Denied):
            self.games.view(g['id'], g['token'])
        active = self.create()
        with self.assertRaises(Conflict):
            self.games.set_listed(active['id'], active['token'], True)
        self.now[0] += IDLE_SECONDS + 1
        self.create()
        self.assertEqual(self.games.store.get(active['id'])['status'], 'abandoned')

    def test_active_limit(self):
        games = self.open(active_limit=1)
        self.create(games=games)
        with self.assertRaises(Exception) as caught:
            self.create(games=games)
        self.assertEqual(type(caught.exception).__name__, 'Busy')

    def test_http_contract(self):
        client = TestClient(create_app(self.games))
        g = client.post('/api/v1/play', json=dict(season='s', p1='human', p2='greedy')).json()
        self.assertEqual(client.get('/api/v1/play/' + g['id']).status_code, 403)
        moved = client.post(f'/api/v1/play/{g["id"]}/step', headers={'X-Game-Token': g['token']},
                            json=dict(revision=0, action=dict(player=1, type='pawn', target=[4, 1])))
        self.assertEqual(moved.json()['revision'], 1)
        stale = client.post(f'/api/v1/play/{g["id"]}/step', headers={'X-Game-Token': g['token']}, json=dict(revision=0))
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(client.get('/api/v1/play/' + g['id'], headers={'X-Game-Token': g['watch_token']}).json()['role'], 'watcher')
        self.assertEqual(client.put(f'/api/v1/play/{g["id"]}/listing', headers={'X-Game-Token': g['token']},
                                    json=dict(listed=True)).status_code, 409)
        self.assertEqual(client.post('/api/v1/play', json=dict(season='s', p1='human', p2='missing')).status_code, 404)
        self.assertEqual(client.delete('/api/v1/play/' + g['id'], headers={'X-Game-Token': g['token']}).json(), dict(closed=True))
        self.assertEqual(client.get('/health').json()['active'], 0)


if __name__ == '__main__':
    unittest.main()
