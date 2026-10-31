import gzip
import json
from pathlib import Path
import tempfile
import unittest
from fastapi import HTTPException
from backend.agent_arena import Evidence, Sessions, CreateGame, GameStep
from backend import test_replay as fixtures
from backend.replay import _stable_hash


class ArenaTests(unittest.TestCase):
    def setUp(self):
        fixtures.ReplayTests.setUpClass()
        self.fixture = fixtures.ReplayTests()
        self.fixture.setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'arena' / 'test'
        (self.base / 'blocks').mkdir(parents=True)
        self.models = type('Models', (), {'detail': lambda _, sha, verify_sha=False: {'run_refs':[{'run_id':'exact-run','checkpoint_sha256':sha}]}})()
        self.ev = Evidence(self.root, self.models)
        self.publish()

    def publish(self):
        self.fixture.seal()
        self.base.joinpath('manifest.json').write_text(json.dumps(self.fixture.manifest))
        self.base.joinpath('report.json').write_text(json.dumps({'status':'complete','manifest_sha256':self.fixture.manifest['manifest_sha256'],'ranking':[],'matrix':{}}))
        with gzip.open(self.base / 'blocks' / ('a'*20+'.json.gz'),'wt') as f:
            json.dump(self.fixture.block,f)

    def test_manifest_and_traversal_fail_closed(self):
        self.assertEqual(len(self.ev.roster('test')['agents']),2)
        with self.assertRaises(ValueError):self.ev.season('../test')
        with self.assertRaises(ValueError):self.ev.replay('test','../a:0')
        v=json.loads((self.base/'report.json').read_text());v['manifest_sha256']='0'*64
        (self.base/'report.json').write_text(json.dumps(v))
        with self.assertRaises(ValueError):self.ev.season('test')

    def test_seasons_mark_gate_protocols(self):
        self.assertIsNone(self.ev.seasons()['seasons'][0]['gate'])
        self.fixture.manifest['roster']['gate'] = 'gate-v1'
        self.publish()
        self.assertEqual(self.ev.seasons()['seasons'][0]['gate'], 'gate-v1')

    def test_rankings_join_exact_sha_per_season(self):
        rows, warnings = self.ev.rankings()
        self.assertEqual(warnings, [])
        self.assertEqual(list(rows), ['a'*64])
        self.assertEqual(rows['a'*64][0]['season'], 'test')
        self.assertEqual(rows['a'*64][0]['agent'], 'policy')
        self.assertIsNone(rows['a'*64][0]['elo'])
        v = json.loads((self.base/'report.json').read_text())
        v['ranking'] = [{'id':'policy','rank':1,'elo':1530.0,'wins':3,'draws':1,'losses':0}]
        (self.base/'report.json').write_text(json.dumps(v))
        entry = self.ev.rankings()[0]['a'*64][0]
        self.assertEqual((entry['rank'], entry['elo'], entry['wins']), (1, 1530.0, 3))
        (self.root/'arena'/'broken').mkdir()
        (self.root/'arena'/'broken'/'report.json').write_text('{}')
        rows, warnings = self.ev.rankings()
        self.assertEqual(len(rows['a'*64]), 1)
        self.assertTrue(any(w.startswith('broken:') for w in warnings))

    def test_pair_replay_exact_turn_and_source(self):
        result=self.ev.games('test','policy','minimax')
        self.assertEqual(len(result['games']),1)
        replay=self.ev.replay('test',result['games'][0]['id'])
        self.assertEqual(replay['positions'][0]['turn'],2)
        self.assertEqual(replay['positions'][-1]['pawns'],[[4,1],[4,7]])
        self.assertEqual(replay['diagnostics_scope'],'game_aggregate')
        self.assertNotIn('candidate_scores',replay)
        self.assertEqual(self.ev.training('test','policy')['runs'][0]['checkpoint_sha256'],'a'*64)
        self.assertEqual(self.ev.training('test','minimax')['runs'],[])

    def test_heuristic_only_replay_and_corruption(self):
        self.fixture.manifest['roster']['agents'][0]['algo']={'type':'greedy'}
        self.publish()
        replay=self.ev.replay('test','a'*20+':0')
        self.assertIsNone(replay['source']['checkpoint_sha256'])
        self.fixture.block['games'][0]['actions'][0]['target']=[99,99]
        with gzip.open(self.base/'blocks'/('a'*20+'.json.gz'),'wt') as f:json.dump(self.fixture.block,f)
        with self.assertRaises(ValueError):self.ev.replay('test','a'*20+':0')


class SessionTests(unittest.TestCase):
    def setUp(self):
        class Ev:
            def spec(_,season,identifier):
                spec={'id':identifier,'algo':{'type':'greedy'},'policy':{'type':'top_k','k':1}}
                return dict(id=identifier,name=identifier,kind='heuristic',spec=spec,sha256=None,playable=True,identity=_stable_hash(spec))
        self.sessions=Sessions(Ev(),capacity=2)
        self.game=self.sessions.create(CreateGame(season='s',p1='human',p2='greedy'))

    def test_human_agent_revision_and_capability(self):
        g=self.game
        with self.assertRaises(HTTPException):self.sessions.get(g['id'],'wrong')
        first=self.sessions.step(g['id'],g['token'],GameStep(revision=0,action={'player':1,'type':'pawn','target':[4,1]}))
        self.assertEqual(first['revision'],1)
        with self.assertRaises(HTTPException) as e:self.sessions.step(g['id'],g['token'],GameStep(revision=0))
        self.assertEqual(e.exception.status_code,409)
        second=self.sessions.step(g['id'],g['token'],GameStep(revision=1))
        self.assertEqual(second['revision'],2)
        self.assertEqual(len(second['positions']),2)

    def test_illegal_no_mutation_and_capacity(self):
        g=self.game
        with self.assertRaises(ValueError):self.sessions.step(g['id'],g['token'],GameStep(revision=0,action={'player':1,'type':'pawn','target':[0,8]}))
        self.assertEqual(self.sessions.get(g['id'],g['token'])['revision'],0)
        self.sessions.create(CreateGame(season='s',p1='human',p2='human'))
        with self.assertRaises(HTTPException) as e:self.sessions.create(CreateGame(season='s',p1='human',p2='human'))
        self.assertEqual(e.exception.status_code,429)


if __name__=='__main__':unittest.main()
