"""Promotion decisions: gate evidence, verified store, append-only log, HTTP token."""
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from apps.jobs.app import create_app
from apps.jobs.controller import Controller
from apps.jobs.tests.test_controller import FakeDocker, FakeGit

SHA = 'a' * 64


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = [1_760_000_000.0]
        self.controller = Controller(data_root=self.root, git=FakeGit(), docker=FakeDocker(), image='img:test',
                                     source_ref='feature/x', clock=lambda: self.now[0])

    def gate(self, status='complete', candidate=SHA):
        job = self.controller.submit(dict(
            entrypoint='run_arena_matrix', config={'candidates': [candidate], 'gate': 'gate-v1'}))
        self.controller.store.update(job['id'], status='succeeded', finished=self.now[0])
        season = self.root / 'arena' / job['id']
        season.mkdir(parents=True)
        (season / 'manifest.json').write_text(json.dumps({'roster': {'agents': [
            {'id': 'cand', 'algo': {'params': {'sha256': candidate}}}, {'id': 'greedy', 'algo': {}}]}}))
        (season / 'report.json').write_text(json.dumps({'status': status, 'protocol': 'paired', 'manifest_sha256': 'm' * 64,
            'ranking': [{'id': 'cand', 'rank': 1, 'elo': 1600.0, 'score': .7, 'games': 10, 'wins': 7, 'draws': 0, 'losses': 3},
                        {'id': 'greedy', 'rank': 2}],
            'matrix': {'cand': {'cand': None, 'greedy': {'games': 10, 'wins': 7, 'draws': 0, 'losses': 3, 'score': .7}}}}))
        return job['id']

    def verify(self, sha=SHA):
        (self.root / 'models').mkdir(exist_ok=True)
        (self.root / 'models' / 'verified.json').write_text(json.dumps({'models': {sha: {'size': 1, 'mtime_ns': 1}}}))

    def test_promotion_needs_a_verified_gate_candidate_and_a_reason(self):
        p = self.controller.promotions
        season = self.gate()
        with self.assertRaisesRegex(ValueError, 'verified'):
            p.promote(SHA, season, 'ok')
        self.verify()
        with self.assertRaisesRegex(ValueError, 'reason'):
            p.promote(SHA, season, '  ')
        with self.assertRaisesRegex(ValueError, 'gate season'):
            p.promote(SHA, 'job-other', 'ok')
        self.assertEqual(p.gates('b' * 64), [])
        record = p.promote(SHA, season, ' beats the gate ')
        self.assertEqual((record['note'], record['evidence']['rank'], record['evidence']['opponents']),
                         ('beats the gate', 1, {'greedy': {'games': 10, 'wins': 7, 'draws': 0, 'losses': 3, 'score': .7}}))
        with self.assertRaisesRegex(ValueError, 'Already deployed'):
            p.promote(SHA, season, 'again')
        p.withdraw(SHA, 'regression found')
        self.assertEqual(p.current(), {})
        self.assertEqual([r['action'] for r in p.history()], ['promote', 'withdraw'])
        with self.assertRaisesRegex(ValueError, 'not deployed'):
            p.withdraw(SHA, 'twice')

    def test_incomplete_gate_seasons_are_not_evidence(self):
        self.gate(status='partial')
        self.assertEqual(self.controller.promotions.gates(SHA), [])

    def test_http_promotion_needs_the_token(self):
        season = self.gate()
        self.verify()
        token = self.root / 'token'
        token.write_text('secret')
        client = TestClient(create_app(self.controller, str(token)))
        body = dict(sha256=SHA, season=season, note='gate passed')
        self.assertEqual(client.post('/api/v1/promotions', json=body).status_code, 401)
        self.assertEqual(client.get(f'/api/v1/promotions/candidates/{SHA}').json()['gates'][0]['season'], season)
        self.assertEqual(client.post('/api/v1/promotions', json=body, headers={'X-Platform-Token': 'secret'}).json()['action'], 'promote')
        self.assertEqual(client.post('/api/v1/promotions', json=body, headers={'X-Platform-Token': 'secret'}).status_code, 422)
        self.assertEqual([r['sha256'] for r in client.get('/api/v1/promotions').json()['deployed']], [SHA])


if __name__ == '__main__':
    unittest.main()
