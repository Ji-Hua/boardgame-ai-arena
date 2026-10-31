"""/api/v1 model index: exact SHA joins, separate capability counts and HTTP errors."""
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.platform_v1 import TimedCache, install, model_detail, model_listing

A, B = 'a'*64, 'b'*64


class Models:
    def __init__(self):
        self.verify = []

    def listing(self):
        return dict(models=[dict(sha256=A, available=True, analyzable=True, playable=True, deployed=False),
                            dict(sha256=B, available=False, analyzable=False, playable=False, deployed=False)],
                    count=2, sources=[], warnings=['catalog warning'])

    def detail(self, sha, verify_sha=False):
        self.verify.append(verify_sha)
        if sha != A:
            raise KeyError(sha)
        return dict(sha256=A, available=True, warnings=[])


class Evidence:
    def rankings(self):
        return {A: [dict(season='s1', agent='policy', elo=1510.0)]}, ['season warning']


class PlatformV1Tests(unittest.TestCase):
    def test_listing_marks_ranked_by_exact_sha_and_counts_states_separately(self):
        result = model_listing(Models(), Evidence())
        rows = {r['sha256']: r for r in result['models']}
        self.assertTrue(rows[A]['ranked'])
        self.assertEqual(rows[A]['seasons'][0]['season'], 's1')
        self.assertFalse(rows[B]['ranked'])
        self.assertEqual(rows[B]['seasons'], [])
        self.assertEqual(result['summary']['total'], 2)
        self.assertEqual(result['summary']['ranked'], 1)
        self.assertEqual(result['summary']['playable'], 1)
        self.assertEqual(result['summary']['deployed'], 0)
        self.assertEqual(result['warnings'], ['catalog warning', 'season warning'])

    def test_detail_verifies_sha(self):
        models = Models()
        self.assertTrue(model_detail(models, Evidence(), A)['ranked'])
        self.assertEqual(models.verify, [True])

    def test_timed_cache_reuses_value_until_ttl(self):
        now = [0.0]
        cache = TimedCache(ttl=60, clock=lambda: now[0])
        calls = []
        compute = lambda: calls.append(1) or len(calls)
        self.assertEqual(cache.get(compute), 1)
        now[0] = 59
        self.assertEqual(cache.get(compute), 1)
        now[0] = 60
        self.assertEqual(cache.get(compute), 2)

    def test_http_routes_and_unknown_model(self):
        app = FastAPI()
        app.state.models, app.state.evidence = Models(), Evidence()
        install(app)
        client = TestClient(app)
        self.assertEqual(client.get('/api/v1/models').json()['summary']['ranked'], 1)
        self.assertEqual(client.get('/api/v1/models/' + A).json()['seasons'][0]['agent'], 'policy')
        self.assertEqual(client.get('/api/v1/models/' + B).status_code, 404)
        self.assertEqual(client.get('/api/v1/experiments/..%2Fx').status_code, 404)
        self.assertEqual(client.get('/api/v1/experiments/.hidden').status_code, 422)
        self.assertEqual(client.get('/api/v1/experiments/missing').status_code, 404)


if __name__ == '__main__':
    unittest.main()
