"""/api/v1 model index: exact SHA joins, separate capability counts and HTTP errors."""
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import json
from pathlib import Path
import tempfile
from backend.platform_v1 import TimedCache, alias_legacy, diagnoses_index, install, model_detail, model_listing

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

    def test_diagnoses_index_joins_exact_sha_across_experiments_and_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = lambda sha, wins: dict(sha256=sha, model_id='m', games=50, counts=dict(wins=wins, losses=50 - wins), tactical_wins=18, tactical_count=18)
            for path, rows in (('experiments/e1/arm/diagnoses.json', [row(A, 38), row(B, 10)]),
                               ('jobs/runs/job-1/diagnoses.json', [row(A, 37)]),
                               ('experiments/e1/notes.json', [row(A, 1)])):
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(json.dumps(rows))
            (root / 'jobs/runs/job-2').mkdir(parents=True)
            (root / 'jobs/runs/job-2/diagnoses.json').write_text('{broken')
            index, warnings = diagnoses_index(root)
            self.assertEqual([(d['source'], d['wins']) for d in index[A]], [('experiments/e1/arm/diagnoses.json', 38), ('jobs/runs/job-1/diagnoses.json', 37)])
            self.assertEqual(len(index[B]), 1)
            self.assertEqual(len(warnings), 1)

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


class LegacyAliasTests(unittest.TestCase):
    def test_v1_paths_share_legacy_handlers(self):
        app = FastAPI()
        app.get('/api/runs')(lambda: {'runs': []})
        app.get('/api/runs/{run_id}/metrics')(lambda run_id: {'run': run_id})
        app.get('/api/agents/seasons/{season}')(lambda season: {'season': season})
        app.get('/api/models')(lambda: {'catalog': True})
        app.get('/api/models-extra')(lambda: {})
        app.get('/api/platform/models')(lambda: {})
        alias_legacy(app)
        client = TestClient(app)
        self.assertEqual(client.get('/api/v1/research/runs').json(), {'runs': []})
        self.assertEqual(client.get('/api/v1/research/runs/r1/metrics').json(), {'run': 'r1'})
        self.assertEqual(client.get('/api/v1/seasons/s1').json(), {'season': 's1'})
        self.assertEqual(client.get('/api/v1/research/catalog').json(), {'catalog': True})
        self.assertEqual(client.get('/api/runs/r1/metrics').json(), {'run': 'r1'})
        paths = {r.path for r in app.routes}
        self.assertNotIn('/api/v1/research/catalog-extra', paths)
        self.assertFalse(any(p.startswith('/api/v1/platform') for p in paths))


if __name__ == '__main__':
    unittest.main()
