"""Job controller: validation, GPU exclusivity, registration, cancellation, restart reconciliation, token."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from apps.jobs.app import create_app
from apps.jobs.controller import Controller

COMMIT = 'c' * 40
YAML = 'episodes: 10\nmodel_arch: mlp\n'


class FakeGit:
    def resolve(self, ref):
        if ref not in ('feature/x', COMMIT):
            raise ValueError(f'Unknown source commit or ref: {ref}')
        return COMMIT

    def export(self, commit, target):
        (Path(target) / 'README').write_text(commit)

    def files(self, commit, directory):
        return ['smoke.yaml', 'notes.txt', 'cnn_x/01-arm.json', 'dqn/nested.yaml']

    def show(self, commit, path):
        if path.endswith('smoke.yaml'):
            return YAML
        if path.endswith('cnn_x/01-arm.json'):
            return '{"training": {}}'
        raise KeyError(path)


class FakeDocker:
    def __init__(self):
        self.runs, self.states, self.stopped, self.removed = [], {}, [], []

    def run(self, args):
        name = args[args.index('--name') + 1]
        self.runs.append(args)
        self.states[name] = dict(Running=True, ExitCode=0)
        return name + '-id'

    def state(self, name):
        return self.states.get(name)

    def logs(self, name, tail=None):
        return 'log line\n'

    def stop(self, name, grace=30):
        self.stopped.append(name)

    def remove(self, name):
        self.removed.append(name)
        self.states.pop(name, None)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = [1_760_000_000.0]
        self.docker = FakeDocker()
        self.controller = self.make()

    def make(self):
        return Controller(data_root=self.root, git=FakeGit(), docker=self.docker, image='img:test',
                          source_ref='feature/x', max_cpu_jobs=1, clock=lambda: self.now[0])

    def submit(self, entrypoint='train_dqn', **extra):
        config = YAML if entrypoint == 'train_dqn' else {'checkpoints': ['a' * 64], 'opponents': []}
        return self.controller.submit(dict(entrypoint=entrypoint, config=config, **extra))

    def finish(self, job, code=0):
        self.docker.states[job['container']] = dict(Running=False, ExitCode=code)

    def test_submission_is_validated_and_pinned_to_a_commit(self):
        job = self.submit(name='smoke-1')
        self.assertEqual((job['status'], job['spec']['source']['commit']), ('queued', COMMIT))
        self.assertTrue(job['spec']['resources']['gpu'])
        for bad in (dict(entrypoint='rm -rf', config=YAML), dict(entrypoint='train_dqn', config='- a list'),
                    dict(entrypoint='train_dqn', config=YAML, name='../x'),
                    dict(entrypoint='train_dqn', config=YAML, source='unknown'),
                    dict(entrypoint='train_dqn', config=YAML, resources={'cpus': 99})):
            with self.assertRaises(ValueError):
                self.controller.submit(bad)
        rows = self.controller.entrypoints()['entrypoints']
        self.assertEqual(rows['train_dqn']['templates'], ['smoke.yaml'])
        self.assertEqual(rows['train_cnn_experiment']['templates'], ['cnn_x/01-arm.json'])
        self.assertEqual(self.controller.template('train_cnn_experiment', 'cnn_x/01-arm.json'), '{"training": {}}')
        with self.assertRaises(KeyError):
            self.controller.template('train_cnn_experiment', 'cnn_x/../../x.json')
        self.assertEqual(self.controller.template('train_dqn', 'smoke.yaml'), YAML)
        with self.assertRaises(KeyError):
            self.controller.template('train_dqn', '../secret.yaml')

    def test_one_gpu_job_at_a_time_and_bounded_cpu_jobs(self):
        first, second = self.submit(), self.submit()
        cpu1, cpu2 = self.submit('evaluate_checkpoints'), self.submit('evaluate_checkpoints')
        self.controller.tick()
        status = {j['id']: j['status'] for j in self.controller.store.listing()}
        self.assertEqual([status[j['id']] for j in (first, second, cpu1, cpu2)], ['running', 'queued', 'running', 'queued'])
        args = self.docker.runs[0]
        self.assertIn('nvidia.com/gpu=all', args)
        self.assertIn('none', args)  # --network none
        self.assertIn('USER=quoridor-job', args)
        self.assertIn('--init', args)
        self.assertIn(f'QUORIDOR_SOURCE_COMMIT={COMMIT}', args)
        self.assertIn(f'{self.root / "jobs" / "runs" / first["id"]}:/job/run:rw', args)
        self.assertNotIn('nvidia.com/gpu=all', self.docker.runs[1])
        spec = json.loads((self.root / 'jobs' / first['id'] / 'job.json').read_text())
        self.assertEqual((spec['run_dir'], spec['source']['commit']), ('/job/run', COMMIT))
        self.finish(self.controller.store.get(first['id']), 1)
        self.controller.tick()
        self.assertEqual(self.controller.store.get(first['id'])['status'], 'failed')
        self.assertEqual(self.controller.store.get(second['id'])['status'], 'running')

    def test_success_registers_verified_checkpoints_into_discoverable_catalog(self):
        job = self.submit(name='smoke')
        self.controller.tick()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'checkpoints').mkdir()
        (run / 'metrics.jsonl').write_text('')
        data = b'weights'
        (run / 'checkpoints' / 'ep00010.pt').write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        (run / 'manifest.json').write_text(json.dumps(dict(checkpoints=[dict(
            file='checkpoints/ep00010.pt', sha256=sha, output_semantics='q_values', metadata={'action_count': 209})])))
        self.finish(self.controller.store.get(job['id']))
        self.controller.tick()
        done = self.controller.store.get(job['id'])
        self.assertEqual((done['status'], done['registered']), ('succeeded', [sha]))
        self.assertEqual((self.root / 'models' / f'{sha}.pt').read_bytes(), data)
        catalog = json.loads((self.root / 'catalog-jobs.json').read_text())
        self.assertEqual(catalog['models'][0]['id'], 'smoke_ep00010')
        self.assertEqual(catalog['models'][0]['training_job'], job['id'])
        self.assertEqual(catalog['models'][0]['run_ref'], dict(source='legacy', run_id=job['id']))
        self.assertIsNone(catalog['models'][0]['experiment'])
        self.assertEqual([j['id'] for j in self.controller.store.listing(experiment='none-such')], [])
        self.assertEqual((self.root / 'jobs' / job['id'] / 'stdout.log').read_text(), 'log line\n')
        self.assertIn(f'qjob-{job["id"]}', self.docker.removed)

    def test_research_metrics_are_flattened_and_diagnoses_exposed(self):
        job = self.submit()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'experiment').mkdir()
        (run / 'experiment' / 'research-metrics.jsonl').write_text(
            json.dumps({'episode': 32, 'totals': {'learner_wins': 6}, 'imitation_ce': 1.38, 'flag': True}) + '\n{"partial')
        (run / 'experiment' / 'diagnoses.json').write_text(json.dumps([{'model_id': 'm', 'games': 50}]))
        metrics = self.controller.metrics(job['id'])
        self.assertEqual(metrics['metric_source'], 'experiment/research-metrics.jsonl')
        self.assertEqual(metrics['metrics'], [{'episode': 32, 'totals.learner_wins': 6, 'imitation_ce': 1.38}])
        self.assertEqual(self.controller.detail(job['id'])['diagnoses'][0]['games'], 50)

    def test_metrics_resume_from_an_offset_with_alerts_and_no_nan(self):
        job = self.submit()
        run = self.root / 'jobs' / 'runs' / job['id']
        rows = [dict(type='train', episode=i, loss=0.01, truncation_rate=0.6 if i == 3 else 0.0) for i in range(4)]
        (run / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n{"type": "train", "episode": 9, "loss": NaN}\n')
        full = self.controller.metrics(job['id'])
        self.assertEqual((full['offset'], full['total'], len(full['metrics'])), (0, 5, 5))
        self.assertIsNone(full['metrics'][-1]['loss'])
        self.assertEqual([a['code'] for a in full['alerts']], ['non_finite'])
        tail = self.controller.metrics(job['id'], after=3)
        self.assertEqual([r['episode'] for r in tail['metrics']], [3, 9])
        self.assertEqual(self.controller.metrics(job['id'], limit=2)['offset'], 3)

    def test_checkpoints_and_sample_games_are_listed_in_training_order(self):
        job = self.submit()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'checkpoints').mkdir()
        for name in ('ep00100_step900.pt', 'ep00050_step400.pt'):
            (run / 'checkpoints' / name).write_bytes(b'x')
        (run / 'eval_replays.jsonl').write_text(json.dumps(dict(episode=50, opponent_name='greedy', game_length=20,
                                                                replay_path='replays/eval_ep00050_greedy_game0000.json')) + '\n')
        rows = self.controller.detail(job['id'])['checkpoints']
        self.assertEqual([(r['episode'], r['step'], r['sha256']) for r in rows], [(50, 400, None), (100, 900, None)])
        (run / 'manifest.json').write_text(json.dumps(dict(checkpoints=[
            dict(file='checkpoints/ep00100_step900.pt', sha256='e' * 64, metadata=dict(episode_count=100))])))
        self.assertEqual([(r['name'], r['sha256']) for r in self.controller.detail(job['id'])['checkpoints']],
                         [('ep00100_step900.pt', 'e' * 64)])
        self.assertEqual(self.controller.metrics(job['id'])['replays'][0]['game_length'], 20)

    def test_stream_sends_rows_then_ends_for_a_finished_job(self):
        job = self.submit()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'metrics.jsonl').write_text(json.dumps(dict(type='train', episode=1, loss=0.2)) + '\n')
        self.controller.store.update(job['id'], status='succeeded')
        client = TestClient(create_app(self.controller, None, stream_interval=0.01))
        body = client.get(f'/api/v1/jobs/{job["id"]}/stream').text
        update = json.loads(body.split('event: update\ndata: ')[1].split('\n\n')[0])
        self.assertEqual((update['status'], update['offset'], update['metrics'][0]['loss']), ('succeeded', 0, 0.2))
        self.assertTrue(body.rstrip().endswith('event: end\ndata: {}'))
        self.assertEqual(client.get('/api/v1/jobs/job-missing/stream').status_code, 404)

    def test_research_runs_register_without_a_training_run_ref(self):
        job = self.submit('evaluate_checkpoints', experiment='exp-1')
        self.controller.tick()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'experiment' / 'checkpoints').mkdir(parents=True)
        (run / 'experiment' / 'checkpoints' / 'ep1.pt').write_bytes(b'r')
        sha = hashlib.sha256(b'r').hexdigest()
        (run / 'manifest.json').write_text(json.dumps(dict(checkpoints=[dict(file='experiment/checkpoints/ep1.pt', sha256=sha)])))
        self.finish(self.controller.store.get(job['id']))
        self.controller.tick()
        entry = json.loads((self.root / 'catalog-jobs.json').read_text())['models'][0]
        self.assertNotIn('run_ref', entry)
        self.assertEqual((entry['experiment'], entry['training_job']), ('exp-1', job['id']))
        self.assertEqual([j['id'] for j in self.controller.store.listing(experiment='exp-1')], [job['id']])

    def test_complete_arena_result_is_published_and_gate_follows_training(self):
        job = self.submit(name='cand', experiment='exp-2', follow_up={'gate': 'gate-v1'})
        with self.assertRaises(ValueError):
            self.submit('evaluate_checkpoints', follow_up={'gate': 'gate-v1'})
        self.controller.tick()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'checkpoints').mkdir()
        (run / 'checkpoints' / 'ep1.pt').write_bytes(b'g')
        sha = hashlib.sha256(b'g').hexdigest()
        (run / 'manifest.json').write_text(json.dumps(dict(checkpoints=[dict(file='checkpoints/ep1.pt', sha256=sha)])))
        self.finish(self.controller.store.get(job['id']))
        self.controller.tick()
        gate = next(j for j in self.controller.store.listing() if j['entrypoint'] == 'run_arena_matrix')
        self.assertEqual((gate['name'], gate['experiment'], gate['spec']['source']['commit']), ('cand-gate', 'exp-2', COMMIT))
        self.assertEqual(json.loads(gate['spec']['config'])['candidates'], [sha])
        arena = self.root / 'jobs' / 'runs' / gate['id'] / 'arena'
        (arena / 'blocks').mkdir(parents=True)
        (arena / 'report.json').write_text(json.dumps(dict(status='complete')))
        (arena / '.lock').write_text('')
        self.finish(self.controller.store.get(gate['id']))
        self.controller.tick()
        published = self.root / 'arena' / gate['id']
        self.assertTrue((published / 'report.json').exists())
        self.assertFalse((published / '.lock').exists())
        self.assertEqual(self.controller.detail(gate['id'])['season'], gate['id'])
        self.assertIsNone(self.controller.detail(job['id'])['season'])

    def test_tampered_checkpoint_fails_registration(self):
        job = self.submit()
        self.controller.tick()
        run = self.root / 'jobs' / 'runs' / job['id']
        (run / 'checkpoints').mkdir()
        (run / 'checkpoints' / 'a.pt').write_bytes(b'x')
        (run / 'manifest.json').write_text(json.dumps(dict(checkpoints=[dict(file='checkpoints/a.pt', sha256='0' * 64)])))
        self.finish(self.controller.store.get(job['id']))
        self.controller.tick()
        self.assertIn('registration failed', self.controller.store.get(job['id'])['error'])
        self.assertFalse((self.root / 'catalog-jobs.json').exists())

    def test_cancel_queued_and_running(self):
        queued = self.submit('evaluate_checkpoints')
        self.assertEqual(self.controller.cancel(queued['id'])['status'], 'cancelled')
        running = self.submit()
        self.controller.tick()
        self.controller.cancel(running['id'])
        self.finish(self.controller.store.get(running['id']), 137)
        self.controller.tick()
        self.assertEqual(self.controller.store.get(running['id'])['status'], 'cancelled')

    def test_restart_reconciles_running_lost_and_interrupted_jobs(self):
        alive, lost = self.submit(), self.submit('evaluate_checkpoints')
        self.controller.tick()
        self.docker.states.pop(f'qjob-{lost["id"]}')
        interrupted = self.submit('evaluate_checkpoints')
        self.controller.store.update(interrupted['id'], status='preparing')
        restarted = self.make()
        restarted.tick()
        self.assertEqual(restarted.store.get(alive['id'])['status'], 'running')
        self.assertEqual(restarted.store.get(lost['id'])['error'], 'container lost')
        self.assertEqual(restarted.store.get(interrupted['id'])['error'], 'controller restarted during preparation')

    def test_timeout_stops_the_container(self):
        job = self.submit(resources={'timeout_h': 0.1})
        self.controller.tick()
        self.now[0] += 0.1 * 3600 + 1
        self.controller.tick()
        self.assertEqual(self.controller.store.get(job['id'])['error'], 'timeout')

    def test_http_writes_need_the_platform_token(self):
        token = self.root / 'token'
        client = TestClient(create_app(self.controller, None))
        self.assertEqual(client.post('/api/v1/jobs', json=dict(entrypoint='train_dqn', config=YAML)).status_code, 403)
        token.write_text('secret\n')
        client = TestClient(create_app(self.controller, str(token)))
        body = dict(entrypoint='train_dqn', config=YAML)
        self.assertEqual(client.post('/api/v1/jobs', json=body, headers={'X-Platform-Token': 'wrong'}).status_code, 401)
        created = client.post('/api/v1/jobs', json=body, headers={'X-Platform-Token': 'secret'}).json()
        self.assertEqual(client.get('/api/v1/jobs').json()['counts']['queued'], 1)
        self.assertEqual(client.get(f'/api/v1/jobs/{created["id"]}').json()['run_id'], created['id'])
        self.assertEqual(client.get('/api/v1/jobs/job-missing').status_code, 404)
        self.assertEqual(client.post(f'/api/v1/jobs/{created["id"]}/cancel').status_code, 401)


if __name__ == '__main__':
    unittest.main()
