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
        return ['smoke.yaml', 'notes.txt']

    def show(self, commit, path):
        if not path.endswith('smoke.yaml'):
            raise KeyError(path)
        return YAML


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
        self.assertEqual(self.controller.entrypoints()['entrypoints']['train_dqn']['templates'], ['smoke.yaml'])
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
        self.assertEqual((self.root / 'jobs' / job['id'] / 'stdout.log').read_text(), 'log line\n')
        self.assertIn(f'qjob-{job["id"]}', self.docker.removed)

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
