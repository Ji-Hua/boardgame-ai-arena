"""MLflow REST logging of finished Jobs against a stub tracking server."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
import urllib.parse

from apps.jobs.mlflow_log import MlflowLog, rows_to_metrics

COMMIT = 'c' * 40


class Stub(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def _reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        Stub.calls.append(('GET', url.path, urllib.parse.parse_qs(url.query)))
        self._reply(404, dict(error_code='RESOURCE_DOES_NOT_EXIST'))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Stub.calls.append(('POST', self.path, body))
        path = self.path.rsplit('/mlflow/', 1)[1]
        self._reply(200, dict(experiment_id='7') if path == 'experiments/create' else
                    dict(run=dict(info=dict(run_id='r1'))) if path == 'runs/create' else {})


class MlflowLogTests(unittest.TestCase):
    def setUp(self):
        Stub.calls = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)

    def test_metrics_keep_finite_numbers_and_eval_win_rates(self):
        rows = [dict(type='train', episode=5, loss=0.5, win_rate=0.2, flag=True, name='x', bad=float('nan'))]
        evals = [dict(episode=5, opponent_name='greedy', win_rate=0.25), dict(opponent_name='greedy', win_rate=0.5)]
        metrics = rows_to_metrics(rows, evals, 1000)
        self.assertEqual(sorted((m['key'], m['step']) for m in metrics), [('eval/greedy/win_rate', 5), ('loss', 5), ('win_rate', 5)])
        twice = rows_to_metrics([dict(episode=5, loss=0.5), dict(episode=5, loss=0.5)], [], 1000)
        self.assertEqual([m['timestamp'] for m in twice], [1000, 1001])  # distinct MLflow keys

    def test_a_failed_write_deletes_the_new_run(self):
        job = dict(id='job-2', name='x', entrypoint='train_dqn', experiment='e', status='succeeded', created=1.0,
                   started=None, finished=2.0, spec=dict(source=dict(commit=COMMIT)))
        logger = MlflowLog(f'http://127.0.0.1:{self.server.server_port}')
        logger._fill = lambda *a: (_ for _ in ()).throw(RuntimeError('boom'))
        with self.assertRaises(RuntimeError):
            logger.log(job, '{}', [], [])
        self.assertTrue(Stub.calls[-1][1].endswith('runs/delete'))

    def test_finished_job_becomes_a_tagged_run(self):
        job = dict(id='job-1', name='smoke', entrypoint='train_dqn', experiment=None, status='failed', created=10.0,
                   started=11.0, finished=20.0, registered=['a' * 64], spec=dict(source=dict(commit=COMMIT)))
        rows = [dict(episode=i, loss=0.1 * i) for i in range(1, 1201)]
        result = MlflowLog(f'http://127.0.0.1:{self.server.server_port}').log(job, 'episodes: 5\nopponent:\n  type: greedy\n', rows, [])
        self.assertEqual(result, dict(run_id='r1', experiment_id='7', experiment='platform/jobs', metrics=1200))
        create = next(c[2] for c in Stub.calls if c[1].endswith('runs/create'))
        tags = {t['key']: t['value'] for t in create['tags']}
        self.assertEqual((tags['quoridor.job_id'], tags['quoridor.source_commit'], create['start_time']), ('job-1', COMMIT, 11000))
        batches = [c[2] for c in Stub.calls if c[1].endswith('runs/log-batch')]
        self.assertEqual([len(b.get('metrics', [])) + len(b.get('params', [])) for b in batches], [3, 1000, 200])
        self.assertIn(dict(key='opponent.type', value='greedy'), batches[0]['params'])
        update = next(c[2] for c in Stub.calls if c[1].endswith('runs/update'))
        self.assertEqual((update['status'], update['end_time']), ('FAILED', 20000))


if __name__ == '__main__':
    unittest.main()
