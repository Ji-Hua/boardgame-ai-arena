"""In-container Job entrypoints: script loading, manifests and validation (CPU only)."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from agent_system.training import jobs

ROOT = Path(__file__).resolve().parents[3]


class JobEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name)

    def test_metadata_is_complete_for_every_runner(self):
        self.assertEqual(set(jobs.ENTRYPOINTS), set(jobs.RUNNERS))
        for meta in jobs.ENTRYPOINTS.values():
            self.assertIn(meta['format'], ('yaml', 'json'))

    def test_train_script_loads_as_a_module(self):
        script = jobs.load_train_script(str(ROOT / 'scripts' / 'train_dqn.py'))
        cfg = script.TrainConfig(eval_games=3, eval_max_steps=50, seed=1)
        self.assertEqual((cfg.eval_games, cfg.seed), (3, 1))
        self.assertTrue(callable(script.run_evaluation))

    def test_manifest_lists_checkpoint_sha_and_metadata(self):
        (self.run_dir / 'checkpoints').mkdir()
        torch.save(dict(observation_version='dqn_obs_v1', action_count=209, model_config=dict(obs_size=292), extra=1),
                   self.run_dir / 'checkpoints' / 'ep00001.pt')
        rows = jobs.write_manifest(self.run_dir, 'q_values')
        saved = json.loads((self.run_dir / 'manifest.json').read_text())['checkpoints']
        self.assertEqual(rows, saved)
        self.assertEqual(saved[0]['sha256'], jobs.sha256(self.run_dir / 'checkpoints' / 'ep00001.pt'))
        self.assertEqual(saved[0]['metadata']['action_count'], 209)
        self.assertNotIn('extra', saved[0]['metadata'])

    def test_evaluation_rejects_abbreviated_or_unregistered_checkpoints(self):
        events = jobs.Events(self.run_dir)
        for config in ({'checkpoints': []}, {'checkpoints': ['abc']}):
            with self.assertRaises(ValueError):
                jobs.evaluate_checkpoints(dict(config=config, data_root=self.tmp.name), self.run_dir, events)

    def test_failure_is_recorded_as_an_event(self):
        job = dict(id='job-x', entrypoint='evaluate_checkpoints', config={'checkpoints': []}, run_dir=str(self.run_dir),
                   data_root=self.tmp.name, source=dict(commit='c' * 40), resources=dict(gpu=False))
        path = self.run_dir / 'job.json'
        path.write_text(json.dumps(job))
        with self.assertRaises(ValueError):
            jobs.main(str(path))
        events = [json.loads(x)['event'] for x in (self.run_dir / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['started', 'failed'])


if __name__ == '__main__':
    unittest.main()
