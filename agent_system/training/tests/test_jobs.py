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

    def test_window_metrics_summarise_recent_behaviour(self):
        script = jobs.load_train_script(str(ROOT / 'scripts' / 'train_dqn.py'))
        result = script.TrainResult(config=script.TrainConfig(), total_episodes=4)
        result.episode_outcomes = ['win', 'loss', 'truncated', 'truncated']
        result.episode_lengths = [30, 40, 401, 400]
        result.episode_learner_moves = [15, 20, 200, 200]
        result.episode_wall_moves = [5, 0, 10, 10]
        result.episode_repeated_positions = [0, 0, 100, 60]
        result.recent_grad_norm = [1.0, 3.0]
        row = script._window_metrics(result, 2, 8.0)
        self.assertEqual((row['win_rate'], row['truncation_rate'], row['wall_ratio'], row['repeat_rate']), (0.0, 1.0, 0.05, 0.4))
        self.assertEqual((row['max_episode_length'], row['grad_norm'], row['episodes_per_s'], row['q_mean']), (401, 2.0, 0.5, None))

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

    def test_manifest_semantics_prefer_diagnosis_then_algorithm(self):
        (self.run_dir / 'experiment' / 'checkpoints').mkdir(parents=True)
        for name, algorithm in (('a.pt', 'ppo_imitation'), ('b.pt', 'double_dqn'), ('c.pt', None)):
            torch.save(dict(model_config=dict(algorithm=algorithm)), self.run_dir / 'experiment' / 'checkpoints' / name)
        directory = self.run_dir / 'experiment' / 'checkpoints'
        known = {jobs.sha256(directory / 'c.pt'): 'policy_logits'}
        rows = jobs.write_manifest(self.run_dir, None, directory, known)
        self.assertEqual([r['file'] for r in rows], ['experiment/checkpoints/a.pt', 'experiment/checkpoints/b.pt', 'experiment/checkpoints/c.pt'])
        self.assertEqual([r['output_semantics'] for r in rows], ['policy_logits', 'q_values', 'policy_logits'])

    def test_gate_roster_pairs_each_candidate_with_every_gate_opponent(self):
        import os
        cwd = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, cwd)
        roster, gate = jobs.gate_roster({'candidates': ['a' * 64, 'a' * 64, 'b' * 64], 'names': {'a' * 64: 'repro-A'}}, '/data')
        ids = [a['id'] for a in roster['agents']]
        self.assertEqual(ids[:2], ['repro-A', 'cand-bbbbbbbbbbbb'])
        self.assertEqual(len(roster['matchups']), 2 * len(gate['agents']))
        self.assertEqual(roster['agents'][0]['algo']['params'], dict(checkpoint='/data/models/' + 'a' * 64 + '.pt', sha256='a' * 64, device='cpu'))
        for bad in ({'candidates': []}, {'candidates': ['a' * 64], 'names': {'a' * 64: 'greedy'}}, {'candidates': ['a' * 64], 'gate': '../x'}):
            with self.assertRaises(ValueError):
                jobs.gate_roster(bad, '/data')

    def test_diagnosis_validates_identities_and_book_location(self):
        events = jobs.Events(self.run_dir)
        for config in ({'checkpoints': []}, {'checkpoints': ['a' * 64], 'book': '/etc/passwd'}):
            with self.assertRaises(ValueError):
                jobs.diagnose_checkpoints(dict(config=config, data_root=self.tmp.name), self.run_dir, events)

    def test_cnn_experiment_requires_a_training_object(self):
        with self.assertRaises(ValueError):
            jobs.train_cnn_experiment(dict(id='j', config='{}'), self.run_dir, jobs.Events(self.run_dir))

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
