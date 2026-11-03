import json
from pathlib import Path
import tempfile
import unittest

from agent_system.games.contracts import KUHN_POKER, QUORIDOR
from agent_system.training.jobs import main
from agent_system.training.openspiel_job import PolicyAgent, validate_config

class OpenSpielJobTests(unittest.TestCase):
    def run_job(self, root, algorithm="cfr", iterations=200, eval_every=20):
        run = root / f"{algorithm}-{eval_every}"
        spec = dict(id="test", entrypoint="train_openspiel", project_id="poker",
            game_ref=KUHN_POKER.wire(), source=dict(commit="c"*40),
            image="test-image", run_dir=str(run),
            config=dict(algorithm=algorithm, iterations=iterations, eval_every=eval_every, seed=42))
        path = root / "job.json"; path.write_text(json.dumps(spec))
        self.assertEqual(main(str(path)), 0)
        return run

    def test_cfr_exports_reloads_and_reduces_exploitability(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_job(Path(tmp))
            rows = [json.loads(x) for x in (run/"research-metrics.jsonl").read_text().splitlines()]
            self.assertLess(rows[-1]["exploitability"], rows[0]["exploitability"]/5)
            self.assertEqual(rows[-1]["step"], 200)
            policy = json.loads((run/"policy.json").read_text())
            self.assertEqual(len(policy["policies"]), 12)
            self.assertEqual(policy["policy_kind"], "average_policy")
            self.assertFalse((run/"manifest.json").exists())
            smoke = json.loads((run/"policy-smoke.json").read_text())
            self.assertEqual([r["policy_role"] for r in smoke], ["p1", "p2"])
            PolicyAgent.load(run/"policy.json")
            policy["policies"][0]["probabilities"]["0"] = 99
            with self.assertRaisesRegex(ValueError, "integrity"):
                PolicyAgent(policy)
            policy["game_ref"] = QUORIDOR.wire()
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                PolicyAgent(policy)

    def test_q_learning_reproducible_independent_of_eval_frequency(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.run_job(Path(tmp), "tabular_q_learning", 100, 20)
            b = self.run_job(Path(tmp), "tabular_q_learning", 100, 50)
            pa, pb = [json.loads((p/"policy.json").read_text()) for p in (a,b)]
            self.assertEqual(pa["policies"], pb["policies"])
            self.assertEqual(pa["policy_kind"], "greedy_q_policy")
            self.assertEqual(pa["progress_unit"], "episode")
            self.assertTrue(any(set(r["probabilities"].values()) != {0.5} for r in pa["policies"]))

    def test_invalid_configs_rejected_before_run(self):
        for value in (dict(iterations=True), dict(iterations=0), dict(iterations=100001),
                      dict(seed=-1), dict(eval_every=0), dict(algorithm="ppo"), dict(game="other")):
            with self.assertRaises(ValueError):
                validate_config(value)
