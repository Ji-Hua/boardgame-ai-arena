"""SPECIFIED: Rust replay identity/integrity and read-only Arena evidence coverage.

Set ARENA_REPLAY_TEST_ROOT to an external arena directory to validate every
C-resnet game, optionally ARENA_REPLAY_TEST_AGENT for one exact roster ID and
ARENA_REPLAY_TEST_EXPECTED_GAMES for its expected count. No artifacts are written.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.replay import _dependencies, _stable_hash, arena_replay


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter, cls.Player, cls.RuleEngine, _, snapshot = _dependencies()
        cls.snapshot = staticmethod(snapshot)

    def setUp(self):
        self.engine = self.RuleEngine.standard()
        self.sha = "a" * 64
        self.checkpoint = dict(filename="update012000.pt", sha256=self.sha, episode=12000,
                               step=12000, progress_type="gradient_updates", model_type="policy",
                               artifact_ref="runs:/registered/checkpoint")
        state = self.engine.initial_state()
        # An odd-length prefix is deliberate: evaluated play must start as P2.
        prefix = [{"player": 1, "type": "horizontal", "target": [0, 0]}]
        state = self.engine.apply_action(state, self.adapter._action_to_engine(prefix[0]))
        wire = self.adapter._serialize_state(state)
        self.manifest = dict(protocol="quoridor-paired-opening-v1", roster={"agents": [
            dict(id="policy", algo=dict(type="dqn", params={"sha256": self.sha})),
            dict(id="minimax", algo=dict(type="minimax", params={"depth": 2})),
        ]}, book=[dict(id=0, actions=prefix, state=wire, state_hash=_stable_hash(wire))])
        actions = [{"player": 2, "type": "vertical", "target": [6, 6]},
                   {"player": 1, "type": "pawn", "target": [4, 1]},
                   {"player": 2, "type": "pawn", "target": [4, 7]}]
        for a in actions:
            state = self.engine.apply_action(state, self.adapter._action_to_engine(a))
        self.block = dict(a="policy", b="minimax", block_id="fixture", opening_id=0, control=False,
                          opening_hash=_stable_hash(wire), games=[dict(
                              p1="policy", p2="minimax", winner=None, steps=len(actions), actions=actions,
                              final_state=self.adapter._serialize_state(state), termination="deterministic_cycle",
                              offender=None, error=None)])
        self.seal()

    def seal(self):
        self.manifest["manifest_sha256"] = _stable_hash({k: v for k, v in self.manifest.items()
                                                       if k != "manifest_sha256"})
        self.block["manifest_sha256"] = self.manifest["manifest_sha256"]
        for game in self.block["games"]:
            game["trajectory_hash"] = _stable_hash([self.block["opening_hash"], game["actions"]])

    def replay(self, **kwargs):
        return arena_replay(self.manifest, self.block, kwargs.get("index", 0),
                            kwargs.get("agent", "policy"), "mlflow:registered", self.checkpoint)

    def test_snapshots_opening_turn_and_schema(self):
        before = deepcopy((self.manifest, self.block, self.checkpoint))
        result = self.replay()
        self.assertEqual(before, (self.manifest, self.block, self.checkpoint))
        self.assertEqual(len(result["states"]), 4)
        self.assertEqual(result["states"][0]["current_player"], 2)
        self.assertEqual(result["states"][0]["h_walls"], [[0, 0]])
        self.assertEqual(result["states"][1]["v_walls"], [[6, 6]])
        self.assertEqual(result["states"][-1]["p1"], [4, 1])
        self.assertEqual(result["states"][-1]["p2"], [4, 7])
        self.assertEqual([a["action_type"] for a in result["actions"]], ["vwall", "pawn", "pawn"])
        self.assertEqual([a["is_dqn_action"] for a in result["actions"]], [False, True, False])
        self.assertEqual(result["actions"][0]["actor_name"], "minimax")
        self.assertEqual(result["actions"][0]["turn_index"], 0)
        self.assertEqual(result["eval_opponent"], dict(name="minimax", type="minimax", depth=2))
        self.assertEqual(result["termination_reason"], "deterministic_cycle")
        self.assertEqual(result["result_from_dqn_perspective"], "draw")
        self.assertIsNone(result["episode"])
        self.assertEqual(result["step"], 12000)
        self.assertEqual(result["source"]["checkpoint_sha256"], self.sha)
        self.assertNotIn("action_id", result["actions"][0])

    def test_standard_control(self):
        self.block.update(opening_id=-1, control=True)
        state = self.engine.initial_state()
        self.block["opening_hash"] = _stable_hash(self.adapter._serialize_state(state))
        self.block["games"][0].update(actions=[], steps=0, final_state=self.adapter._serialize_state(state))
        self.seal()
        result = self.replay()
        self.assertEqual(result["states"], [{**self.snapshot(state), "current_player": 1}])

    def test_forfeit_preserves_outcome_without_engine_winner(self):
        game = self.block["games"][0]
        for termination, illegal in (("illegal_action", 1), ("agent_error", 0)):
            with self.subTest(termination=termination):
                game.update(winner="minimax", termination=termination, offender="policy", error="recorded error")
                result = self.replay()
                self.assertEqual(result["winner"], "P2")
                self.assertEqual(result["result_from_dqn_perspective"], "loss")
                self.assertEqual(result["illegal_acts"], illegal)
                self.assertEqual(result["source"]["offender"], "policy")
                self.assertEqual(result["game_length"], 3)

    def test_selected_agent_as_p2(self):
        self.manifest["roster"]["agents"][1]["algo"]["params"]["sha256"] = self.sha
        self.seal()
        result = self.replay(agent="minimax")
        self.assertEqual(result["dqn_player_id"], "P2")
        self.assertEqual(result["opponent_player_id"], "P1")
        self.assertEqual([a["is_dqn_action"] for a in result["actions"]], [True, False, True])

    def test_goal_winner(self):
        self.block.update(opening_id=-1, control=True)
        state = self.engine.initial_state()
        self.block["opening_hash"] = _stable_hash(self.adapter._serialize_state(state))
        # A fixed legal recorded trajectory, not an agent-generated match.
        targets = [(1, 4, 1), (2, 3, 8), (1, 4, 2), (2, 3, 7),
                   (1, 4, 3), (2, 3, 6), (1, 4, 4), (2, 3, 5),
                   (1, 4, 5), (2, 3, 4), (1, 4, 6), (2, 3, 3),
                   (1, 4, 7), (2, 3, 2), (1, 4, 8)]
        actions = [dict(player=p, type="pawn", target=[x, y]) for p, x, y in targets]
        for a in actions:
            state = self.engine.apply_action(state, self.adapter._action_to_engine(a))
        final = self.adapter._serialize_state(state)
        final.update(game_over=True, winner=1)
        self.block["games"][0].update(actions=actions, steps=len(actions), final_state=final,
                                       winner="policy", termination="goal")
        self.seal()
        result = self.replay()
        self.assertEqual(result["winner"], "P1")
        self.assertEqual(result["result_from_dqn_perspective"], "win")
        self.block["games"][0]["winner"] = "minimax"
        with self.assertRaisesRegex(ValueError, "goal winner"):
            self.replay()
        self.block["games"][0]["winner"] = "policy"
        self.block["games"][0]["actions"].append(dict(player=2, type="pawn", target=[3, 1]))
        self.block["games"][0]["steps"] += 1
        self.seal()
        with self.assertRaisesRegex(ValueError, "after engine goal"):
            self.replay()

    def test_hwall_and_optional_action_id_compatibility(self):
        game = self.block["games"][0]
        game["actions"][0].update(type="horizontal", target=[6, 6], action_id=135)
        state = self.engine.initial_state()
        for action in self.manifest["book"][0]["actions"] + game["actions"]:
            state = self.engine.apply_action(state, self.adapter._action_to_engine(action))
        game["final_state"] = self.adapter._serialize_state(state)
        self.seal()
        result = self.replay()
        self.assertEqual(result["actions"][0]["action_type"], "hwall")
        self.assertEqual(result["actions"][0]["action_id"], 135)
        self.assertIn([6, 6], result["states"][1]["h_walls"])

    def test_corruption_is_rejected(self):
        cases = {
            "manifest SHA": lambda: self.manifest.update(seed=99),
            "block manifest": lambda: self.block.update(manifest_sha256="b" * 64),
            "checkpoint identity": lambda: self.checkpoint.update(sha256="b" * 64),
            "missing checkpoint SHA": lambda: self.checkpoint.update(sha256=None),
            "opening hash": lambda: self.block.update(opening_hash="b" * 64),
            "trajectory hash": lambda: self.block["games"][0].update(trajectory_hash="b" * 64),
            "truncated actions": lambda: self.block["games"][0]["actions"].pop(),
            "missing actions": lambda: self.block["games"][0].pop("actions"),
            "game seats": lambda: self.block["games"][0].update(p2="stranger"),
            "unknown winner": lambda: self.block["games"][0].update(winner="stranger"),
            "missing final": lambda: self.block["games"][0].pop("final_state"),
            "false goal": lambda: self.block["games"][0].update(termination="goal", winner="policy"),
        }
        for name, corrupt in cases.items():
            with self.subTest(name=name):
                self.setUp()
                corrupt()
                with self.assertRaises(ValueError):
                    self.replay()

    def test_semantic_corruption_with_valid_hashes(self):
        for corruption in ("player", "illegal", "opening state", "opening prefix", "control",
                   "control hash", "duplicate identity"):
            with self.subTest(corruption=corruption):
                self.setUp()
                if corruption == "player":
                    self.block["games"][0]["actions"][0]["player"] = 1
                elif corruption == "illegal":
                    self.block["games"][0]["actions"][0].update(type="pawn", target=[0, 0])
                elif corruption == "opening state":
                    self.manifest["book"][0]["state"]["current_player"] = 1
                elif corruption == "opening prefix":
                    self.manifest["book"][0]["actions"] = []
                elif corruption == "control":
                    self.block.update(control=True)
                elif corruption == "control hash":
                    self.block.update(control=True, opening_id=-1, opening_hash="b" * 64)
                else:
                    self.manifest["roster"]["agents"].append(self.manifest["roster"]["agents"][0])
                self.seal()
                with self.assertRaises(ValueError):
                    self.replay()

    def test_every_raw_final_field_is_checked(self):
        for key, child in (("current_player", None), ("pawns", "1"), ("walls_remaining", "1"),
                           ("wall_state", "horizontal_edges"), ("wall_state", "vertical_edges"),
                           ("wall_state", "horizontal_heads"), ("wall_state", "vertical_heads"),
                           ("game_over", None), ("winner", None)):
            with self.subTest(key=key, child=child):
                self.setUp()
                final = self.block["games"][0]["final_state"]
                if key == "pawns":
                    final[key][child]["row"] = 0
                elif child is not None:
                    final[key][child] += 1
                else:
                    final[key] = 1 if key == "winner" else (not final[key] if key == "game_over" else 3 - final[key])
                with self.assertRaisesRegex(ValueError, "Final state mismatch"):
                    self.replay()

    def test_bad_index_and_unknown_agent(self):
        for index in (-1, 1, True, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.replay(index=index)
        with self.assertRaises(ValueError):
            self.replay(agent="unknown")

    def test_dependencies_do_not_import_torch_or_materialize_agents(self):
        script = """
import importlib.abc, sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch' or fullname.endswith('.agents.loader'):
            raise AssertionError('Forbidden replay dependency: ' + fullname)
sys.meta_path.insert(0, Forbidden())
from backend.test_replay import ReplayTests
test = ReplayTests()
test.setUpClass()
test.setUp()
test.replay()
assert 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get("ARENA_REPLAY_TEST_ROOT"), "Set ARENA_REPLAY_TEST_ROOT for original evidence")
class ExternalReplayTests(unittest.TestCase):
    def test_all_original_c_resnet_games_read_only(self):
        root = Path(os.environ["ARENA_REPLAY_TEST_ROOT"])
        manifest = json.loads((root / "manifest.json").read_text())
        selected = os.environ.get("ARENA_REPLAY_TEST_AGENT")
        agents = {a["id"]: a for a in manifest["roster"]["agents"]
                  if a["id"] == selected or (not selected and a["id"].startswith("C-resnet"))}
        self.assertTrue(agents, "No selected C-resnet roster identity")
        counts = Counter()
        terminations = Counter()
        for path in sorted((root / "blocks").glob("*.json.gz")):
            original = path.read_bytes()
            block = json.loads(gzip.decompress(original))
            for agent in agents.keys() & {block["a"], block["b"]}:
                params = agents[agent]["algo"]["params"]
                checkpoint = dict(filename=Path(params["checkpoint"]).name, sha256=params["sha256"],
                                  episode=None, step=12000, progress_type="gradient_updates",
                                  model_type="policy", artifact_ref=params["checkpoint"])
                for index, game in enumerate(block["games"]):
                    with self.subTest(block=path.name, agent=agent, game=index):
                        result = arena_replay(manifest, block, index, agent, "external-test", checkpoint)
                        self.assertEqual(len(result["states"]), game["steps"] + 1)
                        self.assertEqual(len(result["actions"]), game["steps"])
                        self.assertEqual(result["termination_reason"], game["termination"])
                        self.assertEqual(result["result_from_dqn_perspective"], "draw" if game["winner"] is None
                                         else ("win" if game["winner"] == agent else "loss"))
                        counts[agent] += 1
                        terminations[game["termination"]] += 1
            self.assertEqual(original, path.read_bytes())
        for agent in agents:
            self.assertGreater(counts[agent], 0)
            if os.environ.get("ARENA_REPLAY_TEST_EXPECTED_GAMES"):
                self.assertEqual(counts[agent], int(os.environ["ARENA_REPLAY_TEST_EXPECTED_GAMES"]))
        print(f"Original Arena replay counts: {dict(counts)}; terminations: {dict(terminations)}")


if __name__ == "__main__":
    unittest.main()