import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import os

from agent_system.runtime.service.research import ResearchSessions


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "model.pt"
        self.path.write_bytes(b"test checkpoint")
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.meta = dict(observation_version="dqn_obs_cnn_v1", action_space_version="dqn_action_v1",
            action_count=209, model_config={"model_arch": "cnn"}, output_semantics="policy_logits",
            algorithm="policy_distillation", **{"model_config.algorithm": "policy_distillation"})
        self.row = dict(sha256=self.sha, checkpoint=str(self.path), architecture="cnn",
            metadata=self.meta, output_semantics="policy_logits", name="C-resnet_seed-42")
        self.catalog = self.root / "catalog.json"
        self.write_catalog()
        def factory(definition):
            self.assertEqual(definition.spec.algo_params["sha256"], self.sha)
            self.assertEqual(definition.spec.policy_params, {"k": 1})
            return SimpleNamespace(inference=SimpleNamespace(sha256=self.sha, metadata=self.meta,
                network=SimpleNamespace(parameters=lambda: [])), make_action=lambda state, legal: legal[0])
        self.manager = ResearchSessions(self.catalog, [self.root], capacity=1, factory=factory)

    def write_catalog(self):
        self.catalog.write_text(json.dumps({"models": [self.row]}))

    def test_identity_is_pinned_and_capability_is_room_scoped(self):
        result = self.manager.create(self.sha, "room", 2)
        self.assertEqual(result["identity"]["actual_sha256"], self.sha)
        args = (result["session_id"], result["token"])
        legal = [{"type": "pawn", "target": [4, 7], "player": 2}]
        self.assertEqual(self.manager.action(*args, "room", 2, {"current_player": 2}, legal)["action"], legal[0])
        with self.assertRaises(PermissionError):
            self.manager.action(*args, "other-room", 2, {}, legal)
        with self.assertRaises(PermissionError):
            self.manager.action(args[0], "wrong-token", "room", 2, {}, legal)
        with self.assertRaises(RuntimeError):
            self.manager.create(self.sha, "new-room", 1)
        self.manager.destroy(*args)
        self.assertEqual(len(self.manager.sessions), 0)

    def test_unapproved_sha_and_mismatch(self):
        for sha in ("latest", "0" * 64):
            with self.assertRaises(ValueError):
                self.manager.create(sha, "room", 1)
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.manager.create(self.sha, "room", 1)

    def test_incompatible_and_missing(self):
        self.row["architecture"] = "mlp"
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.manager.create(self.sha, "room", 1)
        self.row["architecture"] = "cnn"
        self.write_catalog()
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.manager.create(self.sha, "room", 1)

    def test_outside_root(self):
        self.manager.roots = (self.root / "other",)
        with self.assertRaisesRegex(ValueError, "outside approved"):
            self.manager.create(self.sha, "room", 1)

    def test_runtime_http_disabled_and_service_authorization(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from agent_system.runtime.service.server import app
        client = TestClient(app)
        payload = dict(sha256=self.sha, room_id="room", seat=2)
        with patch.dict("os.environ", {"QUORIDOR_RESEARCH_SERVICE_KEY": ""}):
            self.assertEqual(client.post("/agent/research/create", json=payload).status_code, 404)
        app.state.research = self.manager
        self.addCleanup(lambda: delattr(app.state, "research"))
        with patch.dict("os.environ", {"QUORIDOR_RESEARCH_SERVICE_KEY": "test-service-key"}):
            self.assertEqual(client.post("/agent/research/create", json=payload).status_code, 403)
            result = client.post("/agent/research/create", json=payload,
                headers={"Authorization": "Bearer test-service-key"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["identity"]["actual_sha256"], self.sha)

    def test_loaded_semantics_and_algorithm_mismatch_fail_closed(self):
        self.meta["output_semantics"] = "q_values"
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.manager.create(self.sha, "room", 1)
        self.meta["output_semantics"] = "policy_logits"
        self.meta["algorithm"] = "double_dqn"
        with self.assertRaisesRegex(ValueError, "algorithm"):
            self.manager.create(self.sha, "room", 1)
        self.assertEqual(self.manager.sessions, {})

    def test_expiry_busy_and_wrong_turn(self):
        result = self.manager.create(self.sha, "room", 2)
        args = (result["session_id"], result["token"], "room", 2, {"current_player": 1}, [])
        with self.assertRaisesRegex(ValueError, "current player"):
            self.manager.action(*args)
        self.manager.execution.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.manager.action(*args)
        finally:
            self.manager.execution.release()
        self.manager.sessions[result["session_id"]]["expires"] = 0
        with self.assertRaises(PermissionError):
            self.manager.action(*args)


@unittest.skipUnless(os.environ.get("RESEARCH_PLAY_SMOKE_CATALOG"), "External approved research catalog required")
class RealResearchTests(unittest.TestCase):
    def test_d2_architectures_and_dqn_actual_load_and_action(self):
        import torch
        from quoridor_engine import RuleEngine
        from agent_system.runtime.service.agents.greedy_agent import _action_to_engine, _build_state
        torch.set_num_threads(1)
        catalog = Path(os.environ["RESEARCH_PLAY_SMOKE_CATALOG"])
        rows = json.loads(catalog.read_text())["models"]
        names = {"A-baseline_seed-42", "B-deep_seed-42", "C-resnet_seed-42", "D-wide_seed-42"}
        selected = [row for row in rows if row["name"] in names
            or row["sha256"] == "09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4"
            or "cnn_observation_2k" in row["name"].split(" / ")]
        self.assertEqual(len(selected), 6)
        manager = ResearchSessions(catalog, ["/home/jihua/quoridor-data"])
        engine = RuleEngine.standard()
        state = dict(current_player=1, pawns={"1": {"row": 4, "col": 0}, "2": {"row": 4, "col": 8}},
            walls_remaining={"1": 10, "2": 10}, wall_state={})
        raw = _build_state(state)
        from agent_system.runtime.service.agents.greedy_agent import _engine_action_to_wire
        legal = [_engine_action_to_wire(action, 1) for action in engine.legal_actions(raw)]
        results = []
        for row in selected:
            with self.subTest(model=row["name"]):
                session = manager.create(row["sha256"], row["sha256"], 1)
                result = manager.action(session["session_id"], session["token"], row["sha256"], 1, state, legal)
                engine.apply_action(raw, _action_to_engine(result["action"]))
                self.assertEqual(result["identity"]["actual_sha256"], row["sha256"])
                self.assertEqual(result["identity"]["output_semantics"], row["output_semantics"])
                results.append(dict(identity=result["identity"], action=result["action"], engine_accepted=True))
                manager.destroy(session["session_id"], session["token"])
        evidence = os.environ.get("RESEARCH_PLAY_SMOKE_OUTPUT")
        if evidence:
            Path(evidence).write_text(json.dumps(results, indent=2))