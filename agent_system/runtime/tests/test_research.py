import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

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
        self.assertEqual(self.manager.action(*args, "room", 2, {}, legal)["action"], legal[0])
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