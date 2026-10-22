"""Platform API contracts: original evidence, Rust positions and exact model SHAs."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlsplit

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import create_app
from backend.platform_context import analysis_context, canonical_hash
from backend.platform_models import ModelIndex
from backend.replay import _dependencies, _stable_hash, arena_replay


class Provider:
    source = "mlflow"

    def __init__(self, replay):
        self.payload = replay
        self.path = "opaque/key"
        self.rows = [dict(replay_path=self.path, checkpoint_sha=replay["source"]["checkpoint_sha256"],
                          evaluation_id="recorded-evaluation", game_index=0)]
        self.reads = []

    def replay(self, run_id, path):
        self.reads.append((run_id, path))
        if path != self.path:
            raise KeyError(path)
        return deepcopy(self.payload)

    def replays(self, run_id):
        return dict(run_id="mlflow:registered", replays=deepcopy(self.rows), warnings=["replay warning"])

    def benchmarks(self, run_id):
        return dict(run_id="mlflow:registered", benchmarks=[dict(checkpoint_sha=self.rows[0].get("checkpoint_sha"),
            games=2, wins=1, losses=1, draws=0, win_rate=0.5)], warnings=["benchmark warning"])


class Service:
    def __init__(self, provider):
        self.provider = provider

    def provider_for(self, run_id):
        if run_id not in ("mlflow:registered", "alias"):
            raise KeyError(run_id)
        return self.provider


class PlatformContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter, cls.Player, cls.RuleEngine, _, snapshot = _dependencies()
        cls.snapshot = staticmethod(snapshot)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sha = hashlib.sha256(b"checkpoint").hexdigest()
        self.other_sha = hashlib.sha256(b"other checkpoint").hexdigest()
        self.metadata = dict(observation_version="dqn_obs_cnn_v1", action_space_version="dqn_action_v1",
            action_count=209, model_config=dict(model_arch="cnn", algorithm="policy_distillation"))
        models = []
        for digest, content, name in ((self.sha, b"checkpoint", "same-display"),
                                      (self.other_sha, b"other checkpoint", "other")):
            path = self.root / (digest + ".pt")
            path.write_bytes(content)
            models.append(dict(id=name, sha256=digest, checkpoint=str(path), metadata=self.metadata))
        self.models = models
        self.catalog = self.root / "catalog.json"
        self.write_catalog()
        (self.root / "registry.json").write_text('{"models": []}')
        (self.root / "agent_defs").mkdir()
        self.index = ModelIndex(data_root=self.root, catalog_paths=[self.catalog])
        self.provider = Provider(self.make_replay())
        self.service = Service(self.provider)
        self.client = TestClient(create_app(unified_run_service=self.service, model_index=self.index))

    def write_catalog(self):
        self.catalog.write_text(json.dumps(dict(models=self.models)))

    def make_replay(self, seat=1, seed=None):
        engine = self.RuleEngine.standard()
        state = engine.initial_state()
        # A head at bit 63 and vertical edges above 2**53 must survive as coordinates.
        prefix = [dict(player=1, type="horizontal", target=[7, 7])]
        state = engine.apply_action(state, self.adapter._action_to_engine(prefix[0]))
        self.assertGreater(state.horizontal_heads, 2**53)
        if seed is not None:
            rng = random.Random(seed)
            for _ in range(2 * (seed % 3)):
                action = rng.choice(list(engine.legal_actions(state)))
                prefix.append(dict(player=1 if action.player == self.Player.P1 else 2,
                    type="pawn" if action.kind == "MovePawn" else
                        ("horizontal" if action.coordinate_kind == "Horizontal" else "vertical"),
                    target=[action.target_x, action.target_y]))
                state = engine.apply_action(state, action)
        opening = self.adapter._serialize_state(state)
        opening_hash = _stable_hash(opening)
        manifest = dict(roster={"agents": [dict(id="policy", algo=dict(type="dqn", params={"sha256": self.sha})),
            dict(id="opponent", algo=dict(type="minimax", params={"sha256": self.sha, "depth": 1}))]},
            book=[dict(id=0, state=opening, state_hash=opening_hash, actions=prefix)])
        manifest["manifest_sha256"] = _stable_hash(manifest)
        if seed is None:
            actions = [dict(player=2, type="vertical", target=[6, 6]),
                       dict(player=1, type="pawn", target=[4, 1])]
            for action in actions:
                state = engine.apply_action(state, self.adapter._action_to_engine(action))
        else:
            actions = []
            for _ in range(4):
                action = rng.choice(list(engine.legal_actions(state)))
                wire = dict(player=1 if action.player == self.Player.P1 else 2,
                    type="pawn" if action.kind == "MovePawn" else
                        ("horizontal" if action.coordinate_kind == "Horizontal" else "vertical"),
                    target=[action.target_x, action.target_y])
                actions.append(wire)
                state = engine.apply_action(state, action)
        block = dict(a="policy", b="opponent", block_id="not-an-identity", opening_id=0, control=False,
            opening_hash=opening_hash, manifest_sha256=manifest["manifest_sha256"], games=[dict(
                p1="policy", p2="opponent", steps=len(actions), actions=actions, winner=None,
                final_state=self.adapter._serialize_state(state), termination="ply_limit",
                trajectory_hash=_stable_hash([opening_hash, actions]))])
        return arena_replay(manifest, block, 0, "policy" if seat == 1 else "opponent", "mlflow:registered",
                            dict(filename="not-an-identity.pt", sha256=self.sha))

    def context(self, **kwargs):
        return analysis_context(self.service, self.index, run_id="alias", replay_path=self.provider.path, **kwargs)

    def test_routes_and_existing_evaluation_shapes(self):
        listing = self.client.get("/api/platform/models").json()
        self.assertEqual(listing["count"], 2)
        self.assertEqual({row["sha256"] for row in listing["models"]}, {self.sha, self.other_sha})
        detail = self.client.get("/api/platform/models/" + self.sha).json()
        self.assertTrue(detail["sha_verified"])
        self.assertFalse(detail["actual_load_verified"])
        response = self.client.get("/api/platform/runs/alias/evaluations")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["run_id"], "mlflow:registered")
        self.assertEqual(result["benchmarks"], self.provider.benchmarks("alias")["benchmarks"])
        self.assertEqual(result["replays"], self.provider.rows)
        self.assertEqual(result["warnings"], ["benchmark warning", "replay warning"])

    def test_p1_p2_coordinates_stocks_actual_turn_and_high_masks(self):
        for seat in (1, 2):
            self.provider.payload = self.make_replay(seat=seat)
            before = deepcopy(self.provider.payload)
            for ply in range(3):
                result = self.context(ply=ply)
                self.assertEqual(result["position"]["turn"], 2 if ply % 2 == 0 else 1)
                self.assertEqual(result["position"]["pawns"], [[4, int(ply == 2)], [4, 8]])
                self.assertEqual(result["position"]["walls_remaining"], [9, 10 if ply == 0 else 9])
                self.assertIn(dict(x=7, y=7, orientation="horizontal"), result["position"]["walls"])
                if ply:
                    self.assertIn(dict(x=6, y=6, orientation="vertical"), result["position"]["walls"])
                self.assertEqual(result["actual_action"], None if ply == 2 else
                    (dict(player=2, type="vertical", target=[6, 6]) if ply == 0 else
                     dict(player=1, type="pawn", target=[4, 1])))
                self.assertEqual(result["position_hash"], canonical_hash(result["position"]))
                self.assertEqual(result["recorded_model"]["sha256"], self.sha)
                self.assertEqual(result["analysis_model"]["sha256"], self.sha)
                self.assertEqual(result["analysis_model"]["output_semantics"], "policy_logits")
                # No bitmasks or JS-unsafe integers in the Position transport.
                self.assertNotIn("heads", json.dumps(result["position"]))
            self.assertEqual(before, self.provider.payload)

    def test_random_opening_prefix_and_every_state_roundtrip(self):
        for seed in range(10):
            self.provider.payload = self.make_replay(seed=seed)
            for ply, snapshot in enumerate(self.provider.payload["states"]):
                result = self.context(ply=ply)
                self.assertEqual(result["position"]["pawns"], [snapshot["p1"], snapshot["p2"]])
                self.assertEqual(result["position"]["turn"], snapshot["current_player"])
                self.assertEqual(len(result["position"]["walls"]), len(snapshot["h_walls"]) + len(snapshot["v_walls"]))

    def test_model_override_keeps_recorded_identity_and_context_consistent(self):
        original = self.context(ply=1)
        override = self.context(ply=1, model_sha=self.other_sha)
        for key in ("position", "position_hash", "source", "actual_action", "recorded_model"):
            self.assertEqual(original[key], override[key])
        self.assertEqual(override["analysis_model"]["sha256"], self.other_sha)
        self.assertEqual(override["source"]["run"], dict(source="mlflow", run_id="mlflow:registered"))
        self.assertIn("mlflow%3Aregistered", override["return_route"])

    def test_model_context_excludes_large_catalog_records_and_evidence(self):
        compact = dict(id=self.sha, sha256=self.sha, aliases=["same-display"],
            architecture="cnn", output_semantics="policy_logits", sha_verified=True,
            analyzable=True, actual_load_verified=False)
        large_evidence = "catalog-only-evidence-" * 10000
        model = dict(compact, metadata={"notes": large_evidence},
            catalog_record={"notes": large_evidence}, sources=[large_evidence],
            files=[large_evidence], evidence=[large_evidence],
            checkpoints=[{"path": large_evidence}], registry=[large_evidence],
            run_refs=[large_evidence], agent_specs=[large_evidence])
        before = deepcopy(model)
        with patch.object(self.index, "detail", return_value=model) as detail:
            response = self.client.get("/api/platform/context", params=dict(
                run_id="alias", replay_path=self.provider.path, ply=1))
        self.assertEqual(response.status_code, 200, response.text)
        detail.assert_called_once_with(self.sha, verify_sha=True)
        context = response.json()
        self.assertEqual(context["analysis_model"], compact)
        self.assertEqual(context["recorded_model"], {"sha256": self.sha})
        self.assertNotIn("catalog-only-evidence-", response.text)
        self.assertEqual(model, before)
        encoded = base64.urlsafe_b64encode(json.dumps(context, ensure_ascii=False,
            separators=(",", ":")).encode()).rstrip(b"=")
        self.assertLess(len(encoded), 8192)

    def test_real_context_base64_navigation_envelope_is_bounded(self):
        for model_sha in (None, self.other_sha):
            with self.subTest(model_sha=model_sha):
                params = dict(run_id="alias", replay_path=self.provider.path, ply=1)
                if model_sha:
                    params["model_sha"] = model_sha
                response = self.client.get("/api/platform/context", params=params)
                self.assertEqual(response.status_code, 200, response.text)
                context = response.json()
                self.assertEqual(context["analysis_model"]["sha256"], model_sha or self.sha)
                self.assertEqual(context["source"]["replay_path"], self.provider.path)
                self.assertEqual(context["source"]["ply"], 1)
                self.assertEqual(context["position_hash"], canonical_hash(context["position"]))
                # Match the browser's UTF-8 JSON -> unpadded URL-safe base64 transport.
                encoded = base64.urlsafe_b64encode(json.dumps(context, ensure_ascii=False,
                    separators=(",", ":")).encode()).rstrip(b"=")
                self.assertLess(len(encoded), 8192)
                self.assertEqual(json.loads(base64.urlsafe_b64decode(
                    encoded + b"=" * (-len(encoded) % 4))), context)

    def test_return_route_preserves_replay_path_ply_and_optional_model(self):
        self.provider.path = "eval replays/棋局+one%2Ftwo?seat=2&tag=#opening.json"
        self.provider.rows[0]["replay_path"] = self.provider.path
        for ply in (0, 1, 2):
            for model_sha in (None, self.other_sha):
                with self.subTest(ply=ply, model_sha=model_sha):
                    context = self.context(ply=ply, model_sha=model_sha)
                    route = urlsplit(context["return_route"])
                    self.assertEqual(unquote(route.path), "/replays/mlflow:registered")
                    self.assertEqual(route.fragment, "")
                    expected = {"replay": [self.provider.path], "ply": [str(ply)]}
                    if model_sha:
                        expected["model_sha"] = [model_sha]
                    self.assertEqual(parse_qs(route.query, strict_parsing=True), expected)
                    self.assertEqual(context["source"]["replay_path"], self.provider.path)
                    self.assertEqual(context["source"]["ply"], ply)

    def test_stable_ids_do_not_depend_on_path_or_display_name(self):
        first = self.context()
        self.provider.path = "entirely/different"
        self.provider.rows[0]["replay_path"] = self.provider.path
        self.provider.payload["agent_id"] = "renamed"
        self.provider.payload["checkpoint"] = "renamed.pt"
        second = self.context()
        for key in ("evaluation", "game"):
            self.assertEqual(first["source"][key], second["source"][key])
        self.assertEqual(first["position_hash"], second["position_hash"])
        self.provider.payload["source"]["manifest_sha256"] = "f" * 64
        self.assertNotEqual(first["source"]["evaluation"]["id"], self.context()["source"]["evaluation"]["id"])

    def test_duplicate_sha_aliases_merge_but_conflicts_never_fallback(self):
        self.models.append(dict(self.models[0], id="another-alias"))
        self.write_catalog()
        self.assertEqual(self.context()["analysis_model"]["aliases"], ["another-alias", "same-display"])
        self.assertEqual(self.client.get("/api/platform/models").json()["count"], 2)
        self.models[-1]["metadata"] = dict(self.metadata, model_config=dict(model_arch="mlp"))
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "unsupported, conflicting"):
            self.context()

    def test_invalid_model_missing_unsupported_and_mismatch_fail_closed(self):
        for value in (self.sha[:12], "same-display", self.sha.upper(), "../model"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.context(model_sha=value)
        with self.assertRaises(KeyError):
            self.context(model_sha="f" * 64)
        (self.root / (self.sha + ".pt")).write_bytes(b"corruption")
        with self.assertRaises(ValueError):
            self.context()
        # Explicit override is permitted, but never substitutes the recorded hash.
        self.assertEqual(self.context(model_sha=self.other_sha)["recorded_model"]["sha256"], self.sha)
        (self.root / (self.other_sha + ".pt")).unlink()
        with self.assertRaises(ValueError):
            self.context(model_sha=self.other_sha)

    def test_invalid_ply_and_replay_path_api(self):
        base = dict(run_id="alias", replay_path=self.provider.path)
        for ply in (-1, 3, "bad", "1.5"):
            response = self.client.get("/api/platform/context", params=dict(base, ply=ply))
            self.assertEqual(response.status_code, 422, response.text)
        for path in ("../config.json", "/etc/passwd", "unindexed"):
            response = self.client.get("/api/platform/context", params=dict(base, replay_path=path))
            self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/api/platform/context", params=base).status_code, 200)
        self.assertEqual(self.provider.reads[-1], ("alias", self.provider.path))
        self.assertEqual(self.client.get("/api/platform/context", params={"run_id": "alias"}).status_code, 422)
        self.assertEqual(self.client.get("/api/platform/context", params={"replay_path": "a"}).status_code, 422)

    def test_legacy_turn_from_explicit_actor_not_ply_parity(self):
        for state in self.provider.payload["states"]:
            state.pop("current_player")
        self.provider.source = "legacy"
        self.assertEqual(self.context()["position"]["turn"], 2)
        self.assertEqual(self.context(ply=2)["position"]["turn"], 2)
        self.provider.payload["states"] = self.provider.payload["states"][:1]
        self.provider.payload.update(actions=[], game_length=0)
        with self.assertRaisesRegex(ValueError, "explicit replay turn"):
            self.context()
        self.provider.payload["initial_turn"] = "P2"
        self.assertEqual(self.context()["position"]["turn"], 2)

    def test_corrupt_turn_action_position_and_recorded_identity_rejected(self):
        original = deepcopy(self.provider.payload)
        for corrupt in (lambda r: r["states"][1].update(current_player=2),
                        lambda r: r["states"][1].update(p1=[0, 0]),
                        lambda r: r["actions"][0].update(player_id="P1"),
                        lambda r: r["states"][0].update(walls_remaining_p1=10),
                        lambda r: r["source"].update(checkpoint_sha256=self.other_sha)):
            self.provider.payload = deepcopy(original)
            corrupt(self.provider.payload)
            with self.assertRaises(ValueError):
                self.context()

    def test_duplicate_replay_identity_rejects_ambiguity(self):
        self.provider.rows.append(dict(self.provider.rows[0]))
        self.assertEqual(self.context()["recorded_model"]["sha256"], self.sha)
        self.provider.rows[-1]["checkpoint_sha"] = self.other_sha
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
            self.context()

    def test_index_source_game_identity_and_exact_selected_index_identity(self):
        self.provider.rows[0]["game_index"] = 1
        with self.assertRaisesRegex(ValueError, "Replay/index identity mismatch"):
            self.context()
        self.provider.rows[0]["game_index"] = 0
        self.provider.payload["source"]["game_index"] = 1
        with self.assertRaisesRegex(ValueError, "Replay/source identity mismatch"):
            self.context()
        self.provider.payload["source"]["game_index"] = 0
        wrong = dict(self.index.detail(self.sha, verify_sha=True), sha256=self.other_sha)
        with patch.object(self.index, "detail", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "different exact identity"):
                self.context(model_sha=self.sha)

    def test_unsupported_model_and_service_unavailability_http(self):
        self.models[0]["metadata"] = dict(self.metadata, observation_version="future_unsupported")
        self.write_catalog()
        params = dict(run_id="alias", replay_path=self.provider.path, model_sha=self.sha)
        self.assertEqual(self.client.get("/api/platform/context", params=params).status_code, 422)
        self.assertEqual(self.client.get("/api/platform/models/" + "f" * 64).status_code, 404)
        with patch.object(self.service, "provider_for", side_effect=RuntimeError("offline")):
            self.assertEqual(self.client.get("/api/platform/runs/alias/evaluations").status_code, 503)

    def test_no_sha_or_evaluation_is_guessed_from_checkpoint_filename(self):
        self.provider.payload.pop("source")
        self.provider.rows[0].pop("checkpoint_sha")
        self.provider.rows[0].pop("evaluation_id")
        result = self.context(model_sha=self.other_sha)
        self.assertIsNone(result["recorded_model"])
        self.assertIsNone(result["source"]["evaluation"])
        self.assertEqual(len(result["warnings"]), 3)

    def test_standard_position_and_model_only_context_do_not_read_replay(self):
        for digest in (None, self.sha):
            result = analysis_context(self.service, self.index, model_sha=digest)
            self.assertEqual(result["position"], dict(pawns=[[4, 0], [4, 8]], walls_remaining=[10, 10], walls=[], turn=1))
            self.assertIsNone(result["actual_action"])
            self.assertIsNone(result["recorded_model"])
        self.assertEqual(self.provider.reads, [])

    def test_default_app_binds_index_to_existing_environment_service(self):
        with patch("backend.app.UnifiedRunService", return_value=self.service), patch("backend.app.ModelIndex") as index:
            app = create_app()
        index.assert_called_once_with(unified_run_service=self.service)
        self.assertIs(app.state.runs, self.service)


if __name__ == "__main__":
    unittest.main()