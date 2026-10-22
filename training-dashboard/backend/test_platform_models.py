"""Model index identity, safety and conservative capability contracts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.platform_models import ARCHITECTURE_CATALOG, ModelIndex


class ModelIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.sha = hashlib.sha256(b"checkpoint").hexdigest()
        self.metadata = dict(observation_version="dqn_obs_cnn_v1", action_space_version="dqn_action_v1",
            action_count=209, model_config=dict(model_arch="cnn", cnn_variant="resnet",
                algorithm="policy_distillation"))
        self.write("catalog.json", dict(models=[self.model()]))
        self.write("registry.json", dict(models=[]))
        (self.root / "agent_defs").mkdir()

    def write(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
        return target

    def model(self, **extra):
        return dict(id="C-resnet_seed-42", sha256=self.sha, checkpoint=str(self.checkpoint),
                    metadata=self.metadata, **extra)

    def index(self, **kwargs):
        return ModelIndex(data_root=self.root, catalog_paths=[self.root / "catalog.json"], **kwargs)

    def test_group_full_sha_preserves_aliases_and_evidence(self):
        second = self.model()
        second["id"] = "alias"
        self.write("extra.json", dict(models=[second]))
        index = ModelIndex(data_root=self.root, catalog_paths=[self.root / "catalog.json", self.root / "extra.json"])
        result = index.listing()
        self.assertEqual(result["count"], 1)
        row = result["models"][0]
        self.assertEqual(row["id"], self.sha)
        self.assertEqual(row["aliases"], ["C-resnet_seed-42", "alias"])
        self.assertEqual(len(row["evidence"]), 2)
        self.assertTrue(row["analyzable"])
        self.assertFalse(row["actual_load_verified"])
        self.assertIn("requires_isolated_research_catalog", row["reasons"]["analyzable"])
        self.assertEqual(row["output_semantics"], "policy_logits")

    def test_default_listing_never_hashes_or_deserializes(self):
        with patch("backend.platform_models.hashlib.file_digest", side_effect=AssertionError("expensive")):
            row = self.index().listing()["models"][0]
        self.assertTrue(row["available"])
        self.assertFalse(row["sha_verified"])
        self.assertFalse(row["deployed"])

    def test_hash_cache_stat_invalidation_and_mismatch(self):
        index = self.index()
        original = hashlib.file_digest
        with patch("backend.platform_models.hashlib.file_digest", wraps=original) as hashed:
            self.assertTrue(index.detail(self.sha, verify_sha=True)["sha_verified"])
            self.assertTrue(index.detail(self.sha, verify_sha=True)["sha_verified"])
            self.assertEqual(hashed.call_count, 1)
            self.checkpoint.write_bytes(b"changed checkpoint")
            row = index.detail(self.sha, verify_sha=True)
            self.assertEqual(hashed.call_count, 2)
        self.assertFalse(row["sha_verified"])
        self.assertTrue(row["available"])
        self.assertFalse(row["analyzable"])
        self.assertIn("checkpoint_sha_mismatch", row["conflicts"])

    def test_missing_checkpoint_keeps_identity(self):
        self.checkpoint.unlink()
        row = self.index().detail(self.sha, verify_sha=True)
        self.assertFalse(row["available"])
        self.assertFalse(row["sha_verified"])
        self.assertFalse(row["analyzable"])

    def test_same_id_different_sha_blocks_both(self):
        other = self.model()
        other["sha256"] = "a" * 64
        self.write("registry.json", dict(models=[other]))
        for row in self.index().listing()["models"]:
            self.assertIn("id_sha_conflict:C-resnet_seed-42", row["conflicts"])
            self.assertFalse(row["analyzable"])
            self.assertFalse(row["playable"])
            self.assertFalse(row["deployed"])

    def test_metadata_conflict_disables_capabilities(self):
        other = self.model()
        other["metadata"] = dict(self.metadata, model_config=dict(model_arch="mlp"))
        self.write("registry.json", dict(models=[other]))
        row = self.index(deployed_shas={self.sha: "observed service snapshot"}).detail(self.sha)
        self.assertIn("metadata_conflict:model_config.model_arch", row["conflicts"])
        self.assertFalse(row["analyzable"])
        self.assertFalse(row["deployed"])

    def test_registry_and_catalog_sha_lineage_not_names(self):
        self.write("registry.json", dict(models=[dict(id="unrelated-name", sha256=self.sha,
            run_id="exact-run", name="quoridor-policy", version="4")]))
        self.write("catalog.json", dict(models=[self.model(run_ref=dict(source="legacy", run_id="exact-legacy"))]))
        row = self.index().detail(self.sha)
        self.assertEqual({(r["source"], r["run_id"]) for r in row["run_refs"]},
                         {("mlflow", "exact-run"), ("legacy", "exact-legacy")})
        self.assertTrue(all(r["checkpoint_sha256"] == self.sha for r in row["run_refs"]))

    def test_c_resnet_lineage_joins_exact_run_checkpoint_sha(self):
        class Provider:
            def detail(inner, run_id):
                return dict(checkpoints=[dict(sha256=self.sha if run_id == "correct" else "b" * 64,
                                             filename="update012000.pt")])
        class Service:
            def listing(inner):
                return dict(runs=[dict(source="mlflow", run_id="correct", run_name="different"),
                                  dict(source="mlflow", run_id="wrong", run_name="C-resnet_seed-42")])
            def provider_for(inner, run_id):
                return Provider()
        row = self.index(unified_run_service=Service()).detail(self.sha)
        self.assertEqual([r["run_id"] for r in row["run_refs"]], ["correct"])
        self.assertEqual(row["metadata"]["model_config.cnn_variant"], "resnet")

    def test_agent_spec_is_not_deployment(self):
        self.write("agent_defs/model.json", dict(id="pinned", algo=dict(type="dqn", params=dict(
            sha256=self.sha, checkpoint=str(self.checkpoint))), policy=dict(type="top_k", k=1)))
        row = self.index().detail(self.sha)
        self.assertTrue(row["playable"])
        self.assertFalse(row["deployed"])
        self.assertIn("actual_load_not_verified", row["reasons"]["playable"])

    def test_missing_spec_checkpoint_cannot_borrow_available_catalog_path(self):
        self.write("agent_defs/model.json", dict(id="pinned", algo=dict(type="dqn", params=dict(
            sha256=self.sha, checkpoint=str(self.root / "missing.pt"))), policy=dict(type="top_k", k=1)))
        row = self.index().detail(self.sha)
        self.assertTrue(row["available"])
        self.assertFalse(row["playable"])

    def test_summary_sha_lineage_does_not_call_expensive_detail(self):
        class Service:
            def listing(inner):
                return dict(runs=[dict(source="mlflow", run_id="resnet-run", provenance=dict(
                    catalog_matches=[dict(id="irrelevant", sha256=self.sha)]))])
            def provider_for(inner, run_id):
                raise AssertionError("detail may compute benchmarks and hash checkpoints")
        row = self.index(unified_run_service=Service()).detail(self.sha)
        self.assertEqual(row["run_refs"][0]["run_id"], "resnet-run")

    def test_malformed_spec_is_warning_not_index_failure(self):
        (self.root / "agent_defs/broken.yaml").write_text("algo: [broken")
        result = self.index().listing()
        self.assertEqual(result["count"], 1)
        self.assertTrue(any("Malformed AgentSpec" in w for w in result["warnings"]))

    def test_unapproved_path_and_symlink_are_unavailable(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        target = Path(outside.name) / "outside.pt"
        target.write_bytes(b"checkpoint")
        self.checkpoint.unlink()
        self.checkpoint.symlink_to(target)
        row = self.index().detail(self.sha, verify_sha=True)
        self.assertFalse(row["available"])
        self.assertIn("approved roots", row["checkpoints"][0]["reason"])

    def test_default_explicit_source_manifest_and_env_override(self):
        with patch.dict("os.environ", {}, clear=True):
            result = ModelIndex(data_root=self.root).listing()
        self.assertEqual([s["name"] for s in result["sources"]],
                         ["baseline", "lab", "architecture", "registry", "agent_specs"])
        self.assertEqual(result["sources"][2]["path"], str(self.root / ARCHITECTURE_CATALOG))
        self.assertEqual(result["sources"][2]["status"], "unavailable")
        with patch.dict("os.environ", {"PLATFORM_MODEL_CATALOGS": json.dumps([str(self.root / "catalog.json")])}):
            self.assertEqual(len(ModelIndex(data_root=self.root).catalogs), 1)

    def test_invalid_sha_is_not_a_model_and_detail_rejects_alias(self):
        self.write("catalog.json", dict(models=[dict(id="invalid", sha256="short")]))
        self.assertEqual(self.index().listing()["count"], 0)
        with self.assertRaises(KeyError):
            self.index().detail("C-resnet_seed-42")


if __name__ == "__main__":
    unittest.main()