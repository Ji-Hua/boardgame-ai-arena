import importlib.util
import unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location("project", Path(__file__).resolve().parents[2] / "scripts/project.py")
project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project)

class ReleaseTests(unittest.TestCase):
    def manifest(self):
        return {"schema_version": 1, "release_id": "candidate-1", "source_commit": "a" * 40,
                "contracts": {k: "v1" for k in ("agent_api", "position_schema", "observation", "action_space")},
                "images": {k: "ghcr.io/example/" + k + "@sha256:" + "b" * 64
                           for values in project.ROLES.values() for k in values}}
    def test_host_subset(self):
        self.assertEqual(set(project.validate_release(self.manifest(), "dev")), {"WEB_IMAGE", "GAMEPLAY_IMAGE"})
    def test_mutable_tag_rejected(self):
        m = self.manifest(); m["images"]["runtime"] = "example/runtime:latest"
        with self.assertRaises(ValueError): project.validate_release(m, "training")
    def test_short_commit_rejected(self):
        m = self.manifest(); m["source_commit"] = "abc123"
        with self.assertRaises(ValueError): project.validate_release(m, "dev")
    def test_missing_contract_rejected(self):
        m = self.manifest(); del m["contracts"]["observation"]
        with self.assertRaises(ValueError): project.validate_release(m, "dev")
    def test_env_injection_rejected(self):
        m = self.manifest(); m["images"]["web"] += "\nEVIL=value"
        with self.assertRaises(ValueError): project.validate_release(m, "dev")
    def test_missing_component_rejected(self):
        m = self.manifest(); del m["images"]["lab"]
        with self.assertRaises(ValueError): project.validate_release(m, "training")

    def test_nonobject_manifest_rejected(self):
        with self.assertRaises(ValueError): project.validate_release([], "dev")
