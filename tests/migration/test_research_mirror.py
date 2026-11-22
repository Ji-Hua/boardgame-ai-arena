import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

@unittest.skipUnless(shutil.which("rsync"),"rsync required")
class ResearchMirrorTests(unittest.TestCase):
    def test_private_state_excluded_public_bundle_included(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"source";target=Path(temp)/"target"
            for name in ("jobs/projects.json","policies/actor.npz","policies/policy.json",
                         "jobs/runs/run/private-replays.jsonl","jobs/projects.sqlite",
                         "policies/catalog.sqlite","games/private.sqlite"):
                p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("fixture")
            target.mkdir();(target/"games").mkdir();(target/"games"/"existing").write_text("owned by gameplay")
            subprocess.run(["sh",str(Path(__file__).resolve().parents[2]/"scripts/sync_gameplay_mirror.sh")],
                env={**os.environ,"MIRROR_SOURCE":str(source)+"/","MIRROR_TARGET":str(target)+"/"},
                check=True,capture_output=True)
            self.assertTrue((target/"policies"/"actor.npz").is_file())
            self.assertTrue((target/"jobs"/"projects.json").is_file())
            self.assertTrue((target/"games"/"existing").is_file())
            for name in ("jobs/runs/run/private-replays.jsonl","jobs/projects.sqlite","policies/catalog.sqlite","games/private.sqlite"):
                self.assertFalse((target/name).exists(),name)
