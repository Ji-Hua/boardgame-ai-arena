import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "platform_entry.py"
spec = importlib.util.spec_from_file_location("image_platform_entry", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LabUpstreamTests(unittest.TestCase):
    def test_compose_and_loopback(self):
        for value in ("http://lab:8744", "http://127.0.0.1:8754"):
            self.assertEqual(module.loopback_upstream(value), value)

    def test_reject_other_targets(self):
        for value in ("http://lab:8743", "http://other:8744", "http://lab:8744/api",
                      "http://user@lab:8744", "https://lab:8744", "http://lab:8744?url=x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.loopback_upstream(value)
