# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for types.py — ActionResult dataclass."""

from __future__ import annotations

import unittest

import sys
from pathlib import Path

# Ensure the engine directory is importable
_engine_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_engine_root) not in sys.path:
    sys.path.insert(0, str(_engine_root))

# Import types directly (not via __init__.py which triggers FFI import)
from engine.game_manager.types import ActionResult


class TestActionResult(unittest.TestCase):
    """Tests for ActionResult type."""

    def test_success_result(self):
        r = ActionResult(success=True, state="some_state")
        self.assertTrue(r.success)
        self.assertEqual(r.state, "some_state")
        self.assertIsNone(r.error)

    def test_failure_result(self):
        r = ActionResult(success=False, error="bad move")
        self.assertFalse(r.success)
        self.assertIsNone(r.state)
        self.assertEqual(r.error, "bad move")

    def test_defaults(self):
        r = ActionResult(success=True)
        self.assertIsNone(r.state)
        self.assertIsNone(r.error)

    def test_frozen(self):
        r = ActionResult(success=True, state="s")
        with self.assertRaises(AttributeError):
            r.success = False

    def test_equality(self):
        r1 = ActionResult(success=True, state="s")
        r2 = ActionResult(success=True, state="s")
        self.assertEqual(r1, r2)

    def test_inequality(self):
        r1 = ActionResult(success=True)
        r2 = ActionResult(success=False, error="e")
        self.assertNotEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
