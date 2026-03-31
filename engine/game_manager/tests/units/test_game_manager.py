# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for GameManager — lifecycle, state management, and history.

Tests the GameManager's own logic (lifecycle transitions, state tracking,
history management, error handling) by mocking the FFI layer. No real
Rust engine is needed.

These tests validate:
- Lifecycle state machine (UNINITIALIZED → RUNNING → TERMINAL)
- submit_action commit/reject behavior
- undo mechanics
- History invariants (len(actions) == len(states))
- Query delegation patterns
- Edge cases and error conditions
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from engine.game_manager.game_manager import GameManager
from engine.game_manager.types import ActionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_engine():
    """Create a mock RuleEngine with standard behavior."""
    engine = MagicMock()
    engine.topology.n.return_value = 9
    engine.topology.goal_y.side_effect = lambda p: 8 if p == "P1" else 0
    return engine


def _make_initial_state():
    """Create a mock initial state."""
    state = MagicMock(name="initial_state")
    state.walls_remaining.return_value = 10
    return state


def _make_state(name="state"):
    """Create a mock state."""
    state = MagicMock(name=name)
    state.walls_remaining.return_value = 10
    return state


def _make_action(name="action"):
    """Create a mock action."""
    return MagicMock(name=name)


def _patch_ffi():
    """Return a patch context manager for the ffi module."""
    return patch("engine.game_manager.game_manager.ffi")


# ===========================================================================
# Lifecycle Tests
# ===========================================================================

class TestLifecycleUninitialized(unittest.TestCase):
    """Tests for UNINITIALIZED state."""

    def setUp(self):
        with _patch_ffi() as mock_ffi:
            mock_ffi.create_rule_engine.return_value = _make_mock_engine()
            self.gm = GameManager()
            self.mock_ffi = mock_ffi

    def test_not_initialized_by_default(self):
        self.assertFalse(self.gm.is_initialized())

    def test_not_terminal_by_default(self):
        self.assertFalse(self.gm.is_terminal())

    def test_current_state_is_none_before_init(self):
        self.assertIsNone(self.gm.current_state())

    def test_initial_state_is_none_before_init(self):
        self.assertIsNone(self.gm.initial_state())

    def test_step_count_is_zero_before_init(self):
        self.assertEqual(self.gm.step_count(), 0)

    def test_actions_empty_before_init(self):
        self.assertEqual(self.gm.actions(), [])

    def test_states_empty_before_init(self):
        self.assertEqual(self.gm.states(), [])


class TestLifecycleInitialization(unittest.TestCase):
    """Tests for initialize() transition."""

    def test_initialize_sets_initialized(self):
        with _patch_ffi() as mock_ffi:
            mock_ffi.create_rule_engine.return_value = _make_mock_engine()
            initial = _make_initial_state()
            mock_ffi.initial_state.return_value = initial
            gm = GameManager()
            gm.initialize()

            self.assertTrue(gm.is_initialized())
            self.assertFalse(gm.is_terminal())

    def test_initialize_sets_states(self):
        with _patch_ffi() as mock_ffi:
            mock_ffi.create_rule_engine.return_value = _make_mock_engine()
            initial = _make_initial_state()
            mock_ffi.initial_state.return_value = initial
            gm = GameManager()
            gm.initialize()

            self.assertIs(gm.current_state(), initial)
            self.assertIs(gm.initial_state(), initial)
            self.assertEqual(gm.step_count(), 0)
            self.assertEqual(gm.actions(), [])
            self.assertEqual(gm.states(), [])

    def test_initialize_twice_raises(self):
        with _patch_ffi() as mock_ffi:
            mock_ffi.create_rule_engine.return_value = _make_mock_engine()
            mock_ffi.initial_state.return_value = _make_initial_state()
            gm = GameManager()
            gm.initialize()

            with self.assertRaises(RuntimeError):
                gm.initialize()


class TestLifecycleTermination(unittest.TestCase):
    """Tests for terminate() and terminal state."""

    def setUp(self):
        self.patcher = _patch_ffi()
        self.mock_ffi = self.patcher.start()
        self.mock_ffi.create_rule_engine.return_value = _make_mock_engine()
        self.mock_ffi.initial_state.return_value = _make_initial_state()
        self.gm = GameManager()
        self.gm.initialize()

    def tearDown(self):
        self.patcher.stop()

    def test_terminate_marks_terminal(self):
        self.gm.terminate()
        self.assertTrue(self.gm.is_terminal())

    def test_terminate_is_idempotent(self):
        self.gm.terminate()
        self.gm.terminate()
        self.assertTrue(self.gm.is_terminal())

    def test_terminate_blocks_submit_action(self):
        self.gm.terminate()
        result = self.gm.submit_action(_make_action())
        self.assertFalse(result.success)
        self.assertIn("terminal", result.error)

    def test_terminate_blocks_undo(self):
        self.gm.terminate()
        self.assertFalse(self.gm.undo())

    def test_terminate_allows_query(self):
        self.gm.terminate()
        # These should not raise
        self.gm.current_state()
        self.gm.initial_state()
        self.gm.step_count()
        self.gm.actions()
        self.gm.states()


# ===========================================================================
# Submit Action Tests
# ===========================================================================

class TestSubmitAction(unittest.TestCase):
    """Tests for submit_action behavior."""

    def setUp(self):
        self.patcher = _patch_ffi()
        self.mock_ffi = self.patcher.start()
        self.engine = _make_mock_engine()
        self.mock_ffi.create_rule_engine.return_value = self.engine
        self.initial = _make_initial_state()
        self.mock_ffi.initial_state.return_value = self.initial
        self.gm = GameManager()
        self.gm.initialize()

    def tearDown(self):
        self.patcher.stop()

    def test_reject_if_not_initialized(self):
        with _patch_ffi() as ffi2:
            ffi2.create_rule_engine.return_value = _make_mock_engine()
            gm2 = GameManager()
            result = gm2.submit_action(_make_action())
            self.assertFalse(result.success)
            self.assertIn("not initialized", result.error)

    def test_success_updates_state(self):
        new_state = _make_state("state_1")
        self.mock_ffi.apply_action.return_value = new_state
        action = _make_action("action_1")

        result = self.gm.submit_action(action)

        self.assertTrue(result.success)
        self.assertIs(result.state, new_state)
        self.assertIs(self.gm.current_state(), new_state)

    def test_success_appends_to_history(self):
        new_state = _make_state("state_1")
        self.mock_ffi.apply_action.return_value = new_state
        action = _make_action("action_1")

        self.gm.submit_action(action)

        self.assertEqual(self.gm.step_count(), 1)
        self.assertEqual(len(self.gm.actions()), 1)
        self.assertEqual(len(self.gm.states()), 1)
        self.assertIs(self.gm.actions()[0], action)
        self.assertIs(self.gm.states()[0], new_state)

    def test_failure_does_not_modify_state(self):
        self.mock_ffi.apply_action.side_effect = Exception("Invalid action")
        action = _make_action("bad_action")

        result = self.gm.submit_action(action)

        self.assertFalse(result.success)
        self.assertIn("Invalid action", result.error)
        self.assertIs(self.gm.current_state(), self.initial)
        self.assertEqual(self.gm.step_count(), 0)

    def test_multiple_submissions(self):
        states = [_make_state(f"state_{i}") for i in range(3)]
        actions = [_make_action(f"action_{i}") for i in range(3)]
        self.mock_ffi.apply_action.side_effect = states

        for action in actions:
            result = self.gm.submit_action(action)
            self.assertTrue(result.success)

        self.assertEqual(self.gm.step_count(), 3)
        self.assertIs(self.gm.current_state(), states[2])

    def test_failure_after_success_preserves_previous(self):
        state_1 = _make_state("state_1")
        self.mock_ffi.apply_action.return_value = state_1
        self.gm.submit_action(_make_action("ok"))

        self.mock_ffi.apply_action.side_effect = Exception("bad")
        result = self.gm.submit_action(_make_action("bad"))

        self.assertFalse(result.success)
        self.assertIs(self.gm.current_state(), state_1)
        self.assertEqual(self.gm.step_count(), 1)


# ===========================================================================
# Undo Tests
# ===========================================================================

class TestUndo(unittest.TestCase):
    """Tests for undo behavior."""

    def setUp(self):
        self.patcher = _patch_ffi()
        self.mock_ffi = self.patcher.start()
        self.engine = _make_mock_engine()
        self.mock_ffi.create_rule_engine.return_value = self.engine
        self.initial = _make_initial_state()
        self.mock_ffi.initial_state.return_value = self.initial
        self.gm = GameManager()
        self.gm.initialize()

    def tearDown(self):
        self.patcher.stop()

    def test_undo_empty_returns_false(self):
        self.assertFalse(self.gm.undo())

    def test_undo_not_initialized_returns_false(self):
        with _patch_ffi() as ffi2:
            ffi2.create_rule_engine.return_value = _make_mock_engine()
            gm2 = GameManager()
            self.assertFalse(gm2.undo())

    def test_undo_single_action_restores_initial(self):
        state_1 = _make_state("state_1")
        self.mock_ffi.apply_action.return_value = state_1
        self.gm.submit_action(_make_action())

        result = self.gm.undo()

        self.assertTrue(result)
        self.assertIs(self.gm.current_state(), self.initial)
        self.assertEqual(self.gm.step_count(), 0)
        self.assertEqual(self.gm.actions(), [])
        self.assertEqual(self.gm.states(), [])

    def test_undo_multiple_actions(self):
        state_1 = _make_state("state_1")
        state_2 = _make_state("state_2")
        self.mock_ffi.apply_action.side_effect = [state_1, state_2]
        self.gm.submit_action(_make_action("a1"))
        self.gm.submit_action(_make_action("a2"))

        self.gm.undo()
        self.assertIs(self.gm.current_state(), state_1)
        self.assertEqual(self.gm.step_count(), 1)

        self.gm.undo()
        self.assertIs(self.gm.current_state(), self.initial)
        self.assertEqual(self.gm.step_count(), 0)

    def test_undo_then_submit(self):
        state_1 = _make_state("state_1")
        state_2 = _make_state("state_2")
        state_3 = _make_state("state_3")
        self.mock_ffi.apply_action.side_effect = [state_1, state_2, state_3]

        self.gm.submit_action(_make_action("a1"))
        self.gm.submit_action(_make_action("a2"))
        self.gm.undo()
        self.gm.submit_action(_make_action("a3"))

        self.assertIs(self.gm.current_state(), state_3)
        self.assertEqual(self.gm.step_count(), 2)


# ===========================================================================
# History Tests
# ===========================================================================

class TestHistory(unittest.TestCase):
    """Tests for history query APIs."""

    def setUp(self):
        self.patcher = _patch_ffi()
        self.mock_ffi = self.patcher.start()
        self.engine = _make_mock_engine()
        self.mock_ffi.create_rule_engine.return_value = self.engine
        self.initial = _make_initial_state()
        self.mock_ffi.initial_state.return_value = self.initial
        self.gm = GameManager()
        self.gm.initialize()

    def tearDown(self):
        self.patcher.stop()

    def test_get_state_at_zero_returns_initial(self):
        self.assertIs(self.gm.get_state_at(0), self.initial)

    def test_get_state_at_step(self):
        states = [_make_state(f"s{i}") for i in range(3)]
        self.mock_ffi.apply_action.side_effect = states
        for i in range(3):
            self.gm.submit_action(_make_action(f"a{i}"))

        self.assertIs(self.gm.get_state_at(0), self.initial)
        self.assertIs(self.gm.get_state_at(1), states[0])
        self.assertIs(self.gm.get_state_at(2), states[1])
        self.assertIs(self.gm.get_state_at(3), states[2])

    def test_get_state_at_out_of_range(self):
        with self.assertRaises(IndexError):
            self.gm.get_state_at(1)

    def test_get_state_at_negative(self):
        with self.assertRaises(IndexError):
            self.gm.get_state_at(-1)

    def test_actions_returns_copy(self):
        state_1 = _make_state()
        self.mock_ffi.apply_action.return_value = state_1
        self.gm.submit_action(_make_action())

        actions = self.gm.actions()
        actions.clear()
        self.assertEqual(self.gm.step_count(), 1)

    def test_states_returns_copy(self):
        state_1 = _make_state()
        self.mock_ffi.apply_action.return_value = state_1
        self.gm.submit_action(_make_action())

        states = self.gm.states()
        states.clear()
        self.assertEqual(len(self.gm.states()), 1)

    def test_history_invariant_len_actions_eq_len_states(self):
        states = [_make_state(f"s{i}") for i in range(5)]
        self.mock_ffi.apply_action.side_effect = states
        for i in range(5):
            self.gm.submit_action(_make_action(f"a{i}"))

        self.assertEqual(len(self.gm.actions()), len(self.gm.states()))

        self.gm.undo()
        self.assertEqual(len(self.gm.actions()), len(self.gm.states()))

        self.gm.undo()
        self.assertEqual(len(self.gm.actions()), len(self.gm.states()))


# ===========================================================================
# Rule Semantic Pass-Through Tests
# ===========================================================================

class TestRuleSemanticPassthrough(unittest.TestCase):
    """Tests that rule semantic queries delegate to FFI correctly."""

    def setUp(self):
        self.patcher = _patch_ffi()
        self.mock_ffi = self.patcher.start()
        self.engine = _make_mock_engine()
        self.mock_ffi.create_rule_engine.return_value = self.engine
        self.initial = _make_initial_state()
        self.mock_ffi.initial_state.return_value = self.initial
        self.gm = GameManager()
        self.gm.initialize()

    def tearDown(self):
        self.patcher.stop()

    def test_is_game_over_delegates(self):
        self.mock_ffi.is_game_over.return_value = False
        self.assertFalse(self.gm.is_game_over())
        self.mock_ffi.is_game_over.assert_called_once_with(self.engine, self.initial)

    def test_winner_delegates(self):
        self.mock_ffi.winner.return_value = None
        self.assertIsNone(self.gm.winner())
        self.mock_ffi.winner.assert_called_once_with(self.engine, self.initial)

    def test_remaining_walls_delegates(self):
        self.mock_ffi.remaining_walls.return_value = 10
        self.assertEqual(self.gm.remaining_walls("P1"), 10)
        self.mock_ffi.remaining_walls.assert_called_once_with(self.initial, "P1")

    def test_goal_cells_delegates(self):
        self.mock_ffi.goal_cells.return_value = {(i, 8) for i in range(9)}
        result = self.gm.goal_cells("P1")
        self.mock_ffi.goal_cells.assert_called_once_with(self.engine, "P1")
        self.assertEqual(len(result), 9)

    def test_path_exists_delegates(self):
        self.mock_ffi.path_exists.return_value = True
        self.assertTrue(self.gm.path_exists("P1"))
        self.mock_ffi.path_exists.assert_called_once_with(self.engine, self.initial, "P1")

    def test_shortest_path_len_delegates(self):
        self.mock_ffi.shortest_path_len.return_value = 8
        self.assertEqual(self.gm.shortest_path_len("P1"), 8)
        self.mock_ffi.shortest_path_len.assert_called_once_with(self.engine, self.initial, "P1")

    def test_legal_actions_delegates(self):
        mock_actions = [_make_action("a1"), _make_action("a2")]
        self.mock_ffi.legal_actions.return_value = mock_actions
        result = self.gm.legal_actions()
        self.mock_ffi.legal_actions.assert_called_once_with(self.engine, self.initial)
        self.assertEqual(len(result), 2)


# ===========================================================================
# Replay Tests
# ===========================================================================

class TestReplay(unittest.TestCase):
    """Tests for replay() debug method."""

    def setUp(self):
        self.patcher = _patch_ffi()
        self.mock_ffi = self.patcher.start()
        self.engine = _make_mock_engine()
        self.mock_ffi.create_rule_engine.return_value = self.engine
        self.initial = _make_initial_state()
        self.mock_ffi.initial_state.return_value = self.initial
        self.gm = GameManager()
        self.gm.initialize()

    def tearDown(self):
        self.patcher.stop()

    def test_replay_empty(self):
        result = self.gm.replay()
        self.assertIs(result, self.initial)

    def test_replay_recomputes_from_actions(self):
        state_1 = _make_state("s1")
        state_2 = _make_state("s2")
        replay_s1 = _make_state("replay_s1")
        replay_s2 = _make_state("replay_s2")

        # Initial submits
        self.mock_ffi.apply_action.side_effect = [state_1, state_2]
        a1 = _make_action("a1")
        a2 = _make_action("a2")
        self.gm.submit_action(a1)
        self.gm.submit_action(a2)

        # Replay calls
        self.mock_ffi.apply_action.side_effect = [replay_s1, replay_s2]
        result = self.gm.replay()

        self.assertIs(result, replay_s2)


if __name__ == "__main__":
    unittest.main()
