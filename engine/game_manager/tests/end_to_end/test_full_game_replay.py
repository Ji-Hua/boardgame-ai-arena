# TEST_CLASSIFICATION: SPECIFIED
"""End-to-end test: full game replay through GameManager.

Parses documents/engine/implementation/full_game_replay.md and replays
every step through the GameManager, asserting:
  - ACCEPT steps succeed via submit_action
  - REJECT steps fail and do not modify state
  - State snapshots match at documented checkpoints
  - Game ends with Player 1 winning
  - History invariants hold throughout

This test requires the quoridor_engine PyO3 bindings to be built.
It will be skipped if the bindings are not available.
"""

from __future__ import annotations

import os
import re
import unittest
from dataclasses import dataclass
from pathlib import Path

# Attempt to import the real engine bindings
try:
    import quoridor_engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


# ---------------------------------------------------------------------------
# Step parser (mirrors the Rust test's parser)
# ---------------------------------------------------------------------------

@dataclass
class GameStep:
    label: str
    player: int
    kind: str
    target_x: int
    target_y: int
    target_type: str
    accept: bool


def parse_steps(content: str) -> list[GameStep]:
    """Parse replay steps from the markdown document."""
    step_re = re.compile(r"^Step\s+(\d+(?:\.\d+)?)\s*:")
    action_re = re.compile(
        r"Action:\s*\{\s*player:\s*(\d+)\s*,\s*kind:\s*(\w+)\s*,"
        r"\s*target:\s*\((-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)\)\s*\}"
        r"\s*->\s*(ACCEPT|REJECT)"
    )

    steps: list[GameStep] = []
    current_label = ""

    for line in content.splitlines():
        m = step_re.match(line.strip())
        if m:
            current_label = m.group(1)

        m = action_re.search(line)
        if m:
            steps.append(GameStep(
                label=current_label,
                player=int(m.group(1)),
                kind=m.group(2),
                target_x=int(m.group(3)),
                target_y=int(m.group(4)),
                target_type=m.group(5),
                accept=m.group(6) == "ACCEPT",
            ))

    return steps


def _find_replay_doc() -> str:
    """Locate and read the full_game_replay.md document."""
    # Navigate from this test file to the documents directory
    test_dir = Path(__file__).resolve().parent
    # engine/game_manager/tests/end_to_end/ -> engine/ -> repo root
    repo_root = test_dir.parent.parent.parent.parent
    doc_path = repo_root / "documents" / "engine" / "implementation" / "full_game_replay.md"
    if not doc_path.exists():
        raise FileNotFoundError(f"Replay document not found at {doc_path}")
    return doc_path.read_text()


# ---------------------------------------------------------------------------
# Snapshot definitions from the replay document
# ---------------------------------------------------------------------------

SNAPSHOTS = {
    "10": {"p1": (4, 2), "p2": (4, 3), "p1_walls": 10, "p2_walls": 9},
    "18": {"p1": (3, 4), "p2": (3, 2), "p1_walls": 9, "p2_walls": 8},
    "28": {"p1": (3, 6), "p2": (1, 3), "p1_walls": 6, "p2_walls": 8},
    "38": {"p1": (5, 7), "p2": (1, 6), "p1_walls": 6, "p2_walls": 6},
    "48": {"p1": (4, 7), "p2": (1, 7), "p1_walls": 4, "p2_walls": 4},
    "58": {"p1": (2, 7), "p2": (3, 7), "p1_walls": 2, "p2_walls": 1},
}


# ---------------------------------------------------------------------------
# E2E Test
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_ENGINE, "quoridor_engine PyO3 bindings not available")
class TestFullGameReplayE2E(unittest.TestCase):
    """Replay the full game document through the GameManager."""

    def test_full_game_replay(self):
        from engine.game_manager.game_manager import GameManager

        content = _find_replay_doc()
        steps = parse_steps(content)
        self.assertEqual(len(steps), 109, "Expected 109 steps (61 ACCEPT + 48 REJECT)")

        gm = GameManager()
        gm.initialize()

        # Verify initial state
        state = gm.current_state()
        self.assertEqual(state.pawn_pos(quoridor_engine.Player.P1), (4, 0))
        self.assertEqual(state.pawn_pos(quoridor_engine.Player.P2), (4, 8))

        accept_count = 0
        reject_count = 0
        failures: list[str] = []

        for step in steps:
            # Build action
            if step.target_x < 0 or step.target_y < 0 or step.target_x >= 9 or step.target_y >= 9:
                # Out-of-bounds coordinates can't form a valid action
                if step.accept:
                    failures.append(
                        f"Step {step.label}: unrepresentable action marked ACCEPT"
                    )
                else:
                    reject_count += 1
                continue

            player = quoridor_engine.Player.P1 if step.player == 1 else quoridor_engine.Player.P2

            if step.kind == "MovePawn":
                action = quoridor_engine.Action.move_pawn(player, step.target_x, step.target_y)
            elif step.kind == "PlaceWall":
                if step.target_type == "Horizontal":
                    orientation = quoridor_engine.Orientation.Horizontal
                elif step.target_type == "Vertical":
                    orientation = quoridor_engine.Orientation.Vertical
                else:
                    failures.append(f"Step {step.label}: unknown orientation {step.target_type}")
                    continue
                action = quoridor_engine.Action.place_wall(player, step.target_x, step.target_y, orientation)
            else:
                failures.append(f"Step {step.label}: unknown kind {step.kind}")
                continue

            prev_state = gm.current_state()
            prev_step_count = gm.step_count()

            result = gm.submit_action(action)

            if step.accept:
                if not result.success:
                    failures.append(
                        f"Step {step.label}: expected ACCEPT but got REJECT: {result.error}"
                    )
                    continue

                accept_count += 1

                # Verify history invariant
                self.assertEqual(
                    len(gm.actions()), len(gm.states()),
                    f"History invariant violated at step {step.label}",
                )
                self.assertEqual(gm.step_count(), prev_step_count + 1)

                # Check snapshots at documented checkpoints
                if step.label in SNAPSHOTS:
                    snap = SNAPSHOTS[step.label]
                    cur = gm.current_state()
                    p1_pos = cur.pawn_pos(quoridor_engine.Player.P1)
                    p2_pos = cur.pawn_pos(quoridor_engine.Player.P2)
                    p1_walls = cur.walls_remaining(quoridor_engine.Player.P1)
                    p2_walls = cur.walls_remaining(quoridor_engine.Player.P2)

                    if p1_pos != snap["p1"]:
                        failures.append(f"Snapshot {step.label}: P1 at {p1_pos}, expected {snap['p1']}")
                    if p2_pos != snap["p2"]:
                        failures.append(f"Snapshot {step.label}: P2 at {p2_pos}, expected {snap['p2']}")
                    if p1_walls != snap["p1_walls"]:
                        failures.append(f"Snapshot {step.label}: P1 walls {p1_walls}, expected {snap['p1_walls']}")
                    if p2_walls != snap["p2_walls"]:
                        failures.append(f"Snapshot {step.label}: P2 walls {p2_walls}, expected {snap['p2_walls']}")

            else:  # expected REJECT
                if result.success:
                    failures.append(f"Step {step.label}: expected REJECT but got ACCEPT")
                else:
                    # State must not have changed
                    self.assertIs(
                        gm.current_state(), prev_state,
                        f"Step {step.label}: state changed on REJECT",
                    )
                    self.assertEqual(
                        gm.step_count(), prev_step_count,
                        f"Step {step.label}: step_count changed on REJECT",
                    )
                    reject_count += 1

        # Report any failures
        if failures:
            msg = "\n  ".join(failures)
            self.fail(
                f"\n{len(failures)} step failure(s):\n  {msg}\n\n"
                f"({accept_count} ACCEPT ok, {reject_count} REJECT ok)"
            )

        # Final assertions
        self.assertEqual(accept_count, 61, "Should have 61 ACCEPT steps")
        self.assertEqual(reject_count, 48, "Should have 48 REJECT steps")
        self.assertTrue(gm.is_game_over(), "Game should be over after final step")
        self.assertEqual(
            gm.winner(), quoridor_engine.Player.P1,
            "Player 1 should win"
        )

        # History completeness
        self.assertEqual(gm.step_count(), 61)
        self.assertIs(gm.get_state_at(0), gm.initial_state())

        # Replay consistency
        replayed = gm.replay()
        self.assertEqual(
            replayed, gm.current_state(),
            "replay() should produce the same final state"
        )


# ---------------------------------------------------------------------------
# Mock-based E2E test (validates GameManager orchestration without engine)
# ---------------------------------------------------------------------------

class TestFullGameReplayOrchestration(unittest.TestCase):
    """Validate GameManager orchestration patterns using the replay document.

    This test mocks the FFI layer and verifies that GameManager correctly:
    - Routes ACCEPT/REJECT through submit_action
    - Maintains history on ACCEPT
    - Does NOT modify state on REJECT
    - Handles the full 109-step sequence
    """

    def test_orchestration_accept_reject_counts(self):
        """Verify the replay document has correct step counts."""
        content = _find_replay_doc()
        steps = parse_steps(content)
        self.assertEqual(len(steps), 109)
        accept_count = sum(1 for s in steps if s.accept)
        reject_count = sum(1 for s in steps if not s.accept)
        self.assertEqual(accept_count, 61)
        self.assertEqual(reject_count, 48)

    def test_orchestration_step_labels(self):
        """Verify step labels are properly ordered."""
        content = _find_replay_doc()
        steps = parse_steps(content)

        # All integer-labeled steps should be ACCEPT
        for s in steps:
            if "." not in s.label:
                self.assertTrue(
                    s.accept,
                    f"Integer step {s.label} should be ACCEPT",
                )

    def test_orchestration_mock_game_flow(self):
        """Run the full game through GameManager with a mock FFI."""
        from unittest.mock import patch, MagicMock

        content = _find_replay_doc()
        steps = parse_steps(content)

        with patch("engine.game_manager.game_manager.ffi") as mock_ffi:
            mock_engine = MagicMock()
            mock_ffi.create_rule_engine.return_value = mock_engine
            mock_initial = MagicMock(name="initial_state")
            mock_ffi.initial_state.return_value = mock_initial

            from engine.game_manager.game_manager import GameManager
            gm = GameManager()
            gm.initialize()

            accept_count = 0
            reject_count = 0
            state_counter = [0]

            def mock_apply(engine, state, action):
                if action._should_reject:
                    raise Exception("Rule violation")
                state_counter[0] += 1
                return MagicMock(name=f"state_{state_counter[0]}")

            mock_ffi.apply_action.side_effect = mock_apply

            for step in steps:
                action = MagicMock(name=f"action_{step.label}")
                action._should_reject = not step.accept

                prev_count = gm.step_count()
                result = gm.submit_action(action)

                if step.accept:
                    self.assertTrue(
                        result.success,
                        f"Step {step.label}: expected ACCEPT",
                    )
                    self.assertEqual(gm.step_count(), prev_count + 1)
                    accept_count += 1
                else:
                    self.assertFalse(
                        result.success,
                        f"Step {step.label}: expected REJECT",
                    )
                    self.assertEqual(gm.step_count(), prev_count)
                    reject_count += 1

            self.assertEqual(accept_count, 61)
            self.assertEqual(reject_count, 48)
            self.assertEqual(gm.step_count(), 61)

            # Verify history invariant held throughout
            self.assertEqual(len(gm.actions()), len(gm.states()))
            self.assertEqual(len(gm.actions()), 61)


if __name__ == "__main__":
    unittest.main()
