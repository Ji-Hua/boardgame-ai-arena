# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for Random Agent V2.

Contract under test (documents/agent/agent-interface.md, Section 4.2):
  - Agent MAY attempt wall placements outside the provided legal_actions list
  - Agent MUST return exactly one Action per RequestAction call
  - Action must be in wire format: {player, type, target}

Behavioral requirements:
  - RandomAgentV2 samples from both pawn moves and wall placements
  - Wall placements are generated via local Rule Engine (full action space)
  - Weighted random: ~80% pawn, ~20% wall (by default threshold)
  - All produced actions are valid (accepted by the Engine)
"""

from __future__ import annotations

import random

import pytest

from agent_system.runtime.service.agents.random_agent import RandomAgent, RandomAgentV2

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Initial game state with wall_state bitmaps (no walls placed).
_INITIAL_STATE = {
    "current_player": 1,
    "pawns": {
        "1": {"row": 4, "col": 0},
        "2": {"row": 4, "col": 8},
    },
    "walls_remaining": {"1": 10, "2": 10},
    "wall_state": {
        "horizontal_edges": 0,
        "vertical_edges": 0,
        "horizontal_heads": 0,
        "vertical_heads": 0,
    },
    "game_over": False,
    "winner": None,
}

# Legal pawn actions for P1 from start (provided by Backend).
_P1_LEGAL_PAWN_ACTIONS = [
    {"player": 1, "type": "pawn", "target": [3, 0]},
    {"player": 1, "type": "pawn", "target": [4, 1]},
    {"player": 1, "type": "pawn", "target": [5, 0]},
]


# ---------------------------------------------------------------------------
# Test 1: Action Space — V2 generates both pawn and wall actions
# ---------------------------------------------------------------------------

def test_v2_action_space_includes_walls():
    """RandomAgentV2 must be able to produce wall actions, not just pawn moves.

    Over 200 calls, at least one pawn and one wall action must appear.
    """
    agent = RandomAgentV2(threshold=0.5)  # 50/50 to speed up sampling
    random.seed(42)

    types_seen = set()
    for _ in range(200):
        action = agent.make_action(_INITIAL_STATE, _P1_LEGAL_PAWN_ACTIONS)
        types_seen.add(action["type"])

    assert "pawn" in types_seen, "V2 never produced a pawn action"
    assert types_seen & {"horizontal", "vertical"}, "V2 never produced a wall action"


# ---------------------------------------------------------------------------
# Test 2: Sampling — weighted selection produces both categories
# ---------------------------------------------------------------------------

def test_v2_sampling_distribution():
    """Over many samples, both pawn and wall actions must appear.

    With default threshold=0.8, roughly 80% pawn, 20% wall.
    We verify the proportions are within a reasonable range.
    """
    agent = RandomAgentV2()
    random.seed(123)

    pawn_count = 0
    wall_count = 0
    n = 500

    for _ in range(n):
        action = agent.make_action(_INITIAL_STATE, _P1_LEGAL_PAWN_ACTIONS)
        if action["type"] == "pawn":
            pawn_count += 1
        else:
            wall_count += 1

    assert pawn_count > 0, "No pawn actions sampled"
    assert wall_count > 0, "No wall actions sampled"
    # Expect ~80% pawn within a generous margin
    pawn_ratio = pawn_count / n
    assert 0.6 < pawn_ratio < 0.95, f"Pawn ratio {pawn_ratio:.2f} outside expected range"


# ---------------------------------------------------------------------------
# Test 3: End-to-End — V2 vs V2 game completes with wall placements
# ---------------------------------------------------------------------------

def test_v2_end_to_end_game_with_walls():
    """A full RandomV2 vs RandomV2 game must complete and include wall placements."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "backend-server"))
    from backend.adapters.engine_adapter import EngineAdapter

    random.seed(999)

    adapter = EngineAdapter()
    adapter.initialize()
    agents = {1: RandomAgentV2(), 2: RandomAgentV2()}

    wall_count = 0
    max_steps = 2000

    for step in range(max_steps):
        state = adapter.get_state()
        if state.get("game_over"):
            break
        seat = state["current_player"]
        legal = adapter.legal_pawn_actions()
        action = agents[seat].make_action(state, legal)
        if action["type"] in ("horizontal", "vertical"):
            wall_count += 1
        result = adapter.take_action(action)
        assert result["success"], f"Step {step}: Engine rejected valid action: {result.get('reason')}"
    else:
        pytest.fail(f"Game did not end within {max_steps} steps")

    assert state.get("game_over"), "Game should have ended"
    assert state.get("winner") in (1, 2), "Game must have a winner"
    assert wall_count > 0, "No walls were placed during the game"


# ---------------------------------------------------------------------------
# Test 4: Validity — all actions produced by V2 are accepted by Engine
# ---------------------------------------------------------------------------

def test_v2_actions_always_valid():
    """Every action produced by RandomAgentV2 must be accepted by the Engine.

    Simulates 50 turns and verifies no rejections.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "backend-server"))
    from backend.adapters.engine_adapter import EngineAdapter

    random.seed(7)

    adapter = EngineAdapter()
    adapter.initialize()
    agents = {1: RandomAgentV2(), 2: RandomAgentV2()}

    for step in range(50):
        state = adapter.get_state()
        if state.get("game_over"):
            break
        seat = state["current_player"]
        legal = adapter.legal_pawn_actions()
        action = agents[seat].make_action(state, legal)
        result = adapter.take_action(action)
        assert result["success"], (
            f"Step {step}: Engine rejected action {action}: {result.get('reason')}"
        )


# ---------------------------------------------------------------------------
# Test 5: Wire format — returned action has correct structure
# ---------------------------------------------------------------------------

def test_v2_action_wire_format():
    """Every action must have the correct wire format keys and value types."""
    agent = RandomAgentV2(threshold=0.5)
    random.seed(0)

    for _ in range(50):
        action = agent.make_action(_INITIAL_STATE, _P1_LEGAL_PAWN_ACTIONS)
        assert "player" in action
        assert "type" in action
        assert "target" in action
        assert action["player"] == 1
        assert action["type"] in ("pawn", "horizontal", "vertical")
        assert isinstance(action["target"], list)
        assert len(action["target"]) == 2


# ---------------------------------------------------------------------------
# Test: RandomAgent V1 still works (regression)
# ---------------------------------------------------------------------------

def test_v1_selects_from_legal_actions():
    """RandomAgent V1 must return an action from the provided legal_actions."""
    agent = RandomAgent()
    random.seed(42)

    for _ in range(20):
        action = agent.make_action(_INITIAL_STATE, _P1_LEGAL_PAWN_ACTIONS)
        assert action in _P1_LEGAL_PAWN_ACTIONS
