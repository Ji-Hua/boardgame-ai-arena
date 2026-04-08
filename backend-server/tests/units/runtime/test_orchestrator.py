# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for the Turn Orchestrator retry contract.

Contract under test (documents/agent/agent-interface.md, Section 4):
  - Engine REJECT with reject_kind="INVALID_ACTION" → Backend retries RequestAction
  - Engine REJECT with reject_kind="GAME_END"       → Backend terminates immediately
  - AdvanceCursor is called ONLY after Engine ACCEPT
  - Retry limit exhausted → Backend force-ends the game
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.runtime.orchestrator import MAX_AGENT_RETRIES, maybe_trigger_agent_turn

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_VALID_ACTION = {"player": 1, "type": "pawn", "target": [4, 1]}

# State where it is P1's turn (seat 1 = agent in all below tests)
_STATE_P1_TURN = {
    "current_player": 1,
    "pawns": {"1": {"row": 4, "col": 0}, "2": {"row": 4, "col": 8}},
    "walls_remaining": {"1": 10, "2": 10},
    "game_over": False,
    "winner": None,
}

# State where it is P2's turn (seat 2 = human in all below tests → outer loop exits)
_STATE_P2_TURN = {**_STATE_P1_TURN, "current_player": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seat(actor_type: str = "agent") -> MagicMock:
    seat = MagicMock()
    seat.actor_type = actor_type
    return seat


def _make_room(*, seat1_type: str = "agent", seat2_type: str = "human") -> MagicMock:
    room = MagicMock()
    room.status = "using"
    room.current_game_id = "game-1"
    room.seats = {1: _make_seat(seat1_type), 2: _make_seat(seat2_type)}
    return room


def _make_game(*, phase: str = "running") -> MagicMock:
    game = MagicMock()
    game.game_id = "game-1"
    game.phase = phase
    game.step_count = 1
    game.speed_multiplier = 1.0
    game.result = None
    # Default: always return P1 turn.  Override with side_effect for tests that
    # need the outer loop to exit cleanly after a first successful action.
    game.get_state.return_value = dict(_STATE_P1_TURN)
    game.engine.legal_pawn_actions.return_value = [_VALID_ACTION]
    return game


def _make_deps(game: MagicMock) -> tuple:
    gm = MagicMock()
    gm.get_game.return_value = game
    gm.force_end.return_value = {"winner_seat": None, "termination": "forced"}

    rm = MagicMock()

    hub = MagicMock()
    hub.broadcast = AsyncMock()

    agent_adapter = MagicMock()
    agent_adapter.has_agent.return_value = True
    agent_adapter.request_action = AsyncMock(return_value=_VALID_ACTION)
    agent_adapter.advance_agent.return_value = None
    agent_adapter.destroy_room_agents.return_value = None

    return gm, rm, hub, agent_adapter


def _broadcast_types(hub: MagicMock) -> list[str]:
    return [c.args[1]["type"] for c in hub.broadcast.call_args_list]


# ---------------------------------------------------------------------------
# 1. Retry on INVALID_ACTION rejection
# ---------------------------------------------------------------------------


async def test_retry_on_invalid_action_rejection():
    """Backend retries RequestAction when Engine returns INVALID_ACTION.

    Sequence:
      1. RequestAction  → action A
      2. submit(A)      → INVALID_ACTION reject
      3. RequestAction  → action B   (retry)
      4. submit(B)      → success
    Postcondition: AdvanceCursor called once (after accept, not during retry).
    """
    game = _make_game()
    # After the accepted action the outer loop checks get_state again.
    # Return P2-turn state so the loop exits cleanly (seat 2 is human).
    game.get_state.side_effect = [_STATE_P1_TURN, _STATE_P2_TURN]

    gm, rm, hub, agent_adapter = _make_deps(game)

    reject = {"success": False, "error": "Illegal move", "reject_kind": "INVALID_ACTION"}
    accept = {"success": True, "state": _STATE_P2_TURN, "game_over": False, "result": None}
    gm.submit_action.side_effect = [reject, accept]

    room = _make_room()
    await maybe_trigger_agent_turn(room, "room-1", gm, rm, hub, agent_adapter)

    # RequestAction called twice: initial + one retry
    assert agent_adapter.request_action.call_count == 2
    # AdvanceCursor called once after accept (never during retry)
    assert agent_adapter.advance_agent.call_count == 1
    # state_update broadcast; no game_ended
    types = _broadcast_types(hub)
    assert "state_update" in types
    assert "game_ended" not in types


# ---------------------------------------------------------------------------
# 2. GAME_END rejection terminates immediately without retry
# ---------------------------------------------------------------------------


async def test_game_end_rejection_terminates_immediately():
    """Backend terminates the game when Engine returns reject_kind=GAME_END.

    No retry must be attempted.  game_ended must be broadcast.
    AdvanceCursor must NOT be called.
    """
    game = _make_game()
    game.result = {"winner_seat": None, "termination": "forced"}
    gm, rm, hub, agent_adapter = _make_deps(game)

    gm.submit_action.return_value = {
        "success": False,
        "error": "Game phase is finished, not running",
        "reject_kind": "GAME_END",
    }

    room = _make_room()
    await maybe_trigger_agent_turn(room, "room-1", gm, rm, hub, agent_adapter)

    # RequestAction called exactly once — no retry
    assert agent_adapter.request_action.call_count == 1
    # submit_action called exactly once
    assert gm.submit_action.call_count == 1
    # AdvanceCursor never called
    agent_adapter.advance_agent.assert_not_called()
    # game_ended broadcast; no state_update
    types = _broadcast_types(hub)
    assert "game_ended" in types
    assert "state_update" not in types


# ---------------------------------------------------------------------------
# 3. Replay agent: cursor NOT advanced on rejection, only on ACCEPT
# ---------------------------------------------------------------------------


async def test_replay_agent_cursor_not_advanced_on_rejection():
    """AdvanceCursor is NOT called during retry; only after Engine ACCEPT.

    Verifies the core property of the Replay Agent contract: cursor separation
    between action production and acceptance confirmation.

    Sequence:
      1. RequestAction  → action A
      2. submit(A)      → INVALID_ACTION
      3. RequestAction  → action A  (same — cursor unchanged)
      4. submit(A)      → success
    Postcondition: advance_agent called exactly once (step 4), never between 1-3.
    """
    game = _make_game()
    game.get_state.side_effect = [_STATE_P1_TURN, _STATE_P2_TURN]

    gm, rm, hub, agent_adapter = _make_deps(game)

    reject = {"success": False, "error": "Illegal", "reject_kind": "INVALID_ACTION"}
    accept = {"success": True, "state": _STATE_P2_TURN, "game_over": False, "result": None}
    gm.submit_action.side_effect = [reject, accept]

    # Track advance_agent call count precisely
    advance_call_count: list[int] = [0]

    def _advance(room_id: str, seat: int) -> None:
        advance_call_count[0] += 1

    agent_adapter.advance_agent = _advance

    room = _make_room()
    await maybe_trigger_agent_turn(room, "room-1", gm, rm, hub, agent_adapter)

    # submit_action called twice (one reject + one accept)
    assert gm.submit_action.call_count == 2
    # advance_agent called exactly once (only after accept)
    assert advance_call_count[0] == 1
    # state_update broadcast
    types = _broadcast_types(hub)
    assert "state_update" in types


# ---------------------------------------------------------------------------
# 4. Retry limit exhaustion force-ends the game
# ---------------------------------------------------------------------------


async def test_retry_limit_exhausted_force_ends_game():
    """Backend force-ends the game when retry limit is exhausted.

    The agent always returns an invalid action.  After MAX_AGENT_RETRIES retries
    the game must be force-ended and game_ended broadcast.
    AdvanceCursor must NOT be called.
    """
    game = _make_game()
    game.result = {"winner_seat": None, "termination": "forced"}
    gm, rm, hub, agent_adapter = _make_deps(game)

    reject = {"success": False, "error": "Always invalid", "reject_kind": "INVALID_ACTION"}
    gm.submit_action.return_value = reject

    room = _make_room()
    await maybe_trigger_agent_turn(room, "room-1", gm, rm, hub, agent_adapter)

    # submit_action called: 1 initial + MAX_AGENT_RETRIES retries
    assert gm.submit_action.call_count == MAX_AGENT_RETRIES + 1
    # AdvanceCursor never called (no accept occurred)
    agent_adapter.advance_agent.assert_not_called()
    # force_end called once
    gm.force_end.assert_called_once_with(game)
    # game_ended broadcast; no state_update
    types = _broadcast_types(hub)
    assert "game_ended" in types
    assert "state_update" not in types
