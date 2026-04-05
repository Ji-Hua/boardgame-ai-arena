"""E2E Replay Test — Full game from full_game_replay.md played through backend.

Local Mode: single WebSocket connection controls both seats.
All steps are hardcoded from the replay document.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from backend.main import create_app


# ---------------------------------------------------------------------------
# Replay steps encoded from full_game_replay.md
#
# Each tuple: (step_label, player, action_type, target, expected)
#   action_type: "pawn" | "horizontal" | "vertical"
#   target: [x, y]
#   expected: "accept" | "reject"
# ---------------------------------------------------------------------------

REPLAY_STEPS: list[tuple[str, int, str, list[int], str]] = [
    # Opening and Early Confrontation
    ("1",    1, "pawn", [4, 1], "accept"),
    ("2",    2, "pawn", [4, 7], "accept"),
    ("3",    1, "pawn", [4, 2], "accept"),
    ("4",    2, "pawn", [4, 6], "accept"),
    ("5",    1, "pawn", [4, 3], "accept"),
    ("6",    2, "pawn", [4, 5], "accept"),
    ("6.1",  1, "pawn", [4, 5], "reject"),  # jump while not adjacent
    ("7",    1, "pawn", [4, 4], "accept"),
    ("7.1",  2, "pawn", [4, 4], "reject"),  # occupied square
    ("7.2",  1, "pawn", [4, 5], "reject"),  # wrong player (still P2 turn)
    ("8",    2, "pawn", [4, 3], "accept"),  # legal direct jump
    ("8.1",  1, "pawn", [5, 3], "reject"),  # side jump when direct available
    ("8.2",  1, "pawn", [3, 3], "reject"),  # side jump when direct available
    ("9",    1, "pawn", [4, 2], "accept"),  # legal direct jump
    ("10",   2, "horizontal", [4, 2], "accept"),

    # After step 10: P1@(4,2), P2@(4,3), wall (4,2,H)
    ("10.1", 1, "pawn", [4, 4], "reject"),  # wall blocks jump
    ("10.2", 1, "pawn", [3, 3], "reject"),  # side-jump not satisfied
    ("11",   1, "pawn", [3, 2], "accept"),
    ("12",   2, "horizontal", [4, 3], "accept"),
    ("13",   1, "pawn", [3, 3], "accept"),
    ("13.1", 2, "pawn", [3, 3], "reject"),  # occupied
    ("13.2", 2, "pawn", [4, 2], "reject"),  # blocked by wall
    ("14",   2, "pawn", [2, 3], "accept"),  # legal direct jump
    ("15",   1, "pawn", [3, 4], "accept"),
    ("16",   2, "pawn", [3, 3], "accept"),
    ("17",   1, "horizontal", [7, 0], "accept"),
    ("18",   2, "pawn", [3, 2], "accept"),
    ("18.1", 1, "horizontal", [4, 3], "reject"),  # exact overlap
    ("18.2", 1, "vertical", [4, 3], "reject"),    # crossing wall
    ("18.3", 1, "horizontal", [3, 3], "reject"),  # segment overlap

    # After step 18: P1@(3,4), P2@(3,2)
    ("19",   1, "vertical", [2, 4], "accept"),
    ("19.1", 2, "vertical", [8, 5], "reject"),    # out of bounds
    ("19.2", 2, "horizontal", [8, 5], "reject"),  # out of bounds
    ("20",   2, "pawn", [2, 2], "accept"),
    ("20.1", 1, "pawn", [2, 4], "reject"),  # blocked by wall (2,4,V)
    ("21",   1, "vertical", [0, 1], "accept"),
    ("21.1", 2, "pawn", [0, 2], "reject"),  # two-square move
    ("22",   2, "pawn", [1, 2], "accept"),
    ("23",   1, "pawn", [3, 5], "accept"),
    ("23.1", 2, "pawn", [0, 2], "reject"),  # blocked by wall (0,1,V)
    ("24",   2, "pawn", [1, 1], "accept"),
    ("24.1", 1, "pawn", [3, 7], "reject"),  # two-square move
    ("25",   1, "pawn", [3, 6], "accept"),
    ("25.1", 2, "horizontal", [3, 8], "reject"),  # out of bounds
    ("25.2", 2, "vertical", [3, 8], "reject"),    # out of bounds
    ("26",   2, "pawn", [1, 2], "accept"),
    ("26.1", 1, "pawn", [3, 8], "reject"),  # two-square move
    ("26.2", 1, "pawn", [2, 7], "reject"),  # diagonal move
    ("26.3", 1, "pawn", [5, 7], "reject"),  # illegal distant move
    ("26.4", 1, "pawn", [3, 4], "reject"),  # two-square move
    ("27",   1, "horizontal", [3, 6], "accept"),
    ("27.1", 2, "horizontal", [3, 6], "reject"),  # exact overlap
    ("27.2", 2, "horizontal", [4, 6], "reject"),  # segment overlap
    ("27.3", 2, "vertical", [2, 5], "reject"),    # segment overlap

    # After step 28: P1@(3,6), P2@(1,3)
    ("28",   2, "pawn", [1, 3], "accept"),
    ("28.1", 1, "pawn", [3, 7], "reject"),  # blocked by wall (3,6,H)
    ("29",   1, "pawn", [2, 6], "accept"),
    ("30",   2, "pawn", [1, 4], "accept"),
    ("31",   1, "pawn", [2, 7], "accept"),
    ("32",   2, "horizontal", [2, 7], "accept"),
    ("32.1", 1, "pawn", [2, 8], "reject"),  # blocked by wall (2,7,H)
    ("33",   1, "pawn", [3, 7], "accept"),
    ("33.1", 2, "pawn", [-1, 4], "reject"),  # out of bounds
    ("34",   2, "pawn", [1, 5], "accept"),
    ("35",   1, "pawn", [4, 7], "accept"),
    ("35.1", 2, "pawn", [1, 7], "reject"),  # two-square move
    ("36",   2, "horizontal", [4, 7], "accept"),
    ("36.1", 1, "pawn", [4, 8], "reject"),  # blocked by wall
    ("37",   1, "pawn", [5, 7], "accept"),
    ("37.1", 2, "horizontal", [4, 7], "reject"),  # exact overlap
    ("38",   2, "pawn", [1, 6], "accept"),

    # After step 38: P1@(5,7), P2@(1,6)
    ("38.1", 1, "pawn", [5, 9], "reject"),  # out of bounds
    ("39",   1, "pawn", [6, 7], "accept"),
    ("40",   2, "pawn", [1, 7], "accept"),
    ("40.1", 1, "pawn", [8, 7], "reject"),  # two-square move
    ("41",   1, "horizontal", [6, 7], "accept"),
    ("42",   2, "pawn", [1, 8], "accept"),  # NOT game end (P2 wins at row 0)
    ("43",   1, "pawn", [5, 7], "accept"),
    ("43.1", 2, "pawn", [1, 9], "reject"),  # out of bounds
    ("44",   2, "pawn", [1, 7], "accept"),
    ("45",   1, "pawn", [4, 7], "accept"),
    ("46",   2, "vertical", [0, 6], "accept"),
    ("47",   1, "horizontal", [1, 5], "accept"),
    ("48",   2, "horizontal", [3, 5], "accept"),

    # After step 48: P1@(4,7), P2@(1,7)
    ("49",   1, "horizontal", [5, 5], "accept"),
    ("50",   2, "horizontal", [7, 5], "accept"),
    ("50.1", 1, "horizontal", [8, 7], "reject"),  # out of bounds (2nd segment)
    ("51",   1, "vertical", [7, 6], "accept"),
    ("51.1", 2, "horizontal", [0, 7], "reject"),  # would block all paths
    ("52",   2, "horizontal", [1, 6], "accept"),
    ("53",   1, "pawn", [3, 7], "accept"),
    ("54",   2, "pawn", [2, 7], "accept"),
    ("55",   1, "pawn", [1, 7], "accept"),  # legal horizontal direct jump
    ("56",   2, "pawn", [3, 7], "accept"),
    ("57",   1, "pawn", [2, 7], "accept"),
    ("57.1", 2, "vertical", [1, 7], "reject"),  # would block all paths
    ("57.2", 2, "vertical", [2, 7], "reject"),  # cross wall
    ("58",   2, "vertical", [3, 7], "accept"),

    # After step 58: P1@(2,7), P2@(3,7)
    ("58.1", 1, "pawn", [4, 7], "reject"),  # jump blocked by walls
    ("58.2", 1, "pawn", [3, 8], "reject"),  # diagonal blocked
    ("58.3", 1, "pawn", [3, 6], "reject"),  # diagonal blocked
    ("58.4", 1, "vertical", [2, 6], "reject"),  # would block all paths
    ("59",   1, "horizontal", [2, 3], "accept"),  # random legal wall
    ("60",   2, "pawn", [1, 7], "accept"),  # legal straight jump
    ("60.1", 1, "horizontal", [0, 3], "reject"),  # would block all paths
    ("61",   1, "pawn", [1, 8], "accept"),  # P1 reaches target row — WINS

    # Post-game — all actions must be rejected
    ("61.1", 2, "pawn", [2, 7], "reject"),       # game ended
    ("61.2", 2, "horizontal", [3, 0], "reject"),  # game ended
    ("61.3", 2, "vertical", [3, 0], "reject"),    # game ended
]


# ---------------------------------------------------------------------------
# State snapshots from the replay document for validation at key steps
# ---------------------------------------------------------------------------

STATE_SNAPSHOTS = {
    # After step 10
    10: {
        "p1": (4, 2), "p2": (4, 3),
        "w1_remaining": 10, "w2_remaining": 9,
    },
    # After step 18
    18: {
        "p1": (3, 4), "p2": (3, 2),
        "w1_remaining": 9, "w2_remaining": 8,
    },
    # After step 28
    28: {
        "p1": (3, 6), "p2": (1, 3),
        "w1_remaining": 6, "w2_remaining": 8,
    },
    # After step 38
    38: {
        "p1": (5, 7), "p2": (1, 6),
        "w1_remaining": 6, "w2_remaining": 6,
    },
    # After step 48
    48: {
        "p1": (4, 7), "p2": (1, 7),
        "w1_remaining": 4, "w2_remaining": 4,
    },
    # After step 58
    58: {
        "p1": (2, 7), "p2": (3, 7),
        "w1_remaining": 2, "w2_remaining": 1,
    },
}


def _setup_room_and_game(client: TestClient) -> str:
    """Create room, bind seats, start game. Return room_id."""
    r = client.post("/api/rooms")
    assert r.status_code == 200
    room_id = r.json()["room_id"]

    client.post(f"/api/rooms/{room_id}/join", json={"client_id": "local", "seat": 1})
    client.post(f"/api/rooms/{room_id}/join", json={"client_id": "local2", "seat": 2})
    client.post(f"/api/rooms/{room_id}/select_actor", json={"seat": 1, "actor_type": "human"})
    client.post(f"/api/rooms/{room_id}/select_actor", json={"seat": 2, "actor_type": "human"})

    r = client.post(f"/api/rooms/{room_id}/start_game")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "using"

    # Validate initial state
    state = data["game"]["state"]
    assert state["current_player"] == 1
    assert state["pawns"]["1"] == {"row": 4, "col": 0}
    assert state["pawns"]["2"] == {"row": 4, "col": 8}
    assert state["walls_remaining"]["1"] == 10
    assert state["walls_remaining"]["2"] == 10

    return room_id


def test_full_game_replay_e2e():
    """Play the entire full_game_replay.md through the backend via a single WS connection (Local Mode)."""

    app = create_app()
    client = TestClient(app)
    room_id = _setup_room_and_game(client)

    # Track accepted step count for snapshot validation
    accepted_step_count = 0
    last_state = None
    game_ended = False

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        # Subscribe (Local Mode — one connection, both seats)
        ws.send_json({"type": "subscribe", "client_id": "local"})
        snapshot = ws.receive_json()
        assert snapshot["type"] == "room_snapshot"
        assert snapshot["status"] == "using"

        for step_label, player, action_type, target, expected in REPLAY_STEPS:
            action = {
                "player": player,
                "type": action_type,
                "target": target,
            }

            ws.send_json({"type": "take_action", "action": action})
            result = ws.receive_json()

            if expected == "accept":
                assert result["type"] == "action_result", (
                    f"Step {step_label}: expected action_result, got {result}"
                )
                assert result["success"] is True, (
                    f"Step {step_label}: expected ACCEPT, got REJECT: {result.get('error')}"
                )

                # Must receive state_update
                state_update = ws.receive_json()
                assert state_update["type"] == "state_update", (
                    f"Step {step_label}: expected state_update, got {state_update['type']}"
                )
                last_state = state_update["state"]
                accepted_step_count += 1

                assert state_update["step_count"] == accepted_step_count, (
                    f"Step {step_label}: step_count mismatch: "
                    f"expected {accepted_step_count}, got {state_update['step_count']}"
                )

                # Check state snapshots at key steps
                if accepted_step_count in STATE_SNAPSHOTS:
                    snap = STATE_SNAPSHOTS[accepted_step_count]
                    p1 = last_state["pawns"]["1"]
                    p2 = last_state["pawns"]["2"]
                    assert (p1["row"], p1["col"]) == snap["p1"], (
                        f"Snapshot step {accepted_step_count}: P1 position mismatch: "
                        f"expected {snap['p1']}, got ({p1['row']}, {p1['col']})"
                    )
                    assert (p2["row"], p2["col"]) == snap["p2"], (
                        f"Snapshot step {accepted_step_count}: P2 position mismatch: "
                        f"expected {snap['p2']}, got ({p2['row']}, {p2['col']})"
                    )
                    assert last_state["walls_remaining"]["1"] == snap["w1_remaining"], (
                        f"Snapshot step {accepted_step_count}: P1 walls mismatch"
                    )
                    assert last_state["walls_remaining"]["2"] == snap["w2_remaining"], (
                        f"Snapshot step {accepted_step_count}: P2 walls mismatch"
                    )

                # Check for game_ended event (step 61 is the winning move)
                if step_label == "61":
                    game_ended_msg = ws.receive_json()
                    assert game_ended_msg["type"] == "game_ended", (
                        f"Step {step_label}: expected game_ended, got {game_ended_msg}"
                    )
                    assert game_ended_msg["result"]["winner_seat"] == 1, (
                        f"Step {step_label}: expected P1 wins, got {game_ended_msg['result']}"
                    )
                    assert game_ended_msg["result"]["termination"] == "goal"
                    game_ended = True

            elif expected == "reject":
                # For post-game rejections, the backend may return an error or action_result
                if game_ended:
                    # After game end, the game phase is "finished", so we expect
                    # either an action_result with success=false or an error event
                    assert result.get("success") is False or result.get("type") == "error", (
                        f"Step {step_label}: expected rejection after game end, got {result}"
                    )
                else:
                    assert result["type"] == "action_result", (
                        f"Step {step_label}: expected action_result, got {result}"
                    )
                    assert result["success"] is False, (
                        f"Step {step_label}: expected REJECT, got ACCEPT"
                    )

                # Must NOT receive state_update after a rejection
                # (no additional message should be queued)

    # Final assertions
    assert game_ended, "Game should have ended with P1 winning at step 61"
    assert accepted_step_count == 61, f"Expected 61 accepted steps, got {accepted_step_count}"


def test_local_mode_single_connection():
    """Verify that Local Mode works: one connection sends actions for both seats."""

    app = create_app()
    client = TestClient(app)
    room_id = _setup_room_and_game(client)

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        ws.send_json({"type": "subscribe", "client_id": "local"})
        ws.receive_json()  # room_snapshot

        # P1 moves
        ws.send_json({"type": "take_action", "action": {"player": 1, "type": "pawn", "target": [4, 1]}})
        r = ws.receive_json()
        assert r["success"] is True
        ws.receive_json()  # state_update

        # Same connection, P2 moves
        ws.send_json({"type": "take_action", "action": {"player": 2, "type": "pawn", "target": [4, 7]}})
        r = ws.receive_json()
        assert r["success"] is True
        ws.receive_json()  # state_update

        # Wrong turn: P2 again (should be P1's turn)
        ws.send_json({"type": "take_action", "action": {"player": 2, "type": "pawn", "target": [4, 6]}})
        r = ws.receive_json()
        assert r["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
