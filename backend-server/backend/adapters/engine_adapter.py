"""Engine Adapter — interface between backend and game engine.

Translates between backend wire format (JSON-serializable dicts) and
opaque engine types. This is the ONLY module that imports the engine.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

# Ensure engine package is importable from sibling directory.
_quoridor_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _quoridor_root not in sys.path:
    sys.path.insert(0, _quoridor_root)

from engine.game_manager import GameManager
from engine.game_manager.types import ActionResult
import quoridor_engine
from quoridor_engine import Action, Player, Orientation


def _player_from_seat(seat: int) -> Player:
    if seat == 1:
        return Player.P1
    elif seat == 2:
        return Player.P2
    raise ValueError(f"Invalid seat: {seat}")


def _seat_from_player(player: Player) -> int:
    if player == Player.P1:
        return 1
    if player == Player.P2:
        return 2
    raise ValueError(f"Unknown player: {player}")


def _action_to_engine(action_dict: dict) -> Action:
    player = _player_from_seat(action_dict["player"])
    row, col = action_dict["target"]
    kind = action_dict["type"]

    if kind == "pawn":
        return Action.move_pawn(player, row, col)
    elif kind == "horizontal":
        return Action.place_wall(player, row, col, Orientation.Horizontal)
    elif kind == "vertical":
        return Action.place_wall(player, row, col, Orientation.Vertical)
    else:
        raise ValueError(f"Unknown action type: {kind}")


def _serialize_state(state) -> dict:
    p1_pos = state.pawn_pos(Player.P1)
    p2_pos = state.pawn_pos(Player.P2)
    return {
        "current_player": _seat_from_player(state.current_player),
        "pawns": {
            "1": {"row": p1_pos[0], "col": p1_pos[1]},
            "2": {"row": p2_pos[0], "col": p2_pos[1]},
        },
        "walls_remaining": {
            "1": state.walls_remaining(Player.P1),
            "2": state.walls_remaining(Player.P2),
        },
        "game_over": False,
        "winner": None,
    }


class EngineAdapter:
    """Wraps GameManager, translating wire dicts ↔ engine types."""

    def __init__(self) -> None:
        self._gm: Optional[GameManager] = None

    def initialize(self) -> None:
        self._gm = GameManager()
        self._gm.initialize()

    def get_state(self) -> dict:
        if self._gm is None:
            return {}
        state = self._gm.current_state()
        result = _serialize_state(state)
        game_over = self._gm.is_game_over()
        result["game_over"] = game_over
        if game_over:
            winner = self._gm.winner()
            result["winner"] = _seat_from_player(winner) if winner else None
        return result

    def take_action(self, action_dict: dict) -> dict:
        if self._gm is None:
            return {"success": False, "reason": "Engine not initialized"}

        try:
            engine_action = _action_to_engine(action_dict)
        except (ValueError, KeyError, OverflowError) as e:
            return {"success": False, "reason": f"Invalid action: {e}"}

        try:
            result: ActionResult = self._gm.submit_action(engine_action)
        except (OverflowError, ValueError) as e:
            return {"success": False, "reason": f"Invalid action: {e}"}
        if not result.success:
            return {"success": False, "reason": result.error}
        return {"success": True, "new_state": _serialize_state(result.state)}

    def validate_action(self, action_dict: dict) -> dict:
        if self._gm is None:
            return {"valid": False, "reason": "Engine not initialized"}

        try:
            engine_action = _action_to_engine(action_dict)
        except (ValueError, KeyError) as e:
            return {"valid": False, "reason": f"Invalid action: {e}"}

        # Use legal_actions to check validity
        legal = self._gm.legal_actions()
        # Compare by checking if the action matches any legal action
        for la in legal:
            if (la.player == engine_action.player and
                    la.target_x == engine_action.target_x and
                    la.target_y == engine_action.target_y and
                    str(la.kind) == str(engine_action.kind)):
                return {"valid": True, "reason": None}
        return {"valid": False, "reason": "Action is not legal"}

    def is_game_over(self) -> bool:
        if self._gm is None:
            return False
        return self._gm.is_game_over()

    def winner(self) -> Optional[int]:
        if self._gm is None:
            return None
        w = self._gm.winner()
        return _seat_from_player(w) if w else None
