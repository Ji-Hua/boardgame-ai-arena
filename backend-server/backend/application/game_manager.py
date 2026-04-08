"""Game Manager — game lifecycle within a room.

Owns per-game state. Delegates all rule decisions to the engine adapter.
"""

from __future__ import annotations

import uuid
from typing import Optional, Literal

from backend.adapters.engine_adapter import EngineAdapter


class Game:
    """A single game instance within a room."""

    def __init__(self, game_id: str, engine: EngineAdapter) -> None:
        self.game_id: str = game_id
        self.phase: Literal["starting", "running", "ending", "finished"] = "starting"
        self.engine: EngineAdapter = engine
        self.step_count: int = 0
        self.actions: list[dict] = []
        self.result: Optional[dict] = None
        self.speed_multiplier: float = 1.0

    def get_state(self) -> dict:
        return self.engine.get_state()

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "phase": self.phase,
            "state": self.get_state(),
        }


class GameManager:
    """Creates and manages Game instances."""

    def __init__(self) -> None:
        self._games: dict[str, Game] = {}

    def create_game(self) -> Game:
        game_id = str(uuid.uuid4())
        engine = EngineAdapter()
        engine.initialize()
        game = Game(game_id, engine)
        game.phase = "running"
        self._games[game_id] = game
        return game

    def get_game(self, game_id: str) -> Optional[Game]:
        return self._games.get(game_id)

    def submit_action(self, game: Game, action: dict) -> dict:
        """Submit an action to the game engine.

        Returns: {"success": bool, "error": str|None, "reject_kind": str|None,
                  "state": dict|None, "game_over": bool, "result": dict|None}

        reject_kind is present only on failure:
          "GAME_END"       — game is not running; no retry should be attempted
          "INVALID_ACTION" — engine-level rejection; retry is appropriate
        """
        if game.phase != "running":
            return {
                "success": False,
                "error": f"Game phase is {game.phase}, not running",
                "reject_kind": "GAME_END",
            }

        # Check turn order
        state = game.engine.get_state()
        if action.get("player") != state.get("current_player"):
            return {
                "success": False,
                "error": f"Not your turn. Current turn: seat {state['current_player']}",
                "reject_kind": "INVALID_ACTION",
            }

        result = game.engine.take_action(action)
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("reason", "Invalid action"),
                "reject_kind": "INVALID_ACTION",
            }

        game.step_count += 1
        game.actions.append(action)

        new_state = game.engine.get_state()
        response: dict = {"success": True, "state": new_state, "game_over": False, "result": None}

        if game.engine.is_game_over():
            winner = game.engine.winner()
            game.result = {"winner_seat": winner, "termination": "goal"}
            game.phase = "finished"
            response["game_over"] = True
            response["result"] = game.result

        return response

    def surrender(self, game: Game, loser_seat: int) -> dict:
        winner_seat = 2 if loser_seat == 1 else 1
        game.result = {"winner_seat": winner_seat, "termination": "surrender"}
        game.phase = "finished"
        return game.result

    def force_end(self, game: Game) -> dict:
        game.result = {"winner_seat": None, "termination": "forced"}
        game.phase = "finished"
        return game.result
