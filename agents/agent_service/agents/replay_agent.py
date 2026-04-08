"""Replay Agent — emits predetermined actions from a replay sequence.

Follows the same gameplay path as all other agents:
Backend → Replay Agent → Backend → Engine

Cursor advancement:
- make_action() returns the current action WITHOUT advancing the cursor.
- advance() must be called after the action is ACCEPTED by the engine.
- On rejection, the same action is re-emitted on the next make_action() call.
"""

from __future__ import annotations

from typing import Any

from agents.agent_service.base_agent import BaseAgent


class ReplayAgent(BaseAgent):
    type_id = "replay"
    display_name = "Replay Agent"
    category = "replay"

    def __init__(self) -> None:
        self._actions: list[dict] = []
        self._cursor: int = 0

    def configure(self, config: dict[str, Any]) -> None:
        actions = config.get("actions")
        if not isinstance(actions, list):
            raise ValueError("Replay agent requires 'actions' list in config")
        self._actions = actions
        self._cursor = 0

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        """Return the current action WITHOUT advancing the cursor.

        The caller must call advance() after a successful engine accept.
        """
        if self._cursor >= len(self._actions):
            raise RuntimeError("Replay sequence exhausted — no more actions")
        return self._actions[self._cursor]

    def advance(self) -> None:
        """Advance the cursor to the next action. Call after engine ACCEPT."""
        self._cursor += 1

    def reset(self) -> None:
        self._cursor = 0

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self._actions)
