"""Agent base class — common interface for all agent implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all Quoridor agents.

    Every agent receives GameState as a dict and returns an Action as a dict.
    Both follow the Backend wire format (JSON-serializable).
    """

    type_id: str = ""
    display_name: str = ""
    category: str = ""  # "ai", "replay", "scripted"

    def configure(self, config: dict[str, Any]) -> None:
        """Optional agent-specific configuration (e.g., replay data)."""
        pass

    @abstractmethod
    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        """Given a game state and legal actions, return the chosen action.

        Args:
            game_state: GameState dict from Backend (wire format).
            legal_actions: List of legal pawn action dicts.

        Returns:
            An Action dict in wire format:
            {"player": 1|2, "type": "pawn"|"horizontal"|"vertical", "target": [row, col]}
        """
        ...

    def reset(self) -> None:
        """Reset agent state between games."""
        pass

    def notify_result(self, result: dict) -> None:
        """Notify agent of game result (optional)."""
        pass
