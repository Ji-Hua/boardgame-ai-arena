"""Random Agent — selects actions randomly.

Migrated from quoridor_v0/quoridor-agents. Adapted for new Agent Service interface.
"""

from __future__ import annotations

import random

from agents.agent_service.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    type_id = "random"
    display_name = "Random Agent"
    category = "scripted"

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        return random.choice(legal_actions)


class RandomAgentV2(BaseAgent):
    type_id = "random_v2"
    display_name = "Random Agent V2"
    category = "scripted"

    def __init__(self, threshold: float = 0.8) -> None:
        self._threshold = threshold

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        pawn_actions = [a for a in legal_actions if a.get("type") == "pawn"]
        wall_actions = [a for a in legal_actions if a.get("type") in ("horizontal", "vertical")]
        if pawn_actions and wall_actions:
            if random.random() < self._threshold:
                return random.choice(pawn_actions)
            else:
                return random.choice(wall_actions)
        return random.choice(legal_actions)
