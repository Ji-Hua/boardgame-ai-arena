"""Dummy Agent — always picks the first pawn action.

Migrated from quoridor_v0/quoridor-agents. Adapted for new Agent Service interface.
"""

from __future__ import annotations

from agents.agent_service.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    type_id = "dummy"
    display_name = "Dummy Agent"
    category = "scripted"

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        pawn_actions = [a for a in legal_actions if a.get("type") == "pawn"]
        if pawn_actions:
            return pawn_actions[0]
        return legal_actions[0]
