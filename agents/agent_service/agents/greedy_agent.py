"""Greedy Agent — picks the pawn move that minimizes distance to goal row.

Migrated from quoridor_v0/quoridor-agents. Engine dependency removed;
uses only GameState information from Backend.
"""

from __future__ import annotations

from agents.agent_service.base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    type_id = "greedy"
    display_name = "Greedy Agent"
    category = "ai"

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        current_player = game_state.get("current_player", 1)
        # Goal row: player 1 goes to row 8, player 2 goes to row 0
        goal_row = 8 if current_player == 1 else 0

        pawn_actions = [a for a in legal_actions if a.get("type") == "pawn"]
        if not pawn_actions:
            return legal_actions[0]

        best = min(pawn_actions, key=lambda a: abs(a["target"][0] - goal_row))
        return best
