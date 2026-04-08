"""Random Agent — selects actions randomly.

RandomAgent: uniformly samples from the provided legal_actions (pawn moves only).
RandomAgentV2: uses the local Rust Rule Engine to generate the full action space
(pawn moves + wall placements) and samples with a configurable pawn bias.
"""

from __future__ import annotations

import random

from quoridor_engine import Player, RawState, RuleEngine

from agents.agent_service.base_agent import BaseAgent

_ENGINE = RuleEngine.standard()


def _seat_to_player(seat: int) -> Player:
    return Player.P1 if seat == 1 else Player.P2


def _build_state(game_state: dict) -> RawState:
    """Reconstruct a RawState from the backend wire game_state dict."""
    pawns = game_state["pawns"]
    walls_remaining = game_state["walls_remaining"]
    ws = game_state.get("wall_state", {})
    return RawState(
        pawns["1"]["row"], pawns["1"]["col"],
        pawns["2"]["row"], pawns["2"]["col"],
        int(walls_remaining["1"]),
        int(walls_remaining["2"]),
        int(ws.get("horizontal_edges", 0)),
        int(ws.get("vertical_edges", 0)),
        int(ws.get("horizontal_heads", 0)),
        int(ws.get("vertical_heads", 0)),
        _seat_to_player(game_state["current_player"]),
    )


def _engine_action_to_wire(action, seat: int) -> dict:
    """Convert an engine Action to wire format dict."""
    kind = str(action.kind)
    if kind == "MovePawn":
        return {"player": seat, "type": "pawn", "target": [action.target_x, action.target_y]}
    # PlaceWall
    coord = action.coordinate_kind  # "Horizontal" or "Vertical"
    wire_type = "horizontal" if coord == "Horizontal" else "vertical"
    return {"player": seat, "type": wire_type, "target": [action.target_x, action.target_y]}


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
        seat = game_state["current_player"]
        state = _build_state(game_state)
        all_actions = _ENGINE.legal_actions(state)

        pawn_actions = []
        wall_actions = []
        for a in all_actions:
            wire = _engine_action_to_wire(a, seat)
            if wire["type"] == "pawn":
                pawn_actions.append(wire)
            else:
                wall_actions.append(wire)

        if pawn_actions and wall_actions:
            if random.random() < self._threshold:
                return random.choice(pawn_actions)
            else:
                return random.choice(wall_actions)
        if pawn_actions:
            return random.choice(pawn_actions)
        if wall_actions:
            return random.choice(wall_actions)
        return random.choice(legal_actions)
