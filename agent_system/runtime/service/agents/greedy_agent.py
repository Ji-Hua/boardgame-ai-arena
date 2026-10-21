"""Greedy Agent — one-step lookahead using the local Rust Rule Engine.

Generates the full action space (pawn moves + wall placements) via a local
Rule Engine, evaluates each action by simulating it and computing shortest
path lengths for both players.  Selects the action that maximises:
delta_opp_path - delta_own_path.

Supports an optional top-k policy for controlled randomness.  When no
policy is configured the agent is strictly deterministic (identical to
the original implementation).
"""

from __future__ import annotations

import random
from typing import Any

from quoridor_engine import Action, Orientation, Player, RawState, RuleEngine
from quoridor_engine import calculation

from agent_system.runtime.service.base_agent import BaseAgent
from agent_system.runtime.service.policy import Policy

_ENGINE = RuleEngine.standard()
_TOPO = _ENGINE.topology
_INF = 999


def _seat_to_player(seat: int) -> Player:
    return Player.P1 if seat == 1 else Player.P2


def _action_to_engine(action_dict: dict) -> Action:
    player = _seat_to_player(action_dict["player"])
    row, col = action_dict["target"]
    kind = action_dict["type"]
    if kind == "pawn":
        return Action.move_pawn(player, row, col)
    elif kind == "horizontal":
        return Action.place_wall(player, row, col, Orientation.Horizontal)
    elif kind == "vertical":
        return Action.place_wall(player, row, col, Orientation.Vertical)
    raise ValueError(f"Unknown action type: {kind}")


def _engine_action_to_wire(action, seat: int) -> dict:
    """Convert an engine Action to wire format dict."""
    kind = str(action.kind)
    if kind == "MovePawn":
        return {"player": seat, "type": "pawn", "target": [action.target_x, action.target_y]}
    coord = action.coordinate_kind
    wire_type = "horizontal" if coord == "Horizontal" else "vertical"
    return {"player": seat, "type": wire_type, "target": [action.target_x, action.target_y]}


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


def _path_len(state: RawState, player: Player) -> int:
    result = calculation.shortest_path_len(state, player, _TOPO)
    return result if result is not None else _INF


def score_position(state: RawState) -> list[dict]:
    """Score engine-legal actions, with the distance evidence used by Greedy.

    This in-process entry point is shared by serving and Decision Lab. Order
    remains the engine's order; equal scores retain the first candidate.
    """
    me = state.current_player
    opp = me.opponent()
    before = (_path_len(state, me), _path_len(state, opp))
    results = []
    for action in _ENGINE.legal_actions(state):
        try:
            after = _ENGINE.apply_action(state, action)
        except ValueError:
            continue
        distances = (_path_len(after, me), _path_len(after, opp))
        delta_self = distances[0] - before[0]
        delta_opp = distances[1] - before[1]
        results.append({
            "action": action, "state": after, "before": before,
            "after": distances, "delta_self": delta_self, "delta_opp": delta_opp,
            "score": float(delta_opp - delta_self),
        })
    return results


class GreedyAgent(BaseAgent):
    type_id = "greedy"
    display_name = "Greedy Agent"
    category = "ai"

    def __init__(
        self,
        policy: Policy | None = None,
        seed: int | None = None,
    ) -> None:
        self._policy = policy
        self._rng = random.Random(seed) if seed is not None else None

    def configure(self, config: dict[str, Any]) -> None:
        """Accept policy and seed via configure() for runtime updates."""
        from agent_system.runtime.service.policy import build_policy
        if "policy" in config:
            self._policy = build_policy(config["policy"])
        if "seed" in config:
            self._rng = random.Random(int(config["seed"]))

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        state = _build_state(game_state)
        seat = game_state["current_player"]

        scored_actions: list[tuple[dict, float]] = []
        best_score: float | None = None
        best_wire_action: dict | None = None

        for candidate in score_position(state):
            score = candidate["score"]
            wire_action = _engine_action_to_wire(candidate["action"], seat)

            if self._policy is not None:
                scored_actions.append((wire_action, score))

            if best_score is None or score > best_score:
                best_score = score
                best_wire_action = wire_action

        # Policy path: delegate selection to policy with injected RNG.
        if self._policy is not None and self._rng is not None and scored_actions:
            return self._policy.select(scored_actions, self._rng)

        # Default path: deterministic argmax (original behavior).
        if best_wire_action is not None:
            return best_wire_action
        return legal_actions[0]
