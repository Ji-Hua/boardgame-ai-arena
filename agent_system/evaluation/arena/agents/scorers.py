"""Scorer wrappers — reuse existing agent logic to produce [(action, score)]."""

from __future__ import annotations

from quoridor_engine import Player, RawState, RuleEngine
from quoridor_engine import calculation

from agent_system.evaluation.arena.agents.core import Scorer

_ENGINE = RuleEngine.standard()
_TOPO = _ENGINE.topology
_INF = 999
_MINIMAX_INF = 999_999


# ---------------------------------------------------------------------------
# Shared helpers (extracted from existing agent modules)
# ---------------------------------------------------------------------------

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
    coord = action.coordinate_kind
    wire_type = "horizontal" if coord == "Horizontal" else "vertical"
    return {"player": seat, "type": wire_type, "target": [action.target_x, action.target_y]}


def _path_len(state: RawState, player: Player) -> int:
    result = calculation.shortest_path_len(state, player, _TOPO)
    return result if result is not None else _INF


# ---------------------------------------------------------------------------
# RandomScorer
# ---------------------------------------------------------------------------

class RandomScorer(Scorer):
    """Assigns a constant score to every legal action."""

    def score(self, state: dict) -> list[tuple[dict, float]]:
        raw = _build_state(state)
        seat = state["current_player"]
        all_actions = _ENGINE.legal_actions(raw)
        return [(_engine_action_to_wire(a, seat), 1.0) for a in all_actions]


# ---------------------------------------------------------------------------
# GreedyScorer — reuses existing one-step lookahead logic
# ---------------------------------------------------------------------------

class GreedyScorer(Scorer):
    """One-step lookahead: score = delta_opp_path - delta_own_path."""

    def score(self, state: dict) -> list[tuple[dict, float]]:
        raw = _build_state(state)
        me = raw.current_player
        opp = me.opponent()
        seat = state["current_player"]

        my_before = _path_len(raw, me)
        opp_before = _path_len(raw, opp)

        results: list[tuple[dict, float]] = []
        for engine_action in _ENGINE.legal_actions(raw):
            try:
                next_state = _ENGINE.apply_action(raw, engine_action)
            except ValueError:
                continue
            delta_my = _path_len(next_state, me) - my_before
            delta_opp = _path_len(next_state, opp) - opp_before
            score = float(delta_opp - delta_my)
            results.append((_engine_action_to_wire(engine_action, seat), score))

        return results


# ---------------------------------------------------------------------------
# MinimaxScorer — reuses existing minimax + alpha-beta logic
# ---------------------------------------------------------------------------

class MinimaxScorer(Scorer):
    """Depth-limited minimax with alpha-beta pruning.

    Params:
        depth: search depth (default 2).
    """

    def __init__(self, depth: int = 2) -> None:
        self._depth = depth

    def score(self, state: dict) -> list[tuple[dict, float]]:
        raw = _build_state(state)
        seat = state["current_player"]
        maximizing_player = raw.current_player

        all_engine_actions = _ENGINE.legal_actions(raw)
        alpha = float("-inf")
        beta = float("inf")

        results: list[tuple[dict, float]] = []
        for engine_action in all_engine_actions:
            try:
                next_state = _ENGINE.apply_action(raw, engine_action)
            except ValueError:
                continue

            value = _alphabeta(
                next_state,
                self._depth - 1,
                alpha,
                beta,
                False,
                maximizing_player,
            )
            results.append((_engine_action_to_wire(engine_action, seat), value))
            alpha = max(alpha, value)

        return results


# ---------------------------------------------------------------------------
# Alpha-beta internals (reused from minimax_agent.py)
# ---------------------------------------------------------------------------

def _evaluate(state: RawState, maximizing_player: Player) -> float:
    """Evaluate from the perspective of *maximizing_player*."""
    winner = _ENGINE.winner(state)
    if winner is not None:
        return float("inf") if winner == maximizing_player else float("-inf")
    dist_self = _path_len(state, maximizing_player)
    dist_opp = _path_len(state, maximizing_player.opponent())
    return float(dist_opp - dist_self)


def _alphabeta(
    state: RawState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    maximizing_player: Player,
) -> float:
    if depth == 0 or _ENGINE.is_game_over(state):
        return _evaluate(state, maximizing_player)

    actions = _ENGINE.legal_actions(state)

    if maximizing:
        value = float("-inf")
        for action in actions:
            try:
                next_state = _ENGINE.apply_action(state, action)
            except ValueError:
                continue
            score = _alphabeta(next_state, depth - 1, alpha, beta,
                               False, maximizing_player)
            value = max(value, score)
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value
    else:
        value = float("inf")
        for action in actions:
            try:
                next_state = _ENGINE.apply_action(state, action)
            except ValueError:
                continue
            score = _alphabeta(next_state, depth - 1, alpha, beta,
                               True, maximizing_player)
            value = min(value, score)
            if value <= alpha:
                break
            beta = min(beta, value)
        return value
