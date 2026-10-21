"""Minimax Agent — depth-limited minimax with alpha-beta pruning.

Selects actions by exploring the game tree up to a configurable depth,
assuming optimal play from both players.  Uses shortest-path heuristic
for evaluation and alpha-beta pruning to reduce the search space.

Supports an optional top-k policy for controlled randomness.  When no
policy is configured the agent is strictly deterministic (identical to
the original implementation).

Greedy relationship:
    At depth 1, non-terminal evaluations differ from Greedy by a constant
    baseline. Terminal handling differs: Minimax uses +/- infinity while
    Greedy retains its distance-change score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from quoridor_engine import Player, RawState, RuleEngine
from quoridor_engine import calculation

from agent_system.runtime.service.base_agent import BaseAgent
from agent_system.runtime.service.policy import Policy

_ENGINE = RuleEngine.standard()
_TOPO = _ENGINE.topology
_INF = 999_999


def _seat_to_player(seat: int) -> Player:
    return Player.P1 if seat == 1 else Player.P2


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


@dataclass
class SearchContext:
    """Tracks search statistics for a single decision."""
    depth_limit: int = 3
    nodes: int = 0
    cutoffs: int = 0
    observer: Callable[[str, dict], None] | None = None


def _evaluate(state: RawState, maximizing_player: Player) -> float:
    """Evaluate a state from the perspective of *maximizing_player*.

    Terminal states return +/- infinity.
    Non-terminal states use: dist(opponent_to_goal) - dist(self_to_goal).
    A higher value is better for maximizing_player.
    """
    opponent = maximizing_player.opponent()

    # Terminal check: if either player has won.
    winner = _ENGINE.winner(state)
    if winner is not None:
        if winner == maximizing_player:
            return float("inf")
        else:
            return float("-inf")

    dist_self = _path_len(state, maximizing_player)
    dist_opp = _path_len(state, opponent)
    return dist_opp - dist_self


def _alphabeta(
    state: RawState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    maximizing_player: Player,
    context: SearchContext,
) -> float:
    """Depth-limited minimax with alpha-beta pruning.

    Alpha-beta pruning (Wikipedia-style):
      - Max layer: prune when value >= beta  (beta cutoff)
      - Min layer: prune when value <= alpha (alpha cutoff)

    Args:
        state: Current game state (not mutated).
        depth: Remaining search depth.
        alpha: Best value the maximizer can guarantee.
        beta: Best value the minimizer can guarantee.
        maximizing: True if this is a maximizing layer.
        maximizing_player: The player we are maximizing for (root caller).
        context: SearchContext for metrics tracking.

    Returns:
        Evaluation score from the perspective of maximizing_player.
    """
    context.nodes += 1
    observer = context.observer
    if observer is not None:
        observer("enter", {
            "state": state, "depth": depth, "alpha": alpha, "beta": beta,
            "maximizing": maximizing,
        })

    if depth == 0 or _ENGINE.is_game_over(state):
        value = _evaluate(state, maximizing_player)
        if observer is not None:
            observer("exit", {"value": value, "leaf": True, "pruned": 0})
        return value

    actions = _ENGINE.legal_actions(state)
    pruned = 0

    if maximizing:
        value = float("-inf")
        for index, action in enumerate(actions):
            try:
                next_state = _ENGINE.apply_action(state, action)
            except ValueError:
                continue
            if observer is not None:
                observer("edge", {"action": action})
            score = _alphabeta(next_state, depth - 1, alpha, beta,
                               False, maximizing_player, context)
            value = max(value, score)
            if value >= beta:
                context.cutoffs += 1
                pruned = len(actions) - index - 1
                break
            alpha = max(alpha, value)
        if observer is not None:
            observer("exit", {"value": value, "leaf": False, "pruned": pruned})
        return value
    else:
        value = float("inf")
        for index, action in enumerate(actions):
            try:
                next_state = _ENGINE.apply_action(state, action)
            except ValueError:
                continue
            if observer is not None:
                observer("edge", {"action": action})
            score = _alphabeta(next_state, depth - 1, alpha, beta,
                               True, maximizing_player, context)
            value = min(value, score)
            if value <= alpha:
                context.cutoffs += 1
                pruned = len(actions) - index - 1
                break
            beta = min(beta, value)
        if observer is not None:
            observer("exit", {"value": value, "leaf": False, "pruned": pruned})
        return value


def score_bound(value: float, alpha: float, beta: float) -> str:
    """Conservative fail-soft bounds, relative to the original window."""
    if alpha != float("-inf") and value <= alpha:
        return "upper"
    if beta != float("inf") and value >= beta:
        return "lower"
    return "exact"


def score_position(state: RawState, depth: int = 2, observer=None):
    """The serving root search, with optional read-only diagnostic events."""
    if depth < 1:
        raise ValueError("depth must be positive")
    context = SearchContext(depth_limit=depth, observer=observer)
    results = []
    alpha, beta = float("-inf"), float("inf")
    for action in _ENGINE.legal_actions(state):
        try:
            after = _ENGINE.apply_action(state, action)
        except ValueError:
            continue
        before_nodes, before_cutoffs = context.nodes, context.cutoffs
        context.nodes += 1
        if observer is not None:
            observer("root_action", {"action": action})
        value = _alphabeta(after, depth - 1, alpha, beta, False,
                           state.current_player, context)
        bound = "exact" if depth == 1 or _ENGINE.is_game_over(after) else score_bound(value, alpha, beta)
        results.append({
            "action": action, "state": after, "score": value, "bound": bound,
            "nodes": context.nodes - before_nodes,
            "cutoffs": context.cutoffs - before_cutoffs,
        })
        alpha = max(alpha, value)
    return results, context


def inspect_branch(state: RawState, action, depth: int, observer=None):
    """Search one candidate with a full window, yielding its exact depth value."""
    if depth < 1:
        raise ValueError("depth must be positive")
    after = _ENGINE.apply_action(state, action)
    context = SearchContext(depth_limit=depth, observer=observer)
    if observer is not None:
        observer("root_action", {"action": action})
    value = _alphabeta(after, depth - 1, float("-inf"), float("inf"),
                       False, state.current_player, context)
    return value, context


class MinimaxAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning.

    Configurable depth (default 2). Depth 1 evaluates immediate positions
    without modelling an opponent reply, but retains terminal win/loss scores.

    Configuration via ``configure({"depth": N})`` or constructor kwarg.
    """

    type_id = "minimax"
    display_name = "Minimax Agent"
    category = "ai"

    def __init__(
        self,
        depth: int = 2,
        policy: Policy | None = None,
        seed: int | None = None,
    ) -> None:
        self._depth = depth
        self._policy = policy
        self._rng = random.Random(seed) if seed is not None else None

    def configure(self, config: dict[str, Any]) -> None:
        if "depth" in config:
            self._depth = int(config["depth"])
        from agent_system.runtime.service.policy import build_policy
        if "policy" in config:
            self._policy = build_policy(config["policy"])
        if "seed" in config:
            self._rng = random.Random(int(config["seed"]))

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        state = _build_state(game_state)
        seat = game_state["current_player"]
        scored_actions: list[tuple[dict, float]] = []
        best_value = float("-inf")
        best_wire_action: dict | None = None
        candidates, _ = score_position(state, self._depth)
        for candidate in candidates:
            value = candidate["score"]
            wire_action = _engine_action_to_wire(candidate["action"], seat)

            if self._policy is not None:
                scored_actions.append((wire_action, value))

            if value > best_value:
                best_value = value
                best_wire_action = wire_action

        # Policy path: delegate selection to policy with injected RNG.
        if self._policy is not None and self._rng is not None and scored_actions:
            return self._policy.select(scored_actions, self._rng)

        # Default path: deterministic argmax (original behavior).
        if best_wire_action is not None:
            return best_wire_action
        # Fallback (should not happen in a valid game state).
        return legal_actions[0]
