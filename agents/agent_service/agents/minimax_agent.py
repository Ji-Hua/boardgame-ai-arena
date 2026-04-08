"""Minimax Agent — depth-limited minimax with alpha-beta pruning.

Selects actions by exploring the game tree up to a configurable depth,
assuming optimal play from both players.  Uses shortest-path heuristic
for evaluation and alpha-beta pruning to reduce the search space.

Stateless: each decision depends only on the game_state provided by the
Backend.  Input state is never mutated.

Greedy equivalence:
    MinimaxAgent(depth=1) is functionally equivalent to GreedyAgent.
    At depth 1 the agent evaluates each action by its immediate effect
    on shortest-path lengths, with no opponent modelling.  Deeper depths
    enable the agent to anticipate opponent replies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quoridor_engine import Player, RawState, RuleEngine
from quoridor_engine import calculation

from agents.agent_service.base_agent import BaseAgent

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
                               False, maximizing_player, context)
            value = max(value, score)
            if value >= beta:
                context.cutoffs += 1
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
                               True, maximizing_player, context)
            value = min(value, score)
            if value <= alpha:
                context.cutoffs += 1
                break
            beta = min(beta, value)
        return value


class MinimaxAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning.

    Configurable depth (default 2).  At depth=1 this agent is equivalent
    to the GreedyAgent: it evaluates each action by its immediate effect
    on shortest-path lengths without modelling the opponent's reply.

    Configuration via ``configure({"depth": N})`` or constructor kwarg.
    """

    type_id = "minimax"
    display_name = "Minimax Agent"
    category = "ai"

    def __init__(self, depth: int = 2) -> None:
        self._depth = depth

    def configure(self, config: dict[str, Any]) -> None:
        if "depth" in config:
            self._depth = int(config["depth"])

    def make_action(self, game_state: dict, legal_actions: list[dict]) -> dict:
        state = _build_state(game_state)
        seat = game_state["current_player"]
        maximizing_player = state.current_player

        context = SearchContext(depth_limit=self._depth)

        # Generate full action space via local engine (includes walls).
        all_engine_actions = _ENGINE.legal_actions(state)

        best_value = float("-inf")
        best_engine_action = None
        alpha = float("-inf")
        beta = float("inf")

        for engine_action in all_engine_actions:
            try:
                next_state = _ENGINE.apply_action(state, engine_action)
            except ValueError:
                continue

            context.nodes += 1

            # After our move, opponent plays — so next layer is minimizing.
            value = _alphabeta(
                next_state,
                self._depth - 1,
                alpha,
                beta,
                False,
                maximizing_player,
                context,
            )

            if value > best_value:
                best_value = value
                best_engine_action = engine_action

            alpha = max(alpha, best_value)

        if best_engine_action is not None:
            return _engine_action_to_wire(best_engine_action, seat)
        # Fallback (should not happen in a valid game state).
        return legal_actions[0]


# ---------------------------------------------------------------------------
# Pre-configured difficulty presets
# ---------------------------------------------------------------------------

class MinimaxAgentSimple(MinimaxAgent):
    """Minimax Agent — Simple (depth 2)."""
    type_id = "minimax_simple"
    display_name = "Minimax Agent (Simple)"

    def __init__(self) -> None:
        super().__init__(depth=2)


class MinimaxAgentMedium(MinimaxAgent):
    """Minimax Agent — Medium (depth 3)."""
    type_id = "minimax_medium"
    display_name = "Minimax Agent (Medium)"

    def __init__(self) -> None:
        super().__init__(depth=3)


class MinimaxAgentHard(MinimaxAgent):
    """Minimax Agent — Hard (depth 5)."""
    type_id = "minimax_hard"
    display_name = "Minimax Agent (Hard)"

    def __init__(self) -> None:
        super().__init__(depth=5)
