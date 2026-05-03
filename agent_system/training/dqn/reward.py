"""Reward shaping for DQN Quoridor training — Phase 15A.

Supports two reward modes:

  terminal (default):
    sparse +1/-1 on win/loss only; non-terminal steps receive 0.
    Preserves all prior DQN baseline behaviour exactly.

  distance_delta:
    terminal reward + dense shaping based on the change in
    shortest-path-distance advantage between the learner's
    decision state (prev_state) and the resolved next state
    (after opponent response or terminal).

    advantage(state) = opponent_shortest_distance - learner_shortest_distance
    distance_delta   = next_advantage - prev_advantage
    clipped_delta    = clip(distance_delta, -clip, +clip)
    distance_reward  = weight * clipped_delta
    combined_reward  = terminal_reward + distance_reward

Design rationale:
  * Delta-based, not current-advantage-based: rewarding the current
    advantage every step encourages the agent to drag out winning
    positions.  Rewarding the *change* in advantage aligns with
    "make progress each turn".
  * Terminal reward dominates: with weight=0.01 and clip=2.0 the
    maximum per-step shaping is ±0.02, much smaller than ±1.0.
  * No Python-side BFS: shortest distances are computed exclusively
    via quoridor_engine.calculation.shortest_path_len, which calls
    the authoritative Rust BFS implementation.
  * No time penalty: deferred until the model already learns basic
    play.

Shortest-distance API used:
  quoridor_engine.calculation.shortest_path_len(state, player, topology)
  – requires (RawState, Player, Topology) – same signature as minimax agent.
  – Returns Optional[u32] (None if no path exists).
  – If None, a large sentinel (_INF) is substituted so arithmetic
    remains stable; distance-reward will be large but clipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

# Sentinel distance used when the engine reports no path exists.
# In legal Quoridor the rule engine blocks wall placements that would
# create a disconnected graph, so None should never occur in valid play.
# The sentinel is provided as a safe fallback.
_INF = 999_999.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Configuration for the reward function.

    Attributes
    ----------
    mode : "terminal" | "distance_delta"
        Reward mode.  "terminal" preserves legacy sparse behaviour.
    distance_reward_weight : float
        Scale factor applied to clipped_delta.  Default 0.01.
    distance_delta_clip : float
        Symmetric clip bound for distance_delta.  Default 2.0.
        Must be positive.
    """
    mode: Literal["terminal", "distance_delta"] = "terminal"
    distance_reward_weight: float = 0.01
    distance_delta_clip: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in ("terminal", "distance_delta"):
            raise ValueError(f"Unknown reward mode: {self.mode!r}")
        if not math.isfinite(self.distance_reward_weight) or self.distance_reward_weight < 0:
            raise ValueError(
                f"distance_reward_weight must be finite and non-negative, "
                f"got {self.distance_reward_weight}"
            )
        if not math.isfinite(self.distance_delta_clip) or self.distance_delta_clip <= 0:
            raise ValueError(
                f"distance_delta_clip must be finite and positive, "
                f"got {self.distance_delta_clip}"
            )


# ---------------------------------------------------------------------------
# Breakdown output
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Full reward breakdown for a single learner transition.

    Attributes
    ----------
    terminal_reward : float
        +1.0 (learner win), -1.0 (opponent win), or 0.0 (non-terminal).
    distance_reward : float
        Dense shaping reward.  Zero when mode="terminal".
    combined_reward : float
        terminal_reward + distance_reward.  This is the value stored in
        the replay buffer.
    prev_advantage : float | None
        opponent_dist - learner_dist in prev_state.  None in terminal mode.
    next_advantage : float | None
        opponent_dist - learner_dist in next_state.  None in terminal mode.
    distance_delta : float | None
        next_advantage - prev_advantage (unclipped).  None in terminal mode.
    clipped_delta : float | None
        distance_delta clipped to [-clip, +clip].  None in terminal mode.
    """
    terminal_reward: float
    distance_reward: float
    combined_reward: float
    prev_advantage: float | None = None
    next_advantage: float | None = None
    distance_delta: float | None = None
    clipped_delta: float | None = None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_terminal_reward(learner_player: Any, winner: Any, done: bool) -> float:
    """Return terminal reward from the learner's perspective.

    Parameters
    ----------
    learner_player : Player
        The engine Player object representing the learner.
    winner : Player | None
        The engine Player object that won, or None if game is not over.
    done : bool
        Whether the game is over.

    Returns
    -------
    float
        +1.0 if learner won, -1.0 if opponent won, 0.0 otherwise.
    """
    if not done or winner is None:
        return 0.0
    return 1.0 if winner == learner_player else -1.0


def compute_distance_advantage(engine: Any, state: Any, learner_player: Any) -> float:
    """Compute learner distance advantage in the given state.

    advantage = opponent_shortest_distance - learner_shortest_distance

    A larger (more positive) value is better for the learner.

    Uses quoridor_engine.calculation.shortest_path_len with the engine
    topology — the same Rust BFS call used by the minimax agent.

    Parameters
    ----------
    engine : RuleEngine
        The rule engine instance (provides topology).
    state : RawState
        The game state to evaluate.
    learner_player : Player
        The engine Player representing the learner.

    Returns
    -------
    float
        opponent_distance - learner_distance.
        If either path is unreachable (should not occur in valid play),
        the sentinel value _INF is used for that player's distance.
    """
    from quoridor_engine import calculation  # lazy import; avoids hard dependency at module load

    topo = engine.topology
    opponent = learner_player.opponent()

    learner_raw = calculation.shortest_path_len(state, learner_player, topo)
    opponent_raw = calculation.shortest_path_len(state, opponent, topo)

    learner_dist = float(learner_raw) if learner_raw is not None else _INF
    opponent_dist = float(opponent_raw) if opponent_raw is not None else _INF

    return opponent_dist - learner_dist


def compute_reward_breakdown(
    engine: Any,
    prev_state: Any,
    next_state: Any,
    learner_player: Any,
    terminal_reward: float,
    config: RewardConfig,
) -> RewardBreakdown:
    """Compute the full reward breakdown for one learner transition.

    Parameters
    ----------
    engine : RuleEngine
        The rule engine (provides topology for distance computation).
    prev_state : RawState
        State just before the learner acted (the learner's decision state).
    next_state : RawState
        State after the opponent responded (or terminal state if game ended).
    learner_player : Player
        The engine Player representing the learner.
    terminal_reward : float
        Pre-computed terminal reward (use compute_terminal_reward).
    config : RewardConfig
        Reward configuration.

    Returns
    -------
    RewardBreakdown
        Contains terminal, distance, and combined reward plus diagnostics.

    Notes
    -----
    In terminal mode, distance_reward is always 0.0 and diagnostic fields
    are None.

    In distance_delta mode:
      - prev_advantage is computed from prev_state
      - next_advantage is computed from next_state (even for terminal states,
        which remain valid for shortest-path queries)
      - distance_delta = next_advantage - prev_advantage
      - clipped_delta = clip(distance_delta, -clip, +clip)
      - distance_reward = weight * clipped_delta
      - combined_reward = terminal_reward + distance_reward
    """
    if config.mode == "terminal":
        return RewardBreakdown(
            terminal_reward=terminal_reward,
            distance_reward=0.0,
            combined_reward=terminal_reward,
        )

    # distance_delta mode
    prev_advantage = compute_distance_advantage(engine, prev_state, learner_player)
    next_advantage = compute_distance_advantage(engine, next_state, learner_player)
    distance_delta = next_advantage - prev_advantage

    clip = config.distance_delta_clip
    clipped_delta = max(-clip, min(clip, distance_delta))
    distance_reward = config.distance_reward_weight * clipped_delta
    combined_reward = terminal_reward + distance_reward

    return RewardBreakdown(
        terminal_reward=terminal_reward,
        distance_reward=distance_reward,
        combined_reward=combined_reward,
        prev_advantage=prev_advantage,
        next_advantage=next_advantage,
        distance_delta=distance_delta,
        clipped_delta=clipped_delta,
    )
