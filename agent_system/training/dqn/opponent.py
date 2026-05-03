"""Pluggable training-opponent abstraction for DQN training.

Provides a minimal interface for selecting opponent actions during the DQN
training loop.  The opponent is always the non-learner player; its actions
are applied through the Engine and are never stored as learner transitions.

Supported opponent types
------------------------
``random_legal``
    Uniform-random selection from engine-legal action IDs.  Default.

``minimax``
    Minimax with alpha-beta pruning at a configurable depth.  Reuses the
    existing ``MinimaxAgent`` search logic from the runtime service layer.

Interface
---------
All opponents expose a single method::

    opponent.select_action_id(
        engine: RuleEngine,
        state: RawState,
        legal_ids: list[int],
        rng: random.Random,
    ) -> int

The returned ``int`` is a valid DQN action_id drawn from ``legal_ids``.
The Engine is passed so opponent implementations can call
``engine.legal_actions`` directly without duplicating the rule authority.

Design constraints
------------------
- Opponents must not mutate game state.
- Opponents must return a legal action_id (one present in ``legal_ids``).
- Opponents must use canonical Engine legality — no independent legality
  check outside the Engine.
- The DQN training loop remains the single caller of ``engine.apply_action``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Sequence


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrainingOpponent(ABC):
    """Abstract base class for DQN training opponents."""

    @abstractmethod
    def select_action_id(
        self,
        engine: object,
        state: object,
        legal_ids: list[int],
        rng: random.Random,
    ) -> int:
        """Return a legal DQN action_id for the current state.

        Args:
            engine: ``quoridor_engine.RuleEngine`` instance.
            state:  ``quoridor_engine.RawState`` for the current game state.
            legal_ids: List of legal DQN action IDs (non-empty).
            rng: Random number generator for stochastic opponents.

        Returns:
            A DQN action_id that is present in ``legal_ids``.
        """


# ---------------------------------------------------------------------------
# RandomLegalOpponent
# ---------------------------------------------------------------------------

class RandomLegalOpponent(TrainingOpponent):
    """Selects uniformly at random from engine-legal action IDs."""

    def select_action_id(
        self,
        engine: object,
        state: object,
        legal_ids: list[int],
        rng: random.Random,
    ) -> int:
        return rng.choice(legal_ids)


# ---------------------------------------------------------------------------
# DummyOpponent
# ---------------------------------------------------------------------------

class DummyOpponent(TrainingOpponent):
    """Always selects the first legal action ID from ``legal_ids``.

    Provides an extremely weak, deterministic opponent that nearly always
    loses.  Useful as a positive-reward anchor in mixed-opponent training:
    even a barely-trained agent can win against a dummy opponent, ensuring
    some +1 rewards early in training.

    Args:
        None

    Raises:
        ValueError: If ``legal_ids`` is empty (should never happen in a
                    non-terminal game state).
    """

    def select_action_id(
        self,
        engine: object,
        state: object,
        legal_ids: list[int],
        rng: random.Random,
    ) -> int:
        if not legal_ids:
            raise ValueError(
                "DummyOpponent.select_action_id called with empty legal_ids; "
                "this should only happen in a terminal state."
            )
        return legal_ids[0]


# ---------------------------------------------------------------------------
# PawnRandomOpponent
# ---------------------------------------------------------------------------

class PawnRandomOpponent(TrainingOpponent):
    """Selects uniformly at random from legal PAWN-move action IDs only.

    Unlike ``RandomLegalOpponent`` (which also picks walls) and
    ``DummyOpponent`` (which is biased toward low-ID corners and broken as
    P1), this opponent:

    * never places walls, so it provides a clean benchmark for raw pawn
      progress without wall-complexity noise.
    * is unbiased — moves are chosen uniformly across all legal pawn moves.
    * behaves symmetrically in both P1 and P2 seats.

    Falls back to full ``random_legal`` if the engine returns no legal pawn
    moves (should be impossible in a non-terminal game state).
    """

    # Pawn action IDs occupy [0, BOARD_SIZE*BOARD_SIZE) = [0, 81).
    _PAWN_ID_END: int = 81

    def select_action_id(
        self,
        engine: object,
        state: object,
        legal_ids: list[int],
        rng: random.Random,
    ) -> int:
        pawn_ids = [a for a in legal_ids if a < self._PAWN_ID_END]
        if pawn_ids:
            return rng.choice(pawn_ids)
        # Fallback: no legal pawn moves — should not happen in normal play.
        return rng.choice(legal_ids)


# ---------------------------------------------------------------------------
# MinimaxOpponent
# ---------------------------------------------------------------------------

class MinimaxOpponent(TrainingOpponent):
    """Minimax opponent with alpha-beta pruning at a fixed depth.

    Reuses the search functions from
    ``agent_system.runtime.service.agents.minimax_agent``.

    The minimax search always maximizes for the *current* player (the
    opponent in the training loop), which is correct since Quoridor is
    zero-sum and alternating.

    Args:
        depth: Search depth.  depth=1 is equivalent to greedy.  depth=2 is
               the Phase 14A default.  Larger depths increase latency
               significantly.
    """

    def __init__(self, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError(f"minimax depth must be >= 1, got {depth}")
        self._depth = depth

        # Import lazily so this module can be used without the runtime
        # service layer's full dependency chain.
        from agent_system.runtime.service.agents.minimax_agent import (
            _ENGINE as _mm_engine,
            _alphabeta,
            _evaluate,
            SearchContext,
        )
        self._alphabeta = _alphabeta
        self._evaluate = _evaluate
        self._SearchContext = SearchContext
        self._mm_engine = _mm_engine  # shared singleton RuleEngine used by minimax internals

    @property
    def depth(self) -> int:
        return self._depth

    def select_action_id(
        self,
        engine: object,
        state: object,
        legal_ids: list[int],
        rng: random.Random,
    ) -> int:
        from agent_system.training.dqn.action_space import encode_engine_action

        maximizing_player = state.current_player
        context = self._SearchContext(depth_limit=self._depth)

        all_engine_actions = engine.legal_actions(state)

        best_value = float("-inf")
        best_action = None
        alpha = float("-inf")
        beta = float("inf")

        for engine_action in all_engine_actions:
            try:
                next_state = engine.apply_action(state, engine_action)
            except ValueError:
                continue

            # After our move, opponent plays — so next layer is minimizing.
            value = self._alphabeta(
                next_state,
                self._depth - 1,
                alpha,
                beta,
                False,          # next layer: minimizing
                maximizing_player,
                context,
            )

            if value > best_value:
                best_value = value
                best_action = engine_action

            alpha = max(alpha, best_value)

        if best_action is None:
            # Fallback: should not happen in a legal non-terminal state.
            # Use random as a safe fallback rather than crashing training.
            return rng.choice(legal_ids)

        return encode_engine_action(best_action)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VALID_OPPONENT_TYPES = ("random_legal", "minimax", "dummy", "pawn_random")


def build_opponent(
    opponent_type: str,
    minimax_depth: int = 2,
) -> TrainingOpponent:
    """Construct and return a TrainingOpponent by name.

    Args:
        opponent_type: One of ``"random_legal"``, ``"minimax"``, ``"dummy"``,
                       or ``"pawn_random"``.
        minimax_depth: Depth for minimax opponents (ignored for others).

    Returns:
        A ``TrainingOpponent`` instance.

    Raises:
        ValueError: If ``opponent_type`` is not a known type.
        ValueError: If ``minimax_depth < 1`` for minimax.
    """
    if opponent_type == "random_legal":
        return RandomLegalOpponent()
    if opponent_type == "minimax":
        return MinimaxOpponent(depth=minimax_depth)
    if opponent_type == "dummy":
        return DummyOpponent()
    if opponent_type == "pawn_random":
        return PawnRandomOpponent()
    raise ValueError(
        f"Unknown opponent type: {opponent_type!r}. "
        f"Valid types: {_VALID_OPPONENT_TYPES}"
    )


# ---------------------------------------------------------------------------
# MixedOpponent — per-episode weighted sampler
# ---------------------------------------------------------------------------

class MixedOpponent:
    """Per-episode opponent sampler from a weighted mixture of TrainingOpponent instances.

    This is **not** a ``TrainingOpponent`` itself.  The training loop must call
    ``sample(rng)`` once at the start of each episode to obtain the
    episode-level ``TrainingOpponent`` and its label string.

    Weights need not sum to 1.0; they are normalized internally.

    Example::

        mixed = MixedOpponent([
            (0.70, RandomLegalOpponent(),   "random_legal"),
            (0.20, MinimaxOpponent(depth=1), "minimax(d=1)"),
            (0.10, MinimaxOpponent(depth=2), "minimax(d=2)"),
        ])
        ep_opponent, ep_label = mixed.sample(rng)
    """

    def __init__(
        self,
        entries: list[tuple[float, TrainingOpponent, str]],
    ) -> None:
        """
        Args:
            entries: List of ``(weight, opponent, label)`` tuples.

        Raises:
            ValueError: If entries is empty, any weight < 0, or total weight <= 0.
        """
        if not entries:
            raise ValueError("MixedOpponent requires at least one entry")
        for w, _, label in entries:
            if w < 0:
                raise ValueError(
                    f"Opponent weight must be >= 0, got {w!r} for label {label!r}"
                )
        total = sum(w for w, _, _ in entries)
        if total <= 0:
            raise ValueError(
                f"Total opponent weight must be > 0, got {total}"
            )
        self._entries: list[tuple[float, TrainingOpponent, str]] = [
            (w / total, op, lbl) for w, op, lbl in entries
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, rng: random.Random) -> tuple[TrainingOpponent, str]:
        """Sample one opponent for the current episode.

        Uses ``rng`` for reproducible sampling.

        Returns:
            ``(opponent, label)`` — the selected episode-level
            ``TrainingOpponent`` and its human-readable label string.
        """
        r = rng.random()
        cumulative = 0.0
        for weight, opp, label in self._entries:
            cumulative += weight
            if r < cumulative:
                return opp, label
        # Floating-point fallback: return last entry.
        _, opp, label = self._entries[-1]
        return opp, label

    @property
    def entries(self) -> list[tuple[float, TrainingOpponent, str]]:
        """Normalized ``(weight, opponent, label)`` entries (read-only copy)."""
        return list(self._entries)

    def description(self) -> str:
        """Human-readable description of the mix, e.g. ``mixed(random_legal:0.70, minimax(d=1):0.20)``."""
        parts = [f"{lbl}:{w:.2f}" for w, _, lbl in self._entries]
        return "mixed(" + ", ".join(parts) + ")"


def build_mixed_opponent(
    entries: list[tuple[float, str, int]],
) -> "MixedOpponent":
    """Build a :class:`MixedOpponent` from a list of ``(weight, opponent_type, depth)`` tuples.

    Args:
        entries: Each entry is ``(weight, opponent_type, minimax_depth)``.
                 ``opponent_type`` must be ``"random_legal"`` or ``"minimax"``.
                 ``minimax_depth`` is used only when ``opponent_type == "minimax"``;
                 it is ignored (but still validated for ``"minimax"``) otherwise.

    Returns:
        A :class:`MixedOpponent` ready for per-episode sampling.

    Raises:
        ValueError: If entries is empty, any weight < 0, total weight <= 0,
                    opponent type is unknown, or minimax depth < 1.
    """
    if not entries:
        raise ValueError("build_mixed_opponent requires at least one entry")
    built: list[tuple[float, TrainingOpponent, str]] = []
    for weight, opp_type, depth in entries:
        if weight < 0:
            raise ValueError(f"Weight must be >= 0, got {weight!r}")
        op = build_opponent(opp_type, minimax_depth=depth)
        if opp_type == "minimax":
            label = f"minimax(d={depth})"
        else:
            label = opp_type
        built.append((weight, op, label))
    return MixedOpponent(built)
