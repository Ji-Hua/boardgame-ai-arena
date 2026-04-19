"""Policy abstraction for action selection in heuristic agents.

Provides a reusable policy layer that sits between evaluation (scoring)
and final action selection.  All randomness flows through an injected
``random.Random`` instance — no global state.

Mirrors the arena policy interface (``arena/agents/core.py``) so that
both systems share identical selection semantics.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Selects a single action from scored candidates."""

    @abstractmethod
    def select(
        self, scored_actions: list[tuple[dict, float]], rng: random.Random
    ) -> dict:
        """Choose one action from *scored_actions*.

        Args:
            scored_actions: [(action, score)] — non-empty.
            rng: Instance-owned RNG for stochastic policies.

        Returns:
            A single action dict.
        """
        ...


class TopKPolicy(Policy):
    """Sort by score descending, take top-k, uniformly select one.

    Deterministic argmax when k=1 (stable sort preserves order for ties).
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k

    @property
    def k(self) -> int:
        return self._k

    def select(
        self, scored_actions: list[tuple[dict, float]], rng: random.Random
    ) -> dict:
        # Stable sort descending by score; original order preserved for ties.
        sorted_actions = sorted(scored_actions, key=lambda x: x[1], reverse=True)
        top = sorted_actions[: self._k]
        chosen, _ = rng.choice(top)
        return chosen


def build_policy(config: dict[str, Any] | None) -> Policy | None:
    """Build a Policy from a config dict, or return None.

    Args:
        config: Policy configuration, e.g. ``{"type": "top_k", "k": 3}``.
                If None, returns None (caller should use default behavior).

    Returns:
        A Policy instance, or None.
    """
    if config is None:
        return None
    policy_type = config.get("type", "top_k")
    if policy_type == "top_k":
        return TopKPolicy(k=config.get("k", 1))
    raise ValueError(
        f"Unknown policy type '{policy_type}'. Available: ['top_k']"
    )
