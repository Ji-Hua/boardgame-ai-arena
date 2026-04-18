"""Core abstractions: Scorer, Policy, Agent, AgentInstance."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class Scorer(ABC):
    """Evaluates all legal actions and assigns comparable scores.

    Must enumerate ALL legal actions.  Must NOT select the final action.
    """

    @abstractmethod
    def score(self, state: dict) -> list[tuple[dict, float]]:
        """Return [(action, score)] for every legal action.

        Args:
            state: GameState dict (backend wire format).

        Returns:
            List of (action_dict, score) pairs covering the full legal
            action space.
        """
        ...


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

    argmax behaviour when k=1.
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k

    def select(
        self, scored_actions: list[tuple[dict, float]], rng: random.Random
    ) -> dict:
        # Sort descending by score; stable sort preserves original order for ties.
        sorted_actions = sorted(scored_actions, key=lambda x: x[1], reverse=True)
        top = sorted_actions[: self._k]
        chosen, _ = rng.choice(top)
        return chosen


@dataclass
class Agent:
    """Complete agent definition: Scorer + Policy.

    Canonical identity is derived deterministically from
    algo_type + algo_params + policy descriptor, independent of the
    human-readable ``id``.
    """

    id: str
    scorer: Scorer
    policy: Policy
    algo_type: str = ""
    algo_params: dict[str, Any] = field(default_factory=dict)
    policy_type: str = ""
    policy_params: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        """Deterministic identity string derived from config.

        Format: ``algo_type[/param=val,...][+policy_type[/param=val,...]]``

        Examples:
            ``random+top_k``
            ``minimax/depth=3+top_k/k=1``
            ``greedy+top_k``
        """
        parts: list[str] = [self.algo_type] if self.algo_type else [type(self.scorer).__name__]
        if self.algo_params:
            param_str = ",".join(
                f"{k}={v}" for k, v in sorted(self.algo_params.items())
            )
            parts.append(f"/{param_str}")
        parts.append("+")
        parts.append(self.policy_type if self.policy_type else type(self.policy).__name__)
        if self.policy_params:
            pp_str = ",".join(
                f"{k}={v}" for k, v in sorted(self.policy_params.items())
            )
            parts.append(f"/{pp_str}")
        return "".join(parts)

    def act(self, state: dict, rng: random.Random) -> dict:
        scored = self.scorer.score(state)
        return self.policy.select(scored, rng)


class AgentInstance:
    """Runtime realisation of an Agent for a single game.

    Owns its own RNG — no global random usage.
    Deterministic given the same seed.
    """

    def __init__(self, agent: Agent, seed: int) -> None:
        self._agent = agent
        self._rng = random.Random(seed)

    @property
    def agent_id(self) -> str:
        return self._agent.id

    def act(self, state: dict) -> dict:
        return self._agent.act(state, self._rng)

    def reset(self, seed: int | None = None) -> None:
        """Reset RNG for a new game.  If seed is None, keep current state."""
        if seed is not None:
            self._rng = random.Random(seed)
