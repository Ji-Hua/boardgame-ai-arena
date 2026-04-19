"""Arena data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchResult:
    """Result of a series of games between two agents."""
    agent_a: str
    agent_b: str
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0


@dataclass
class GameRecord:
    """Result of a single game."""
    agent_a: str
    agent_b: str
    winner: str | None
    num_steps: int
    seed: int


@dataclass
class TournamentResult:
    """Result of a round-robin tournament."""
    results: dict[tuple[str, str], MatchResult] = field(default_factory=dict)
