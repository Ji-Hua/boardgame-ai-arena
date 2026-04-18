"""Experiment runner — execute experiment definitions via ArenaRunner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from arena.agents.core import Agent, AgentInstance
from arena.experiment_loader import Experiment, Match
from arena.models import GameRecord, MatchResult
from arena.runner import (
    ArenaRunner,
    play_single_game,
    VERBOSITY_QUIET,
    VERBOSITY_NORMAL,
    VERBOSITY_VERBOSE,
)

_DEFAULT_NUM_GAMES = 50


@dataclass
class ExperimentResult:
    """Collected results from running an experiment."""

    experiment_id: str
    match_results: list[tuple[str, str, MatchResult]] = field(default_factory=list)
    game_records: list[GameRecord] = field(default_factory=list)


def run_experiment(
    experiment: Experiment,
    agent_registry: dict[str, Agent],
    base_seed: int = 42,
    verbosity: int = VERBOSITY_QUIET,
) -> ExperimentResult:
    """Execute all matches defined in *experiment*.

    For each match, looks up agents by ID in *agent_registry*, then runs
    the match through the existing ArenaRunner game-execution path to
    preserve identical behavior (alternating sides, per-game seeds, etc.).

    Args:
        experiment: Loaded Experiment definition.
        agent_registry: Mapping from agent ID → Agent.
        base_seed: Starting seed for deterministic execution.
        verbosity: VERBOSITY_QUIET / NORMAL / VERBOSE.

    Returns:
        ExperimentResult containing per-match MatchResults and all
        individual GameRecords.

    Raises:
        KeyError: If a match references an unknown agent ID.
    """
    result = ExperimentResult(experiment_id=experiment.id)

    for match_idx, match in enumerate(experiment.matches):
        # --- resolve agents ---
        agent_a = _resolve_agent(match.agent_1_id, agent_registry)
        agent_b = _resolve_agent(match.agent_2_id, agent_registry)

        num_games: int = match.params.get("num_games", _DEFAULT_NUM_GAMES)
        pair_seed = base_seed + match_idx * num_games

        print(f"\n{agent_a.id} vs {agent_b.id} ({num_games} games)...")

        match_result = MatchResult(agent_a=agent_a.id, agent_b=agent_b.id)
        pair_start = time.monotonic()

        for g in range(num_games):
            seed = pair_seed + g
            if g % 2 == 0:
                p1_def, p2_def = agent_a, agent_b
            else:
                p1_def, p2_def = agent_b, agent_a

            p1 = AgentInstance(p1_def, seed=seed * 2)
            p2 = AgentInstance(p2_def, seed=seed * 2 + 1)

            game_start = time.monotonic()
            record = play_single_game(p1, p2, seed, verbosity=verbosity)
            game_elapsed = time.monotonic() - game_start

            result.game_records.append(record)

            if record.winner == agent_a.id:
                match_result.wins_a += 1
            elif record.winner == agent_b.id:
                match_result.wins_b += 1
            else:
                match_result.draws += 1

            if verbosity >= VERBOSITY_NORMAL:
                w = record.winner or "draw"
                print(
                    f"  game {g + 1:3d}/{num_games} | seed={seed} | "
                    f"{record.num_steps:3d} steps | winner={w} | "
                    f"{game_elapsed:.1f}s"
                )

        pair_elapsed = time.monotonic() - pair_start
        print(f"  {agent_a.id}: {match_result.wins_a} wins")
        print(f"  {agent_b.id}: {match_result.wins_b} wins")
        print(f"  Draws: {match_result.draws}")
        print(f"  ({pair_elapsed:.1f}s total)")

        result.match_results.append(
            (agent_a.id, agent_b.id, match_result)
        )

    return result


def _resolve_agent(
    agent_id: str, registry: dict[str, Agent]
) -> Agent:
    """Look up an agent by ID; raise a clear error if missing."""
    if agent_id not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Unknown agent ID '{agent_id}'. "
            f"Available agents: {available}"
        )
    return registry[agent_id]
