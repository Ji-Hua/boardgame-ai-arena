"""Arena runner — orchestrates matches between agents.

All agents are config-driven: loaded from YAML via the
``AlgoType → Scorer → Policy → Agent → AgentInstance`` pipeline.

``play_single_game`` accepts ``AgentInstance`` objects.
``ArenaRunner`` accepts ``Agent`` definitions and materialises
``AgentInstance`` objects with per-game seeds for deterministic,
reproducible evaluation.
"""

from __future__ import annotations

import random
import time
from typing import Any

from backend.adapters.engine_adapter import EngineAdapter
from agent_system.evaluation.arena.models import GameRecord, MatchResult, TournamentResult
from agent_system.evaluation.arena.agents.core import Agent, AgentInstance

# Verbosity levels
VERBOSITY_QUIET = 0
VERBOSITY_NORMAL = 1   # per-game summary line
VERBOSITY_VERBOSE = 2  # per-step progress


def play_single_game(
    agent_a: AgentInstance,
    agent_b: AgentInstance,
    seed: int,
    verbosity: int = VERBOSITY_QUIET,
) -> GameRecord:
    """Play one full game between two AgentInstances via EngineAdapter.

    The game loop:
      1. Initialize engine via EngineAdapter
      2. Query state and legal actions from the adapter
      3. Ask the current agent for an action
      4. Apply the action through the adapter
      5. Repeat until game over

    Returns a GameRecord with winner name and step count.
    """
    random.seed(seed)

    adapter = EngineAdapter()
    adapter.initialize()

    instances = {1: agent_a, 2: agent_b}
    num_steps = 0
    max_steps = 500  # safety limit

    while not adapter.is_game_over() and num_steps < max_steps:
        state = adapter.get_state()
        current_player = state["current_player"]
        legal_actions = adapter.legal_pawn_actions()
        instance = instances[current_player]

        step_start = time.monotonic()
        action = instance.act(state)
        step_elapsed = time.monotonic() - step_start

        if verbosity >= VERBOSITY_VERBOSE:
            print(f"    step {num_steps:3d} | P{current_player} ({instance.agent_id}) "
                  f"-> {action.get('type','?')} {action.get('target','')} "
                  f"({step_elapsed:.2f}s)")

        result = adapter.take_action(action)

        if not result.get("success"):
            # If an agent produces an invalid action, it loses.
            winner_seat = 1 if current_player == 2 else 2
            winner_instance = instances[winner_seat]
            if verbosity >= VERBOSITY_VERBOSE:
                print(f"    !! INVALID action by {instance.agent_id}: {result.get('reason','')}")
            return GameRecord(
                agent_a=agent_a.agent_id,
                agent_b=agent_b.agent_id,
                winner=winner_instance.agent_id,
                num_steps=num_steps,
                seed=seed,
            )

        num_steps += 1

    winner_seat = adapter.winner()
    if winner_seat is not None:
        winner_name = instances[winner_seat].agent_id
    else:
        winner_name = None  # draw / max steps reached

    return GameRecord(
        agent_a=agent_a.agent_id,
        agent_b=agent_b.agent_id,
        winner=winner_name,
        num_steps=num_steps,
        seed=seed,
    )


class ArenaRunner:
    """Orchestrates matches and tournaments between config-driven agents.

    Accepts ``Agent`` definitions (from YAML) and materialises
    ``AgentInstance`` objects with per-game seeds.
    """

    def run_match(
        self,
        agent_a: Agent,
        agent_b: Agent,
        num_games: int,
        base_seed: int = 0,
    ) -> MatchResult:
        """Run num_games between two agents, alternating who goes first.

        Even-indexed games: agent_a plays as P1, agent_b as P2.
        Odd-indexed games: agent_b plays as P1, agent_a as P2.
        """
        result = MatchResult(agent_a=agent_a.id, agent_b=agent_b.id)

        for i in range(num_games):
            seed = base_seed + i
            if i % 2 == 0:
                p1_def, p2_def = agent_a, agent_b
            else:
                p1_def, p2_def = agent_b, agent_a

            p1 = AgentInstance(p1_def, seed=seed * 2)
            p2 = AgentInstance(p2_def, seed=seed * 2 + 1)

            record = play_single_game(p1, p2, seed)

            if record.winner == agent_a.id:
                result.wins_a += 1
            elif record.winner == agent_b.id:
                result.wins_b += 1
            else:
                result.draws += 1

        return result

    def run_tournament(
        self,
        agents: list[Agent],
        num_games: int,
        base_seed: int = 0,
    ) -> TournamentResult:
        """Run a round-robin tournament: every pair plays num_games."""
        tournament = TournamentResult()
        pair_index = 0

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pair_seed = base_seed + pair_index * num_games
                match_result = self.run_match(
                    agents[i], agents[j], num_games, base_seed=pair_seed
                )
                key = (agents[i].id, agents[j].id)
                tournament.results[key] = match_result
                pair_index += 1

        return tournament
