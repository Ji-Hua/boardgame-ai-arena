#!/usr/bin/env python3
"""CLI runner for the Quoridor Arena evaluation system.

Runs a round-robin tournament between YAML-defined agents and stores
results in SQLite.

Usage:
    python scripts/run_arena.py [--num-games N] [--seed S] [--db PATH]
    python scripts/run_arena.py --agent-dir agent_system/definition/agent_defs
    python scripts/run_arena.py --verbose        # per-game progress
    python scripts/run_arena.py --very-verbose    # per-step progress
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

# Ensure project root and backend-server are on the path.
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend-server"))

from agent_system.evaluation.arena.agents.loader import load_agents_from_dir
from agent_system.evaluation.arena.agents.core import AgentInstance

from agent_system.evaluation.arena.runner import ArenaRunner, play_single_game, VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE
from agent_system.evaluation.arena.db import init_db, insert_game, fetch_all_games
from agent_system.evaluation.arena.aggregator import compute_win_rate_matrix, format_matrix_text, format_pairwise_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Quoridor Arena — agent evaluation")
    parser.add_argument("--num-games", type=int, default=50, help="Games per matchup (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--db", type=str, default="arena_results.db", help="SQLite database path")
    parser.add_argument("--agent-dir", type=str, default=str(_root / "agent_system" / "definition" / "agent_defs"),
                        help="Directory containing YAML agent definitions")
    parser.add_argument("--verbose", action="store_true", help="Show per-game progress")
    parser.add_argument("--very-verbose", action="store_true", help="Show per-step progress")
    args = parser.parse_args()

    if args.very_verbose:
        verbosity = VERBOSITY_VERBOSE
    elif args.verbose:
        verbosity = VERBOSITY_NORMAL
    else:
        verbosity = VERBOSITY_QUIET

    # Load agents from YAML definitions
    agents = load_agents_from_dir(args.agent_dir)
    if not agents:
        print(f"No YAML agent definitions found in {args.agent_dir}", file=sys.stderr)
        sys.exit(1)

    agent_names = [a.id for a in agents]

    print(f"Quoridor Arena — Round-Robin Tournament")
    print(f"Agents: {', '.join(agent_names)}")
    print(f"Games per matchup: {args.num_games}")
    print(f"Base seed: {args.seed}")
    print(f"Database: {args.db}")
    print("=" * 60)

    # Open database
    db_path = Path(args.db)
    conn = init_db(db_path)

    # Run tournament — execute each game once, store records, aggregate
    from agent_system.evaluation.arena.models import MatchResult

    pair_index = 0
    num_games = args.num_games
    all_match_results: list[MatchResult] = []

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            print(f"\n{a.id} vs {b.id} ({num_games} games)...")

            pair_seed = args.seed + pair_index * num_games
            match = MatchResult(agent_a=a.id, agent_b=b.id)
            pair_start = time.monotonic()

            for g in range(num_games):
                seed = pair_seed + g
                if g % 2 == 0:
                    p1_def, p2_def = a, b
                else:
                    p1_def, p2_def = b, a

                p1 = AgentInstance(p1_def, seed=seed * 2)
                p2 = AgentInstance(p2_def, seed=seed * 2 + 1)

                game_start = time.monotonic()
                record = play_single_game(p1, p2, seed, verbosity=verbosity)
                game_elapsed = time.monotonic() - game_start
                insert_game(conn, record)

                if record.winner == a.id:
                    match.wins_a += 1
                elif record.winner == b.id:
                    match.wins_b += 1
                else:
                    match.draws += 1

                if verbosity >= VERBOSITY_NORMAL:
                    w = record.winner or "draw"
                    print(f"  game {g+1:3d}/{num_games} | seed={seed} | "
                          f"{record.num_steps:3d} steps | winner={w} | "
                          f"{game_elapsed:.1f}s")

            pair_elapsed = time.monotonic() - pair_start
            print(f"  {a.id}: {match.wins_a} wins")
            print(f"  {b.id}: {match.wins_b} wins")
            print(f"  Draws: {match.draws}")
            print(f"  ({pair_elapsed:.1f}s total)")

            all_match_results.append(match)
            pair_index += 1

    # Aggregate and display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_games = fetch_all_games(conn)
    matrix = compute_win_rate_matrix(all_games)

    print("\nWin-Rate Matrix:")
    print(format_matrix_text(matrix, agent_names))
    print("\nPairwise Results:")
    print(format_pairwise_text(matrix))

    conn.close()
    print(f"\nResults persisted to {args.db}")


if __name__ == "__main__":
    main()
