#!/usr/bin/env python3
"""CLI runner for YAML-defined Arena experiments.

Loads agent definitions and an experiment YAML, then executes the
specified matchups via ArenaRunner.

Usage:
    python scripts/run_experiment.py agent_system/evaluation/arena/experiments/basic.yaml
    python scripts/run_experiment.py agent_system/evaluation/arena/experiments/basic.yaml --verbose
    python scripts/run_experiment.py agent_system/evaluation/arena/experiments/basic.yaml --very-verbose
    python scripts/run_experiment.py agent_system/evaluation/arena/experiments/basic.yaml --seed 0 --db results.db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root and backend-server are on the path.
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend-server"))

from agent_system.evaluation.arena.agents.loader import load_agents_from_dir
from agent_system.evaluation.arena.experiment_loader import load_experiment
from agent_system.evaluation.arena.experiment_runner import run_experiment
from agent_system.evaluation.arena.runner import VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE
from agent_system.evaluation.arena.db import init_db, insert_game, fetch_all_games
from agent_system.evaluation.arena.aggregator import compute_win_rate_matrix, format_matrix_text, format_pairwise_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quoridor Arena — run a YAML-defined experiment"
    )
    parser.add_argument(
        "experiment", type=str, help="Path to experiment YAML file"
    )
    parser.add_argument(
        "--agent-dir",
        type=str,
        default=str(_root / "agent_system" / "definition" / "agent_defs"),
        help="Directory containing YAML agent definitions",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="arena_results.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-game progress"
    )
    parser.add_argument(
        "--very-verbose", action="store_true", help="Show per-step progress"
    )
    args = parser.parse_args()

    if args.very_verbose:
        verbosity = VERBOSITY_VERBOSE
    elif args.verbose:
        verbosity = VERBOSITY_NORMAL
    else:
        verbosity = VERBOSITY_QUIET

    # --- Load agents ---
    agents = load_agents_from_dir(args.agent_dir)
    if not agents:
        print(
            f"No YAML agent definitions found in {args.agent_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    agent_registry = {a.id: a for a in agents}

    # --- Load experiment ---
    try:
        experiment = load_experiment(args.experiment)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading experiment: {exc}", file=sys.stderr)
        sys.exit(1)

    # Collect unique agent IDs referenced by this experiment
    referenced_ids: set[str] = set()
    for m in experiment.matches:
        referenced_ids.add(m.agent_1_id)
        referenced_ids.add(m.agent_2_id)
    agent_names = sorted(referenced_ids)

    print(f"Quoridor Arena — Experiment: {experiment.id}")
    print(f"Agents: {', '.join(agent_names)}")
    print(f"Matches: {len(experiment.matches)}")
    print(f"Base seed: {args.seed}")
    print(f"Database: {args.db}")
    print("=" * 60)

    # --- Open database ---
    db_path = Path(args.db)
    conn = init_db(db_path)

    # --- Run experiment ---
    try:
        exp_result = run_experiment(
            experiment,
            agent_registry,
            base_seed=args.seed,
            verbosity=verbosity,
        )
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    # --- Persist game records ---
    for record in exp_result.game_records:
        insert_game(conn, record)

    # --- Aggregate and display results ---
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
