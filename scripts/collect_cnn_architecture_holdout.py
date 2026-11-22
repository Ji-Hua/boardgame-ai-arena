"""Create an isolated source-game-held-out D2 policy validation dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend-server")]

from agent_system.training.dqn.distillation import collect_heldout_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-state-keys", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026102102)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Held-out data must be outside the source repository")
    config = {
        "seed": args.seed,
        "episodes": args.episodes,
        "workers": args.workers,
        "episode_start": 1_000_000,
        "source_game_family": f"architecture-heldout-{args.seed}",
        "max_plies": 3000,
        "random_prefix_max": 24,
        "exploration": 0.2,
        "random_pawn_probability": 0.7,
    }
    print(json.dumps(collect_heldout_dataset(config, args.output, args.training_state_keys), indent=2))


if __name__ == "__main__":
    main()
