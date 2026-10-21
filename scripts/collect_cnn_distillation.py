"""Generate the shared, episode-budgeted Minimax policy dataset on training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend-server")]
from agent_system.training.dqn.distillation import collect_shared_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config.pop("output")).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Generated data must be outside the source repository")
    print(json.dumps(collect_shared_dataset(config, output), indent=2))
