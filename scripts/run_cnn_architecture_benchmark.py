"""Train the predeclared CNN architecture matrix on a shared D2 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend-server")]

from agent_system.training.dqn.distillation import evaluate_heldout_policy


ARCHITECTURES = {
    "A-baseline": {"cnn_variant": "baseline", "cnn_channels": [32, 64, 64], "cnn_dense_width": 256},
    "B-deep": {"cnn_variant": "deep", "cnn_channels": [64, 64, 64], "cnn_dense_width": 230},
    "C-resnet": {"cnn_variant": "resnet", "cnn_channels": [64, 64, 64], "cnn_dense_width": 230},
    "D-wide": {"cnn_variant": "baseline", "cnn_channels": [56, 112, 112], "cnn_dense_width": 512},
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--book-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--updates", type=int, default=12_000)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Benchmark artifacts must be outside the source repository")
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if args.updates <= 0 or not args.seeds:
        raise ValueError("updates and seeds must be positive")
    for path in (args.dataset, args.holdout, args.book_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    output.mkdir(parents=True)
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": {"path": str(args.dataset.resolve()), "sha256": sha256(args.dataset)},
        "holdout": {"path": str(args.holdout.resolve()), "sha256": sha256(args.holdout)},
        "book_manifest": {"path": str(args.book_manifest.resolve()), "sha256": sha256(args.book_manifest)},
        "architectures": ARCHITECTURES,
        "seeds": args.seeds,
        "updates": args.updates,
        "protocol": "cnn-architecture-distillation-v1",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for architecture, shape in ARCHITECTURES.items():
        for seed in args.seeds:
            run_output = output / architecture / f"seed-{seed}"
            config = {
                "output": str(run_output),
                "book_manifest": str(args.book_manifest.resolve()),
                "hypothesis": "Architecture-only D2 policy-distillation comparison.",
                "training": {
                    "episodes": 2000, "batch_size": 256, "lr": 0.0002,
                    "checkpoint_dir": str(run_output / "checkpoints"),
                    "checkpoint_interval": 500, "agent_id": f"cnn_arch_{architecture}_s{seed}",
                    "seed": seed, "grad_clip_norm": 1.0, "device": "cuda",
                    "model_arch": "cnn", "cnn_activation": "leaky_relu",
                    "cnn_hierarchical": False, "warm_start_checkpoint": None,
                    "algorithm": "policy_distillation", **shape,
                },
                "distillation": {
                    "depth": 2, "dataset": str(args.dataset.resolve()), "updates": args.updates,
                    "checkpoint_updates": [args.updates], "batch_seed": 874,
                    "tactical_fraction": 0.25,
                },
            }
            config_path = output / f"{architecture}-seed-{seed}.json"
            config_path.write_text(json.dumps(config, indent=2))
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/train_cnn_repair.py"), "--config", str(config_path)],
                cwd=ROOT,
                check=True,
            )
            checkpoint = run_output / "checkpoints" / f"update{args.updates:06d}.pt"
            heldout = evaluate_heldout_policy(checkpoint, args.holdout, run_output / "heldout-d2.json")
            results.append({"architecture": architecture, "seed": seed, "checkpoint": str(checkpoint), "heldout": heldout})
            (output / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
