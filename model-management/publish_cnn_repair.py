"""Pin every diagnosed checkpoint in a completed CNN experiment suite.

Creates a new catalog; never overwrites the historical catalog or online roster.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-run", action="append", default=[],
                        help="Run directory names to retain as smoke/audit evidence but not publish")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new catalog path")
    catalog = json.loads(args.catalog.read_text())
    existing_ids = {m["id"] for m in catalog["models"]}
    excluded = set(args.exclude_run)
    count = 0
    for run in sorted(args.suite.iterdir()):
        if run.name in excluded:
            continue
        if not (run / "experiment.json").is_file():
            continue
        if not (run / "result.json").is_file():
            raise ValueError(f"Training has not completed: {run}")
        run_result = json.loads((run / "result.json").read_text())
        diagnoses = json.loads((run / "diagnoses.json").read_text())
        if {Path(d["checkpoint"]).resolve() for d in diagnoses} != set((run / "checkpoints").glob("*.pt")):
            raise ValueError(f"Every checkpoint needs a diagnosis: {run}")
        for diag in diagnoses:
            path, identifier = Path(diag["checkpoint"]), diag["model_id"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != diag["sha256"] or identifier in existing_ids:
                raise ValueError(f"Checkpoint identity mismatch: {identifier}")
            existing_ids.add(identifier)
            dest = args.output.parent / "models" / (digest + ".pt")
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Corrupt model store: {dest}")
            else:
                shutil.copyfile(path, dest)
                dest.chmod(0o444)
            raw = torch.load(dest, map_location="cpu", weights_only=True)
            metadata = {key: raw.get(key) for key in (
                "checkpoint_id", "agent_id", "episode_count", "training_step", "observation_version",
                "action_space_version", "action_count", "created_at", "model_config", "eval_summary")}
            catalog["models"].append({"id": identifier, "sha256": digest, "checkpoint": str(dest),
                "source_path": str(path), "selection_reason": "Complete CNN repair experiment history; includes failed models",
                "metadata": metadata, "training_experiment": str(run),
                "training_status": run_result.get("status", "complete"),
                "diagnostics_dir": str(run / "diagnostics" / identifier)})
            catalog["agents"].append({"id": identifier, "display_name": identifier, "deterministic": True,
                "category": "ai", "algo": {"type": "dqn", "params": {
                    "checkpoint": str(dest), "sha256": digest, "device": "cpu"}},
                "policy": {"type": "top_k", "k": 1}})
            if raw["model_config"].get("cnn_value_head"):
                search = raw["eval_summary"].get("search", {})
                params = {key: search[key] for key in ("simulations", "c_puct", "max_depth")}
                search_id = identifier + f"_mcts{params['simulations']}"
                if any(a["id"] == search_id for a in catalog["agents"]):
                    raise ValueError(f"Search agent identity already exists: {search_id}")
                catalog["agents"].append({"id": search_id, "display_name": search_id, "deterministic": True,
                    "category": "ai", "algo": {"type": "mcts", "params": {
                        "checkpoint": str(dest), "sha256": digest, "device": "cpu", **params}},
                    "policy": {"type": "top_k", "k": 1}})
            count += 1
    if not count:
        raise ValueError("No complete diagnosed CNN runs found")
    catalog["selection"] = "Historical baselines plus every diagnosed CNN repair checkpoint"
    args.output.write_text(json.dumps(catalog, indent=2))
    print(json.dumps({"catalog": str(args.output), "new_models": count}))


if __name__ == "__main__":
    main()
