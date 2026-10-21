"""Lock a CNN candidate using development diagnostics, then evaluate fresh openings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend-server")]
from agent_system.evaluation.arena.matrix import make_openings, stable_hash, standard_opening
from agent_system.evaluation.cnn_diagnostics import diagnose_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026102101)
    parser.add_argument("--candidate-prefix", default=None)
    parser.add_argument("--baseline-prefix", default=None)
    parser.add_argument("--fixed-selection", action="store_true",
                        help="Each prefix must identify exactly one prespecified checkpoint; no development ranking")
    parser.add_argument("--replicate-prefix", action="append", default=[],
                        help="Also lock/evaluate one development-selected model for each training seed")
    parser.add_argument("--comparison-prefix", action="append", default=[],
                        help="Also lock/evaluate a distinct training method or control on the same book")
    parser.add_argument("--reference-diagnosis", type=Path, action="append", default=[],
                        help="Evaluate an already frozen reference on the same fresh opening book")
    parser.add_argument("--exclude-run", action="append", default=[],
                        help="Ignore incomplete or non-protocol subdirectories in the suite")
    parser.add_argument("--exclude-book", type=Path, action="append", default=[])
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[],
                        help="Exclude opening geometries and reflections found in demonstration datasets")
    parser.add_argument("--exclude-state-keys", type=Path, action="append", default=[],
                        help="Exclude all symmetry hashes visited during episode-based training")
    args = parser.parse_args()
    if args.pairs < 1:
        raise ValueError("pairs must be positive")
    candidates = []
    for path in sorted(args.suite.glob("*/diagnoses.json")):
        if path.parent.name in set(args.exclude_run):
            continue
        if not (path.parent / "result.json").exists():
            raise ValueError(f"Training incomplete: {path.parent}")
        candidates.extend(json.loads(path.read_text()))
    if not candidates:
        raise ValueError("No diagnosed candidates")
    # Fixed tie breaks: winning score, tactical completion, then smaller Q
    # magnitude. This choice is persisted before seeing any holdout results.
    def select(prefix):
        rows = [d for d in candidates if not prefix or d["model_id"].startswith(prefix)]
        if not rows:
            raise ValueError(f"No diagnosed models with prefix {prefix}")
        if args.fixed_selection and len(rows) != 1:
            raise ValueError(f"Fixed selection must name exactly one checkpoint: {prefix}")
        return max(rows, key=lambda d: (d["score"], d["tactical_wins"], -d["max_abs_legal_q"]))
    candidate = select(args.candidate_prefix)
    baseline = select(args.baseline_prefix) if args.baseline_prefix else json.loads((args.suite / "baseline/diagnosis.json").read_text())
    replicates = [select(prefix) for prefix in args.replicate_prefix]
    comparisons = [select(prefix) for prefix in args.comparison_prefix]
    references = [json.loads(path.read_text()) for path in args.reference_diagnosis]
    if candidate["sha256"] == baseline["sha256"]:
        raise ValueError("Candidate and baseline must differ")
    output = args.suite / "holdout"
    output.mkdir(exist_ok=False)
    development = json.loads(next(args.suite.glob("*/book.json")).read_text())
    prior_openings = list(development)
    seen = {o["state_hash"] for o in development}
    for path in args.exclude_book:
        data = json.loads(path.read_text())
        previous = data if isinstance(data, list) else data["book"]
        prior_openings.extend(previous)
        seen.update(o["state_hash"] for o in previous)
    seen_geometry, excluded_datasets, excluded_state_keys = set(), [], []
    import numpy as np
    from agent_system.evaluation.arena.agents.scorers import _build_state
    from agent_system.training.dqn.observation_cnn import encode_observation_cnn
    from agent_system.training.dqn.demonstrations import symmetry_key
    def geometry_key(opening):
        return symmetry_key(np.asarray(encode_observation_cnn(_build_state(opening["state"])), dtype=np.float32))
    excluded_book_geometry = {geometry_key(o) for o in prior_openings}
    selected_geometry = set()
    if args.exclude_dataset:
        import torch
        for path in args.exclude_dataset:
            dataset = torch.load(path, map_location="cpu", weights_only=True)
            for split in ("train", "validation"):
                seen_geometry.update(symmetry_key(obs.numpy()) for obs in dataset[split]["obs"])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            excluded_datasets.append({"path": str(path), "sha256": digest})
    dataset_group_count = len(seen_geometry)
    for path in args.exclude_state_keys:
        keys = json.loads(path.read_text())
        if not isinstance(keys, list) or any(not isinstance(k, str) or len(k) != 64 for k in keys):
            raise ValueError(f"Invalid training symmetry keys: {path}")
        seen_geometry.update(keys)
        excluded_state_keys.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "groups": len(keys)})
    book, offset = [], 0
    while len(book) < args.pairs:
        for opening in make_openings(args.pairs*2, args.seed + offset):
            if opening["state_hash"] not in seen:
                geometry = geometry_key(opening)
                if geometry in seen_geometry or geometry in excluded_book_geometry or geometry in selected_geometry:
                    continue
                selected_geometry.add(geometry)
                seen.add(opening["state_hash"])
                opening["id"] = len(book)
                book.append(opening)
                if len(book) == args.pairs:
                    break
        offset += args.pairs*2
    selection = {"candidate_id": candidate["model_id"], "candidate_sha256": candidate["sha256"],
        "baseline_id": baseline["model_id"], "baseline_sha256": baseline["sha256"],
        "criterion": "prespecified unique checkpoint for each prefix" if args.fixed_selection else
                     "development score; tactical wins; smaller maximum legal Q magnitude",
        "seed": args.seed, "pairs": args.pairs, "book": book, "book_sha256": stable_hash(book),
        "excluded_books": [str(p) for p in args.exclude_book],
        "excluded_datasets": excluded_datasets,
        "excluded_dataset_symmetry_groups": dataset_group_count,
        "excluded_state_keys": excluded_state_keys,
        "excluded_symmetry_groups": len(seen_geometry),
        "excluded_book_symmetry_groups": len(excluded_book_geometry),
        "unique_opening_symmetry_groups": len(selected_geometry),
        "replicate_selections": [{"prefix": prefix, "model_id": model["model_id"], "sha256": model["sha256"]}
                                 for prefix, model in zip(args.replicate_prefix, replicates)],
        "method_selections": [{"prefix": prefix, "model_id": model["model_id"], "sha256": model["sha256"]}
                              for prefix, model in zip(args.comparison_prefix, comparisons)],
        "references": [{"model_id": model["model_id"], "sha256": model["sha256"]} for model in references],
        "opening_distribution": "4 random legal plies, pawn probability .7; conditioned on unique states outside the recorded book and dataset exclusions",
        "scope": "paired Greedy evaluation; opening uncertainty is separate from training-seed variability; not proof of strength versus other agents"}
    evaluation_source = Path(__file__).read_bytes()
    selection["evaluation_source_sha256"] = hashlib.sha256(evaluation_source).hexdigest()
    (output / "evaluation-source.py").write_bytes(evaluation_source)
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    (output / "excluded-training-state-keys.json").write_text(json.dumps(sorted(seen_geometry)))
    results = {}
    models = {}
    for model in (baseline, candidate, *replicates, *comparisons, *references):
        if model["model_id"] in models and models[model["model_id"]]["sha256"] != model["sha256"]:
            raise ValueError("Conflicting model identity")
        models[model["model_id"]] = model
    for model in models.values():
        if hashlib.sha256(Path(model["checkpoint"]).read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("Checkpoint changed after development evaluation")
        results[model["model_id"]] = diagnose_checkpoint(model["checkpoint"], model["model_id"], book,
                                                         output / model["model_id"])
        diagnose_checkpoint(model["checkpoint"], model["model_id"], [standard_opening()],
                            output / (model["model_id"] + "_standard"))
    delta = results[candidate["model_id"]]["score"] - results[baseline["model_id"]]["score"]
    # A paired difference lies in [-1,1], so Hoeffding's radius is twice the
    # radius for a score in [0,1]. Opening pairs, not games, are sampling units.
    radius = 2 * math.sqrt(math.log(40)/(2*args.pairs))
    comparison = {"selection": {k: v for k, v in selection.items() if k != "book"},
        "results": results, "score_improvement": delta,
        "paired_improvement_95_interval": [max(-1., delta-radius), min(1., delta+radius)]}
    if replicates:
        # Simultaneous bounds across the prespecified seed comparisons. Each
        # paired difference is in [-1,1]; Bonferroni does not require the models
        # to be independent when they play the same opening book.
        family_radius = 2 * math.sqrt(math.log(40*len(replicates))/(2*args.pairs))
        comparison["replicate_comparisons"] = [{
            "model_id": model["model_id"],
            "score_improvement": (improvement := results[model["model_id"]]["score"] - results[baseline["model_id"]]["score"]),
            "simultaneous_95_interval": [max(-1., improvement-family_radius), min(1., improvement+family_radius)]}
            for model in replicates]
        comparison["replicate_interval_method"] = "Bonferroni-Hoeffding; seat-swapped opening pair is the sampling unit"
    if comparisons:
        # Include the prespecified primary in the method-comparison family.
        family = list({m["model_id"]: m for m in (candidate, *comparisons)}.values())
        family_radius = 2 * math.sqrt(math.log(40 * len(family)) / (2 * args.pairs))
        comparison["method_comparisons"] = [{"model_id": model["model_id"],
            "score_improvement": (improvement := results[model["model_id"]]["score"] - results[baseline["model_id"]]["score"]),
            "simultaneous_95_interval": [max(-1., improvement-family_radius), min(1., improvement+family_radius)]}
            for model in family]
        comparison["method_interval_method"] = "Bonferroni-Hoeffding; paired opening is the sampling unit; all planned methods and controls versus the frozen parent"
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps({"candidate": candidate["model_id"], "score_improvement": delta,
                      "paired_improvement_95_interval": comparison["paired_improvement_95_interval"]}), flush=True)


if __name__ == "__main__":
    main()
