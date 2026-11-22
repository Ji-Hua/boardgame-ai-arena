"""Read-only teacher fidelity and head-to-head audit after the frozen holdout.

Does not train, select checkpoints, or modify weights. Search values use exact
full-window branches when judging CNN actions; pruned root bounds are not Qs.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend-server")]

import numpy as np
import torch
from agent_system.analysis.state import ENGINE, Move, to_position, wire_action
from agent_system.evaluation.arena.agents.scorers import _build_state
from agent_system.evaluation.arena.matrix import play_block, summarize, stable_hash, _materialized
from agent_system.runtime.service.agents.greedy_agent import score_position as greedy_scores
from agent_system.runtime.service.agents.minimax_agent import inspect_branch
from agent_system.training.dqn.checkpoint import load_checkpoint
from agent_system.training.dqn.demonstrations import symmetry_key
from agent_system.training.dqn.model import policy_argmax
from agent_system.training.dqn.observation_cnn import encode_observation_cnn
from agent_system.training.dqn.teacher_training import teacher_candidates
from agent_system.training.dqn.action_space import encode_engine_action


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def _spec(identifier, kind, params):
    return {"id": identifier, "algo": {"type": kind, "params": params},
            "policy": {"type": "top_k", "k": 1}, "deterministic": True}


def _series(specs, pairs, book, output):
    blocks = []
    for left, right in pairs:
        current = [play_block(left, right, opening, max_plies=300, seed=2026102301) for opening in book]
        if any(game["error"] for block in current for game in block["games"]):
            raise RuntimeError("Invalid game in teacher comparison")
        blocks.extend(current)
        with gzip.open(output / f"{left['id']}__{right['id']}.json.gz", "wt") as stream:
            json.dump({"book": book, "blocks": current}, stream)
        print(json.dumps({"series": [left["id"], right["id"]], "games": len(current)*2}), flush=True)
    report = summarize(specs, blocks, len(pairs)*len(book))
    report["book_sha256"] = stable_hash(book)
    _materialized.cache_clear()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--fidelity-only", action="store_true",
                        help="Reuse an already run Arena matrix; only audit policy decisions")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = json.loads((args.suite / "plan.json").read_text())
    selection = json.loads((args.suite / "holdout/selection.json").read_text())
    results = json.loads((args.suite / "holdout/comparison.json").read_text())["results"]
    output = args.suite / "holdout/teacher-audit"
    output.mkdir(exist_ok=False)
    book = selection["book"]
    identifiers = plan["head_to_head"]["models"]
    networks, specs = {}, []
    for identifier in identifiers:
        row = results[identifier]
        path = Path(row["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Frozen model identity changed")
        networks[identifier] = load_checkpoint(path).network.eval()
        specs.append(_spec(identifier, "dqn", {"checkpoint": str(path), "sha256": row["sha256"], "device": "cpu"}))
    if not args.fidelity_only:
        _write(output / "head-to-head.json", _series(specs, list(itertools.combinations(specs, 2)),
            book[:plan["head_to_head"]["opening_pairs"]], output))
        greedy = _spec("greedy", "greedy", {})
        teachers = [_spec(f"minimax_d{d}", "minimax", {"depth": d}) for d in (1, 2)]
        teacher_report = _series([greedy, *teachers], [(t, greedy) for t in teachers], book, output)
        # Star benchmark: compare cells, not pooled scores against unlike opponents.
        teacher_report.pop("ranking")
        teacher_report.pop("rating")
        teacher_report["scope"] = "Each teacher versus the same Greedy; read the pair cells, not a pooled ranking"
        _write(output / "teachers-vs-greedy.json", teacher_report)

    # A common diagnostic state bank: all held-out openings plus decision
    # states from the first 20 paired games of each prespecified model.
    states = {}
    excluded_keys = set()
    for path in plan.get("fidelity", {}).get("exclude_state_keys", []):
        excluded_keys.update(json.loads(Path(path).read_text()))
    filtered = 0
    def remember(state):
        nonlocal filtered
        obs = np.asarray(encode_observation_cnn(state), dtype=np.float32)
        key = symmetry_key(obs)
        if key in excluded_keys:
            filtered += 1
            return
        states.setdefault(key, (state, obs))
    for opening in book:
        remember(_build_state(opening["state"]))
    for identifier in identifiers:
        with gzip.open(args.suite / "holdout" / identifier / "replays.json.gz", "rt") as stream:
            replays = json.load(stream)
        for opening, block in list(zip(replays["book"], replays["blocks"]))[:20]:
            for game in block["games"]:
                state = _build_state(opening["state"])
                for wire in game["actions"]:
                    remember(state)
                    state = ENGINE.apply_action(state, Move(**wire).engine_action())
    counts = collections.Counter()
    fidelity = {i: collections.Counter() for i in identifiers}
    examples = []
    with torch.no_grad():
        for state, obs in states.values():
            choices, root_values, candidates = {}, {}, {}
            for depth in (1, 2):
                rows, action, _ = teacher_candidates(state, depth)
                candidates[depth] = {encode_engine_action(c["action"]): c["action"] for c in rows}
                choices[depth] = action
                root_values[depth] = max(c["score"] for c in rows)
            counts["states"] += 1
            counts["d1_d2_same_action"] += choices[1] == choices[2]
            gs = greedy_scores(state)
            wins = [c for c in gs if ENGINE.is_game_over(c["state"])]
            greedy_label = encode_engine_action((wins[0] if wins else max(gs, key=lambda c: c["score"]))["action"])
            counts["d1_greedy_demo_same_action"] += choices[1] == greedy_label
            legal = torch.zeros(209, dtype=torch.bool)
            legal[list(candidates[1])] = True
            selected = {}
            for identifier, network in networks.items():
                chosen = int(policy_argmax(network(torch.from_numpy(obs)), legal))
                selected[identifier] = wire_action(candidates[1][chosen])
                for depth in (1, 2):
                    prefix = f"d{depth}_"
                    fidelity[identifier][prefix + "same_action"] += chosen == choices[depth]
                    value = root_values[depth] if chosen == choices[depth] else inspect_branch(state, candidates[depth][chosen], depth)[0]
                    best = root_values[depth]
                    fidelity[identifier][prefix + "equally_good_action"] += value == best
                    fidelity[identifier][prefix + "missed_forced_win"] += best == math.inf and value != math.inf
                    fidelity[identifier][prefix + "avoidable_forced_loss"] += best != -math.inf and value == -math.inf
                    if math.isfinite(best) and math.isfinite(value):
                        fidelity[identifier][prefix + "finite_value_regret_sum"] += best - value
                        fidelity[identifier][prefix + "finite_value_regret_count"] += 1
            if choices[1] != choices[2] and len(examples) < 12:
                examples.append({"position": to_position(state), "cnn_actions": selected,
                    "teacher_actions": {f"d{d}": wire_action(candidates[d][choices[d]]) for d in (1, 2)}})
    training = {}
    for run in args.suite.iterdir():
        if (run / "experiment.json").is_file():
            config = json.loads((run / "experiment.json").read_text())
            if "distillation" in config:
                identifier = next((i for i in identifiers if i.startswith(config["training"]["agent_id"] + "_")), None)
                if identifier is None:
                    continue
                data = torch.load(config["distillation"]["dataset"], map_location="cpu", weights_only=True)["train"]
                generator = torch.Generator().manual_seed(2026102302)
                ix = torch.randperm(len(data["obs"]), generator=generator)[:2048]
                correct = 0
                with torch.no_grad():
                    for batch in ix.split(128):
                        chosen = policy_argmax(networks[identifier](data["obs"][batch]), data["legal"][batch])
                        correct += int((chosen == data[f"expert_d{config['distillation']['depth']}"][batch]).sum())
                own = data["obs"][:, 0].flatten(1).argmax(1)
                training[identifier] = {"sampled_replay_states": len(ix), "expert_action_matches": correct,
                    "accuracy": correct/len(ix), "scope": "unaugmented shared training positions, not validation; reflected optimal ties can differ",
                    "total_replay_states": len(own), "own_pawn_file_counts": torch.bincount(own//9, minlength=9).tolist(),
                    "no_placed_walls": int((data["obs"][:, 2:4].sum((1, 2, 3)) == 0).sum())}
                continue
        if not (run / "teacher-replay.pt").is_file():
            continue
        config = json.loads((run / "experiment.json").read_text())
        identifier = config["training"]["agent_id"] + "_ep02000"
        data = torch.load(run / "teacher-replay.pt", map_location="cpu", weights_only=True)["train"]
        generator = torch.Generator().manual_seed(2026102302)
        ix = torch.randperm(len(data["obs"]), generator=generator)[:2048]
        correct = 0
        with torch.no_grad():
            for batch in ix.split(128):
                chosen = policy_argmax(networks[identifier](data["obs"][batch]), data["legal"][batch])
                correct += int((chosen == data["expert"][batch]).sum())
        own = data["obs"][:, 0].flatten(1).argmax(1)
        winning = data["done"] & (data["reward"] > 0)
        training[identifier] = {"sampled_replay_states": len(ix), "expert_action_matches": correct,
            "accuracy": correct/len(ix), "scope": "training replay, not validation",
            "total_replay_states": len(own), "own_pawn_file_counts": torch.bincount(own//9, minlength=9).tolist(),
            "winning_actions_by_file": torch.bincount(data["action"][winning]//9, minlength=9).tolist(),
            "no_placed_walls": int((data["obs"][:, 2:4].sum((1, 2, 3)) == 0).sum())}
    report = {"common_state_counts": dict(counts), "fidelity": {k: dict(v) for k, v in fidelity.items()},
        "training_replay": training, "examples": examples,
        "excluded_training_symmetry_groups": len(excluded_keys), "filtered_state_visits": filtered,
        "scope": ("All bank states exclude the supplied training symmetry groups. " if excluded_keys else
                  "Diagnostic bank includes evaluation trajectories; only opening states were excluded from all recorded training geometries. ") +
                 "Full-window equality allows teacher-optimal ties. The bank is common to all models and never used for training."}
    _write(output / "fidelity.json", report)
    (output / "audit-source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
