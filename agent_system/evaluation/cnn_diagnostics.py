"""Paired Greedy matches and Decision Lab probes for every CNN checkpoint."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path

import torch
from quoridor_engine import Player

from agent_system.analysis.neural import Runner
from agent_system.analysis.state import ENGINE, Move, Position, build_state, to_position, wire_action
from agent_system.evaluation.arena.agents.scorers import _build_state
from agent_system.evaluation.arena.matrix import play_block, stable_hash, _materialized, decision_summary
from agent_system.evaluation.arena.ratings import score_interval
from agent_system.training.dqn.action_space import encode_engine_action
from agent_system.training.dqn.model import policy_argmax


def diagnose_checkpoint(checkpoint, model_id, book, output, max_plies=300, search=None):
    """Evaluation never changes weights or applies a tactical action override.

    Games within each opening are correlated; confidence bounds use the mean
    of each seat-swapped pair. Development results must not be sold as holdout.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    spec = {"id": model_id, "algo": {"type": "dqn", "params": {
        "checkpoint": str(checkpoint.resolve()), "sha256": digest, "device": "cpu"}},
        "policy": {"type": "top_k", "k": 1}, "deterministic": True}
    search_scorer = None
    if search is not None:
        from agent_system.evaluation.arena.agents.mcts_scorer import MCTSScorer
        spec["algo"]["type"] = "mcts"
        spec["algo"]["params"].update(search)
        search_scorer = MCTSScorer(**spec["algo"]["params"])
    greedy = {"id": "greedy", "algo": {"type": "greedy", "params": {}},
              "policy": {"type": "top_k", "k": 1}, "deterministic": True}
    blocks = [play_block(spec, greedy, opening, max_plies=max_plies, seed=20261020) for opening in book]
    with gzip.open(output / "replays.json.gz", "wt") as stream:
        json.dump({"book": book, "blocks": blocks}, stream)
    runner = Runner(str(checkpoint), digest)
    counts = collections.Counter()
    seats = {"p1": collections.Counter(), "p2": collections.Counter()}
    terminations, pair_scores, examples = collections.Counter(), [], []
    active = torch.zeros(runner.network.dense_width, dtype=torch.bool)
    max_q = 0.0
    for opening, block in zip(book, blocks):
        points = []
        for game in block["games"]:
            if game["error"]:
                raise RuntimeError(f"Invalid benchmark game: {game['error']}")
            result = "draws" if game["winner"] is None else "wins" if game["winner"] == model_id else "losses"
            points.append({"wins": 1.0, "draws": .5, "losses": 0.0}[result])
            counts[result] += 1
            seat = 1 if game["p1"] == model_id else 2
            seats[f"p{seat}"][result] += 1
            terminations[game["termination"]] += 1
            state = _build_state(opening["state"])
            for ply, wire in enumerate(game["actions"]):
                action = Move(**wire).engine_action()
                after = ENGINE.apply_action(state, action)
                if wire["player"] == seat:
                    _, q, _, tensors, type_q = runner.forward(state, layer="head.1", heads=True)
                    hidden = tensors["head.1"].reshape(-1)
                    legal = sorted(ENGINE.legal_actions(state), key=encode_engine_action)
                    mask = torch.zeros_like(q, dtype=torch.bool)
                    mask[[encode_engine_action(a) for a in legal]] = True
                    policy_index = int(policy_argmax(q, mask, type_q))
                    chosen_index = encode_engine_action(action) if search is not None else policy_index
                    counts["search_changed_policy_argmax"] += search is not None and chosen_index != policy_index
                    chosen = next(a for a in legal if encode_engine_action(a) == chosen_index)
                    assert encode_engine_action(chosen) == encode_engine_action(action)
                    active |= hidden != 0
                    counts["decisions"] += 1
                    counts["wall_selected" if chosen_index >= 81 else "pawn_selected"] += 1
                    counts["all_hidden_zero"] += bool((hidden == 0).all())
                    counts["output_exactly_bias"] += torch.equal(q, runner.network.head[-1].bias.detach())
                    max_q = max(max_q, max(abs(q[encode_engine_action(a)].item()) for a in legal))
                    wins = [a for a in legal if a.kind == "MovePawn" and a.target_y == (8 if seat == 1 else 0)]
                    counts["immediate_win_available"] += bool(wins)
                    missed = bool(wins) and not ENGINE.is_game_over(after)
                    counts["missed_immediate_win"] += missed
                    counts["missed_win_wrong_type"] += missed and chosen_index >= 81 and type_q is not None
                    if missed and len(examples) < 8:
                        examples.append({"opening_id": opening["id"], "ply": ply,
                            "position": to_position(state), "selected": wire,
                            "winning": [wire_action(a) for a in wins]})
                state = after
        pair_scores.append(sum(points) / 2)
    tactical = []
    for seat in (1, 2):
        for x in range(9):
            position = Position(pawns=((x, 7), (4, 3)) if seat == 1 else ((4, 5), (x, 1)), turn=seat)
            state = build_state(position)
            _, q, _, _, type_q = runner.forward(state, heads=True)
            legal = sorted(ENGINE.legal_actions(state), key=encode_engine_action)
            mask = torch.zeros_like(q, dtype=torch.bool)
            mask[[encode_engine_action(a) for a in legal]] = True
            chosen_index = int(policy_argmax(q, mask, type_q))
            if search_scorer is not None:
                from backend.adapters.engine_adapter import _serialize_state
                chosen_wire = max(search_scorer.score(_serialize_state(state)), key=lambda pair: pair[1])[0]
                chosen_index = encode_engine_action(Move(**chosen_wire).engine_action())
            chosen = next(a for a in legal if encode_engine_action(a) == chosen_index)
            tactical.append({"position": position.model_dump(mode="json"), "action": wire_action(chosen),
                             "win": ENGINE.is_game_over(ENGINE.apply_action(state, chosen))})
    result = {"model_id": model_id, "checkpoint": str(checkpoint), "sha256": digest,
              "search": search, **decision_summary([g for b in blocks for g in b["games"]], model_id),
              "output_semantics": runner.inference.metadata["output_semantics"],
              "max_abs_legal_output": max_q,
              "book_sha256": stable_hash(book), "games": len(blocks)*2,
              "unique_openings": len({o['state_hash'] for o in book}), "counts": dict(counts),
              "seats": {k: dict(v) for k, v in seats.items()}, "terminations": dict(terminations),
              "score": sum(pair_scores)/len(pair_scores), "paired_score_95_interval": score_interval(pair_scores),
              "interval_method": "fixed-sample Hoeffding over seat-swapped opening pairs",
              "hidden_units_ever_active": int(active.sum()), "max_abs_legal_q": max_q,
              "tactical_wins": sum(t['win'] for t in tactical), "tactical_count": len(tactical),
              "tactical": tactical, "missed_win_examples": examples}
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    _materialized.cache_clear()
    print(json.dumps({k: result[k] for k in ("model_id", "counts", "score", "tactical_wins", "max_abs_legal_q")}), flush=True)
    return result


def benchmark_inference(specs, book, output, warmup_rounds=3):
    """Same-position CPU latency probe using existing Arena agents and masks.

    Run after training/Arena jobs finish. Warmups are reported separately; the
    measured order rotates by opening so a method is not always measured first.
    This measures decisions, not extra games or strength estimates.
    """
    import time
    from agent_system.definition.agent_spec import parse_agent_spec
    from agent_system.evaluation.arena.agents.loader import materialize_agent
    from agent_system.evaluation.arena.agents.core import AgentInstance
    from agent_system.evaluation.arena.matrix import standard_opening
    from backend.adapters.engine_adapter import _action_to_engine
    output = Path(output)
    if output.exists():
        raise ValueError("Latency results already exist")
    instances = {s["id"]: AgentInstance(materialize_agent(parse_agent_spec(s)), seed=0) for s in specs}
    for _ in range(warmup_rounds):
        for instance in instances.values():
            instance.act(standard_opening()["state"])
    rows = []
    ids = list(instances)
    for index, opening in enumerate(book):
        order = ids[index % len(ids):] + ids[:index % len(ids)]
        for identifier in order:
            start = time.perf_counter()
            action = instances[identifier].act(opening["state"])
            elapsed = time.perf_counter() - start
            # Validate outside the measured region.
            ENGINE.apply_action(_build_state(opening["state"]), _action_to_engine(action))
            rows.append({"model_id": identifier, "opening_id": opening["id"],
                "milliseconds": 1000*elapsed, "action": action,
                "search_cost": instances[identifier].last_decision_stats})
    summary = {}
    for identifier in ids:
        latencies = sorted(r["milliseconds"] for r in rows if r["model_id"] == identifier)
        counts = collections.Counter()
        for r in rows:
            if r["model_id"] == identifier:
                for key, value in r["search_cost"].items():
                    counts[key] = max(counts[key], value) if key == "max_depth" else counts[key] + value
        summary[identifier] = {"positions": len(latencies), "mean_ms": sum(latencies)/len(latencies),
            "p50_ms": latencies[len(latencies)//2], "p95_ms": latencies[min(len(latencies)-1,int(.95*len(latencies)))],
            "counters": dict(counts)}
    result = {"summary": summary, "rows": rows, "agents": specs, "book_sha256": stable_hash(book),
        "warmup_rounds_per_agent": warmup_rounds, "torch_threads": torch.get_num_threads(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "CPU latency on identical heldout root positions, rotated method order; no games generated"}
    output.write_text(json.dumps(result, indent=2))
    return result
