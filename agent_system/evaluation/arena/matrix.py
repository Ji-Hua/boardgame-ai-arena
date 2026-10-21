"""Reproducible seat-swapped opening benchmark using canonical Arena agents."""
from __future__ import annotations

import collections
import functools
import hashlib
import json
import random
import time

from quoridor_engine import Player, RuleEngine
from backend.adapters.engine_adapter import _action_to_engine, _serialize_state
from agent_system.definition.agent_spec import parse_agent_spec
from agent_system.evaluation.arena.agents.core import AgentInstance
from agent_system.evaluation.arena.agents.loader import materialize_agent
from agent_system.evaluation.arena.agents.scorers import _build_state, _engine_action_to_wire
from agent_system.evaluation.arena.ratings import fit_elo, score_interval
from agent_system.training.dqn.action_space import encode_engine_action


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_openings(count, seed):
    engine = RuleEngine.standard()
    book = []
    for i in range(count):
        rng = random.Random(seed + i)
        state = engine.initial_state()
        actions = []
        for _ in range(4):
            legal = sorted(engine.legal_actions(state), key=encode_engine_action)
            pawn = [a for a in legal if str(a.kind) == "MovePawn"]
            walls = [a for a in legal if str(a.kind) != "MovePawn"]
            candidates = pawn if rng.random() < 0.7 or not walls else walls
            action = rng.choice(candidates)
            seat = 1 if state.current_player == Player.P1 else 2
            actions.append(_engine_action_to_wire(action, seat))
            state = engine.apply_action(state, action)
        wire = _serialize_state(state)
        book.append({"id": i, "seed": seed+i, "actions": actions,
                     "state": wire, "state_hash": stable_hash(wire)})
    return book


def standard_opening():
    wire = _serialize_state(RuleEngine.standard().initial_state())
    return {"id": -1, "seed": 0, "actions": [], "state": wire, "state_hash": stable_hash(wire)}


@functools.lru_cache(maxsize=64)
def _materialized(encoded):
    return materialize_agent(parse_agent_spec(json.loads(encoded)))


def worker_init():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _memoryless_deterministic(spec):
    return spec.get("deterministic", True) and spec.get("policy", {}).get("k", 1) == 1


def play_block(spec_a, spec_b, opening, max_plies=500, seed=0):
    """One atomic pair: same opening, identity-bound RNG seeds, opposite seats."""
    specs = [spec_a, spec_b]
    definitions = {s["id"]: _materialized(json.dumps(s, sort_keys=True)) for s in specs}
    seeds = {s["id"]: int(stable_hash([seed, opening["id"], s["id"]])[:12], 16) for s in specs}
    deterministic = all(_memoryless_deterministic(s) for s in specs)
    games = []
    engine = RuleEngine.standard()
    for order in (specs, specs[::-1]):
        ids = [s["id"] for s in order]
        instances = {i+1: AgentInstance(definitions[key], seeds[key]) for i, key in enumerate(ids)}
        state = _build_state(opening["state"])
        actions = []
        times = {key: 0.0 for key in ids}
        moves = {key: 0 for key in ids}
        latencies = {key: [] for key in ids}
        diagnostics = {key: collections.Counter() for key in ids}
        repeated_states = 0
        seen = set()
        winner = offender = error = None
        termination = "ply_limit"
        start = time.monotonic()
        for _ in range(max_plies):
            wire = _serialize_state(state)
            key = stable_hash(wire)
            repeated_states += key in seen
            if deterministic and key in seen:
                termination = "deterministic_cycle"
                break
            seen.add(key)
            seat = wire["current_player"]
            active = ids[seat-1]
            # Pawn moves alone can win immediately; include jumps near the goal.
            y = state.pawn_pos(state.current_player)[1]
            near_goal = y >= 6 if seat == 1 else y <= 2
            immediate = any(a.kind == "MovePawn" and a.target_y == (8 if seat == 1 else 0)
                            for a in engine.legal_actions(state)) if near_goal else False
            diagnostics[active]["immediate_win_available"] += immediate
            before = time.monotonic()
            try:
                action = instances[seat].act(wire)
            except Exception as exc:
                offender, winner = active, ids[2-seat]
                termination, error = "agent_error", f"{type(exc).__name__}: {exc}"[:300]
                break
            finally:
                elapsed = time.monotonic()-before
                times[active] += elapsed
                latencies[active].append(1000 * elapsed)
                moves[active] += 1
            for stat, value in instances[seat].last_decision_stats.items():
                if stat == "max_depth":
                    diagnostics[active][stat] = max(diagnostics[active][stat], value)
                else:
                    diagnostics[active][stat] += value
            try:
                if action.get("player") != seat:
                    raise ValueError("Wrong action player")
                state = engine.apply_action(state, _action_to_engine(action))
            except Exception as exc:
                offender, winner = active, ids[2-seat]
                termination, error = "illegal_action", f"{type(exc).__name__}: {exc}"[:300]
                break
            actions.append(action)
            diagnostics[active]["missed_immediate_win"] += immediate and not engine.is_game_over(state)
            if engine.is_game_over(state):
                winning_seat = 1 if engine.winner(state) == Player.P1 else 2
                winner, termination = ids[winning_seat-1], "goal"
                break
        final_state = _serialize_state(state)
        final_state["game_over"] = engine.is_game_over(state)
        final_state["winner"] = (ids.index(winner)+1) if termination == "goal" else None
        games.append({
            "p1": ids[0], "p2": ids[1], "winner": winner, "steps": len(actions),
            "termination": termination, "offender": offender, "error": error,
            "seed_by_agent": seeds, "actions": actions,
            "final_state": final_state,
            "decision_seconds": times, "decisions": moves,
            "decision_milliseconds": latencies,
            "decision_diagnostics": {k: dict(v) for k, v in diagnostics.items()},
            "repeated_states": repeated_states,
            "seconds": time.monotonic()-start,
            "trajectory_hash": stable_hash([opening["state_hash"], actions]),
        })
    return {"a": spec_a["id"], "b": spec_b["id"], "opening_id": opening["id"],
            "opening_hash": opening["state_hash"], "control": opening["id"] < 0, "games": games}


def _points(game, agent):
    return 0.5 if game["winner"] is None else float(game["winner"] == agent)


def decision_summary(games, agent):
    """Optional fields preserve compatibility with historical Arena blocks."""
    samples = sorted(x for g in games for x in g.get("decision_milliseconds", {}).get(agent, []))
    counters = collections.Counter()
    for g in games:
        for stat, value in g.get("decision_diagnostics", {}).get(agent, {}).items():
            if stat == "max_depth":
                counters[stat] = max(counters[stat], value)
            else:
                counters[stat] += value
    return {"decision_diagnostics": dict(counters),
            "mean_decision_ms": 1000 * sum(g["decision_seconds"][agent] for g in games) / max(1, sum(g["decisions"][agent] for g in games)),
            "p50_decision_ms": samples[len(samples)//2] if samples else None,
            "p95_decision_ms": samples[min(len(samples)-1, int(.95*len(samples)))] if samples else None,
            "repetition_games": sum(g.get("repeated_states", 0) > 0 for g in games),
            "repetition_rate": sum(g.get("repeated_states", 0) > 0 for g in games) / len(games) if games else None}


def summarize(specs, blocks, expected_blocks):
    ids = [s["id"] for s in specs]
    matrix = {key: {} for key in ids}
    pair_rows = []
    rank_rows = []
    ranked = [b for b in blocks if not b["control"]]
    control = [b for b in blocks if b["control"]]
    all_games = [g for b in blocks for g in b["games"]]
    for a in ids:
        for b in ids:
            if a == b:
                matrix[a][b] = None
                continue
            series = [x for x in ranked if {x["a"], x["b"]} == {a, b}]
            games = [g for x in series for g in x["games"]]
            scores = [sum(_points(g, a) for g in x["games"])/2 for x in series]
            w = sum(g["winner"] == a for g in games)
            losses = sum(g["winner"] == b for g in games)
            draws = len(games)-w-losses
            cell = {
                "games": len(games), "pairs": len(series), "wins": w, "losses": losses, "draws": draws,
                "score": (w+draws/2)/len(games) if games else None,
                "ci95": score_interval(scores),
                "win_ci95": score_interval([sum(g["winner"] == a for g in x["games"])/2 for x in series]),
                "draw_ci95": score_interval([sum(g["winner"] is None for g in x["games"])/2 for x in series]),
                "loss_ci95": score_interval([sum(g["winner"] == b for g in x["games"])/2 for x in series]),
                **decision_summary(games, a),
                "p1_games": sum(g["p1"] == a for g in games),
                "p2_games": sum(g["p2"] == a for g in games),
                "p1_score": sum(_points(g, a) for g in games if g["p1"] == a)/len(series) if series else None,
                "p2_score": sum(_points(g, a) for g in games if g["p2"] == a)/len(series) if series else None,
                "terminations": dict(collections.Counter(g["termination"] for g in games)),
                "unique_trajectories": len({g["trajectory_hash"] for g in games}),
            }
            matrix[a][b] = cell
            if ids.index(a) < ids.index(b) and games:
                pair_rows.append((a, b, w+draws/2, len(games)))
        games = [g for block in ranked if a in (block["a"], block["b"]) for g in block["games"]]
        points = sum(_points(g, a) for g in games)
        opening_scores = collections.defaultdict(list)
        for block in ranked:
            if a in (block["a"], block["b"]):
                opening_scores[block["opening_id"]].append(sum(_points(g, a) for g in block["games"])/2)
        block_means = [sum(v)/len(v) for v in opening_scores.values()]
        rank_rows.append({
            "id": a, "games": len(games), "score": points/len(games) if games else None,
            **decision_summary(games, a),
            "wins": sum(g["winner"] == a for g in games),
            "draws": sum(g["winner"] is None for g in games),
            "losses": sum(g["winner"] not in (a, None) for g in games),
            "ci95": score_interval(block_means),
            "forfeits": sum(g["offender"] == a for g in games),
            "mean_decision_ms": 1000*sum(g["decision_seconds"][a] for g in games)/max(1,sum(g["decisions"][a] for g in games)),
            "p1_score": sum(_points(g,a) for g in games if g["p1"]==a)/max(1,sum(g["p1"]==a for g in games)),
            "p2_score": sum(_points(g,a) for g in games if g["p2"]==a)/max(1,sum(g["p2"]==a for g in games)),
        })
    anchor = "random_legal" if "random_legal" in ids else ids[0]
    ratings = fit_elo(ids, pair_rows, anchor)
    for row in rank_rows:
        row["elo"] = ratings[row["id"]]
    rank_rows.sort(key=lambda r: (-(r["score"] if r["score"] is not None else -1), -r["elo"], r["id"]))
    for i, row in enumerate(rank_rows):
        previous = rank_rows[i-1] if i else None
        row["rank"] = (previous["rank"] if previous and row["score"] == previous["score"] else i+1)
    controls = [{"a": b["a"], "b": b["b"],
                 "a_score": sum(_points(g,b["a"]) for g in b["games"])/2,
                 "winners": [g["winner"] for g in b["games"]],
                 "terminations": [g["termination"] for g in b["games"]]} for b in control]
    return {
        "status": "complete" if len(blocks) == expected_blocks else "running",
        "completed_blocks": len(blocks), "expected_blocks": expected_blocks,
        "ranked_games": len(ranked)*2, "control_games": len(control)*2,
        "agents": specs, "matrix": matrix, "ranking": rank_rows, "standard_controls": controls,
        "terminations": dict(collections.Counter(g["termination"] for g in all_games)),
        "forfeits": [dict(p1=g["p1"], p2=g["p2"], offender=g["offender"], error=g["error"]) for g in all_games if g["offender"]],
        "confidence": "95% fixed-sample Hoeffding bounds on seat-swapped pair scores; marginal, conservative",
        "rating": f"Regularized batch Elo; {anchor}=1500; provisional within this protocol",
    }
