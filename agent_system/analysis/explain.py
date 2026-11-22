"""Read-only decision explanations using the actual Greedy/Minimax kernels."""
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from agent_system.definition.agent_spec import AgentSpec
from agent_system.runtime.service.agents import greedy_agent, minimax_agent
from .state import ENGINE, action_id, build_state, notation, snapshot, to_position, wire_action

SCHEMA_VERSION = "quoridor-decision-analysis-v1"


class AnalysisLimit(Exception):
    """Search did not complete; no complete recommendation may be emitted."""


def number(value):
    return value if math.isfinite(value) else ("+inf" if value > 0 else "-inf")


def position_hash(position):
    return hashlib.sha256(json.dumps(position, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def provenance():
    files = [Path(__file__), Path(__file__).with_name("state.py"),
             Path(greedy_agent.__file__), Path(minimax_agent.__file__)]
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    import quoridor_engine
    engine_path = Path(quoridor_engine.__file__)
    binaries = list(engine_path.parent.glob("*.so")) if engine_path.suffix == ".py" else [engine_path]
    return {"schema": SCHEMA_VERSION, "source_sha256": source_hash,
            "engine_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
            "tie_break": "first_in_engine_order", "policy": "deterministic_top1"}


PROVENANCE = provenance()


class Recorder:
    """Observe canonical alpha-beta events; storage caps do not prune search."""

    def __init__(self, root_player, *, node_limit=75000, seconds=6, trace_limit=1200):
        self.root_player = root_player
        self.node_limit, self.deadline, self.trace_limit = node_limit, time.monotonic()+seconds, trace_limit
        self.visits = 0
        self.stack, self.nodes, self.roots = [], [], []
        self.incoming = None

    def __call__(self, event, data):
        if event in ("edge", "root_action"):
            self.incoming = wire_action(data["action"])
            return
        if event == "enter":
            self.visits += 1
            if self.visits > self.node_limit or time.monotonic() > self.deadline:
                raise AnalysisLimit("搜索达到资源上限，未产生完整结论。请降低搜索深度后重试。")
            frame = {**data, "id": self.visits, "action": self.incoming,
                     "children": [], "best": None, "line": [], "child_count": 0,
                     "parent": self.stack[-1]["id"] if self.stack else None}
            self.stack.append(frame)
            self.incoming = None
            return
        frame = self.stack.pop()
        value = data["value"]
        bound = "exact" if data["leaf"] else minimax_agent.score_bound(value, frame["alpha"], frame["beta"])
        line = ([frame["action"]] if frame["action"] else []) + frame["line"]
        # Keep by visit id, so the root and early branches survive the cap.
        if frame["id"] <= self.trace_limit:
            self.nodes.append({
                "id": frame["id"], "parent": frame["parent"], "action": frame["action"],
                "label": notation(frame["action"]) if frame["action"] else "root",
                "depth_remaining": frame["depth"], "role": "MAX" if frame["maximizing"] else "MIN",
                "alpha": number(frame["alpha"]), "beta": number(frame["beta"]),
                "value": number(value), "bound": bound, "leaf": data["leaf"],
                "pruned": data["pruned"], "children": frame["children"],
                "visited_children": frame["child_count"], "position": to_position(frame["state"]),
            })
        if self.stack:
            parent = self.stack[-1]
            parent["child_count"] += 1
            if frame["id"] <= self.trace_limit:
                parent["children"].append(frame["id"])
            if parent["best"] is None or (value > parent["best"] if parent["maximizing"] else value < parent["best"]):
                parent["best"], parent["line"] = value, line
        else:
            self.roots.append({"line": line, "bound": bound, "value": number(value), "node_id": frame["id"]})

    def trace(self):
        return {"nodes": sorted(self.nodes, key=lambda x: x["id"]),
                "recorded": len(self.nodes), "visited": self.visits,
                "omitted": self.visits-len(self.nodes)}


def envelope(state, algorithm, depth):
    position = to_position(state)
    spec = AgentSpec(id="greedy" if algorithm == "greedy" else f"minimax_d{depth}",
                     algo_type=algorithm,
                     algo_params={} if algorithm == "greedy" else {"depth": depth},
                     policy_type="top_k", policy_params={"k": 1})
    return {"provenance": PROVENANCE, "created_at": datetime.now(timezone.utc).isoformat(),
            "position_hash": position_hash(position), "algorithm": algorithm,
            "agent_spec": asdict(spec), "depth": depth, **snapshot(state)}


def analyze(position, algorithm="greedy", depth=2, *, node_limit=75000, seconds=6):
    if algorithm not in ("greedy", "minimax"):
        raise ValueError("Unsupported analysis algorithm")
    if depth not in (1, 2, 3):
        raise ValueError("Minimax depth must be 1, 2, or 3")
    start = time.monotonic()
    state = build_state(position)
    result = envelope(state, algorithm, 1 if algorithm == "greedy" else depth)
    if ENGINE.is_game_over(state):
        return {**result, "candidates": [], "selected_id": None, "complete": True,
                "stats": {"seconds": time.monotonic()-start, "nodes": 0, "cutoffs": 0}}
    greedy_rows = greedy_agent.score_position(state)
    evidence = {action_id(r["action"]): r for r in greedy_rows}
    if algorithm == "greedy":
        rows = greedy_rows
        nodes, cutoffs = len(rows), 0
    else:
        recorder = Recorder(state.current_player, node_limit=node_limit, seconds=seconds, trace_limit=0)
        rows, context = minimax_agent.score_position(state, depth, recorder)
        nodes, cutoffs = recorder.visits, context.cutoffs
    selected = max(rows, key=lambda r: r["score"])
    candidates = []
    for index, row in enumerate(rows):
        item_id = action_id(row["action"])
        g = evidence[item_id]
        winner = ENGINE.winner(row["state"])
        candidates.append({
            "id": item_id, "order": index, "action": wire_action(row["action"]),
            "label": notation(row["action"]), "value": number(row["score"]),
            "bound": row.get("bound", "exact"), "greedy_score": g["score"],
            "before_self_opp": list(g["before"]), "after_self_opp": list(g["after"]),
            "delta_self": g["delta_self"], "delta_opp": g["delta_opp"],
            "immediate_win": winner == state.current_player,
            "nodes": row.get("nodes", 1), "cutoffs": row.get("cutoffs", 0),
        })
    return {**result, "candidates": candidates, "selected_id": action_id(selected["action"]),
            "complete": True, "stats": {"nodes": nodes, "cutoffs": cutoffs,
                                       "seconds": time.monotonic()-start}}


def branch(position, move, algorithm="minimax", depth=2, *, node_limit=75000, seconds=6, trace_limit=1200):
    state = build_state(position)
    action = move.engine_action()
    after = ENGINE.apply_action(state, action)
    start = time.monotonic()
    if algorithm == "greedy":
        row = next(r for r in greedy_agent.score_position(state) if action_id(r["action"]) == action_id(action))
        value = row["score"]
        line, trace, nodes, cutoffs = [wire_action(action)], {"nodes": [], "omitted": 0}, 1, 0
    elif algorithm == "minimax" and depth in (1, 2, 3):
        recorder = Recorder(state.current_player, node_limit=node_limit, seconds=seconds, trace_limit=trace_limit)
        value, context = minimax_agent.inspect_branch(state, action, depth, recorder)
        line, trace, nodes, cutoffs = recorder.roots[0]["line"], recorder.trace(), recorder.visits, context.cutoffs
    else:
        raise ValueError("Unsupported algorithm or depth")
    pv, current = [], state
    from .state import Move
    for item in line:
        current = ENGINE.apply_action(current, Move.model_validate(item).engine_action())
        pv.append({"action": item, "label": notation(item), **snapshot(current)})
    return {"action_id": action_id(action), "position_hash": position_hash(to_position(state)),
            "algorithm": algorithm, "depth": 1 if algorithm == "greedy" else depth,
            "value": number(value), "bound": "exact", "principal_variation": pv,
            "after": snapshot(after), "trace": trace, "complete": True,
            "stats": {"nodes": nodes, "cutoffs": cutoffs, "seconds": time.monotonic()-start}}
