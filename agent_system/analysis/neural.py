"""DQN diagnostics over the shared runtime model and observation encoder."""
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import threading
import time

import torch
from torch import nn

from agent_system.definition.agent_spec import AgentSpec
from agent_system.runtime.inference.dqn import DQNInference
from agent_system.runtime.service.agents import greedy_agent
from agent_system.training.dqn.action_space import ACTION_COUNT, decode_action_id, encode_engine_action
from agent_system.training.dqn.model import network_outputs, route_mask
from . import explain, neural_catalog
from .state import ENGINE, action_id, build_state, notation, snapshot, to_position, wire_action

# This independent diagnostic process never reserves the training GPU.
torch.set_num_threads(1)
LOAD_LOCK = threading.RLock()
MAX_PARAMETERS = 20_000_000
WINDOW_SIZE = 256
INPUT_NAMES = ["己方棋子", "对方棋子", "横墙锚点", "竖墙锚点", "己方墙库存 / 10", "对方墙库存 / 10", "己方目标行"]


def stats(tensor, histogram=False):
    flat = tensor.detach().double().reshape(-1)
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("网络出现非有限数值（NaN/Inf），未返回分析结果。")
    result = {"count": flat.numel(), "min": flat.min().item(), "max": flat.max().item(),
              "mean": flat.mean().item(), "std": flat.std(unbiased=False).item(),
              "zero_fraction": (flat == 0).double().mean().item()}
    if histogram:
        low, high = result["min"], result["max"]
        if low == high:
            radius = max(abs(low)*.01, .5)
            low, high = low-radius, high+radius
        result["histogram"] = {"edges": torch.linspace(low, high, 21, dtype=torch.float64).tolist(),
                               "counts": torch.histc(flat, bins=20, min=low, max=high).to(torch.int64).tolist()}
    return result


def input_views(tensor):
    """Retain encoder indexing [x,y], including MLP's 8x8 wall grids."""
    data = tensor.detach().cpu()
    if data.ndim == 3:
        return [{"name": name, "layout": "board_xy", "values": data[i].tolist()}
                for i, name in enumerate(INPUT_NAMES)]
    views = []
    for name, offset, size in zip(INPUT_NAMES, (0, 81, 162, 226), (9, 9, 8, 8)):
        views.append({"name": name, "layout": "board_xy", "values": data[offset:offset+size*size].reshape(size, size).tolist()})
    views.extend({"name": INPUT_NAMES[i+4], "layout": "scalar", "values": [[data[290+i].item()]]} for i in range(2))
    return views


def source_provenance():
    from agent_system.runtime.inference import dqn
    from agent_system.training.dqn import model, checkpoint, observation, observation_cnn, action_space
    paths = [Path(__file__), Path(neural_catalog.__file__)] + [Path(m.__file__) for m in
             (dqn, model, checkpoint, observation, observation_cnn, action_space)]
    digest = hashlib.sha256(explain.PROVENANCE["source_sha256"].encode() + b"".join(p.read_bytes() for p in paths)).hexdigest()
    return {**explain.PROVENANCE, "schema": "quoridor-neural-analysis-v1", "source_sha256": digest,
            "tie_break": "lowest_action_index", "device": "cpu", "torch_version": str(torch.__version__)}


PROVENANCE = source_provenance()


class Runner:
    def __init__(self, path, digest):
        self.inference = DQNInference(path, digest, "cpu")
        self.network = self.inference.network
        self.parameter_count = sum(p.numel() for p in self.network.parameters())
        if self.parameter_count > MAX_PARAMETERS:
            raise ValueError(f"模型超过分析上限 {MAX_PARAMETERS:,} 个参数。")
        self.modules = {name: module for name, module in self.network.named_modules()
                        if name and not list(module.children())}
        self.weights = {}
        for name, module in self.modules.items():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.weights[name] = {"shape": list(module.weight.shape), "stats": stats(module.weight, True),
                                      "bias": module.bias.detach().tolist() if module.bias is not None else None}
        self.lock = threading.RLock()

    def forward(self, state, *, capture=False, layer=None, gradients=False, heads=False):
        x = torch.tensor(self.inference.encoder(state), dtype=torch.float32, requires_grad=gradients)
        layers, tensors, hooks = [], {}, []

        def observe(name, module, inputs, output):
            # CNN inserts a singleton batch dimension; remove only that dimension.
            value = output[0] if output.ndim in (2, 4) and output.shape[0] == 1 else output
            if capture:
                layers.append({"name": name, "type": type(module).__name__, "shape": list(value.shape),
                               "activation": stats(value), "weight": self.weights.get(name)})
            if name == layer:
                tensors[name] = value.detach().clone()

        try:
            if capture or layer:
                for name, module in self.modules.items():
                    hooks.append(module.register_forward_hook(lambda m, i, o, name=name: observe(name, m, i, o)))
            with torch.set_grad_enabled(gradients):
                if getattr(self.network, "value_head", None) is not None:
                    q, _ = self.network.forward_policy_value(x)
                    type_q = None
                else:
                    q, type_q = network_outputs(self.network, x)
                q = q.reshape(-1)
            if q.numel() != ACTION_COUNT or not bool(torch.isfinite(q).all()):
                raise ValueError("网络输出必须是 209 个有限动作评分。")
            if type_q is not None and not bool(torch.isfinite(type_q).all()):
                raise ValueError("类别头必须输出有限 Q 值。")
            return (x, q, layers, tensors, type_q) if heads else (x, q, layers, tensors)
        finally:
            for hook in hooks:
                hook.remove()


@lru_cache(maxsize=2)
def cached_runner(path, digest, file_identity):
    return Runner(path, digest)


def get_runner(identifier):
    entry = neural_catalog.resolve(identifier)
    try:
        stat = Path(entry["checkpoint"]).stat()
        if stat.st_size > 512*1024*1024:
            raise ValueError("checkpoint 超过 512 MB 分析上限。")
        with LOAD_LOCK:
            runner = cached_runner(entry["checkpoint"], entry["sha256"], (stat.st_ino, stat.st_size, stat.st_mtime_ns))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"模型加载失败：{exc}") from exc
    return entry, runner


def identity(entry, runner):
    return {**entry, "metadata": runner.inference.metadata, "parameters": runner.parameter_count,
            "output_semantics": runner.inference.metadata["output_semantics"],
            "architecture": runner.inference.metadata["model_config"].get("model_arch", "mlp")}


def all_outputs(state, q, type_q=None):
    legal = {encode_engine_action(a) for a in ENGINE.legal_actions(state)} if not ENGINE.is_game_over(state) else set()
    rows = []
    routed = None
    if type_q is not None and legal:
        mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
        mask[list(legal)] = True
        routed, _, _ = route_mask(type_q, mask)
    for index, value in enumerate(q.detach().tolist()):
        action = decode_action_id(index, state.current_player)
        rows.append({"action_index": index, "id": action_id(action), "label": notation(action),
                     "action": wire_action(action), "value": value, "legal": index in legal,
                     "masked_value": value if index in legal else "-inf"})
        if type_q is not None:
            eligible = bool(routed[index]) if routed is not None else False
            rows[-1].update(route_eligible=eligible, routed_value=value if eligible else "-inf")
    return rows


def analyze(position, model_id):
    start = time.monotonic()
    state = build_state(position)
    entry, runner = get_runner(model_id)
    with runner.lock:
        x, q, layers, _, type_q = runner.forward(state, capture=True, heads=True)
    outputs = all_outputs(state, q, type_q)
    legal = [r for r in outputs if r["legal"]]
    ranked = sorted([r for r in legal if r.get("route_eligible", True)], key=lambda r: (-r["value"], r["action_index"]))
    chosen = ranked[0] if ranked else None
    raw_best = max(outputs, key=lambda r: r["value"])
    evidence = {action_id(r["action"]): r for r in greedy_agent.score_position(state)} if legal else {}
    candidates = []
    for row in legal:
        g = evidence[row["id"]]
        candidates.append({**row, "order": row["action_index"], "bound": "exact", "greedy_score": g["score"],
                           "before_self_opp": list(g["before"]), "after_self_opp": list(g["after"]),
                           "delta_self": g["delta_self"], "delta_opp": g["delta_opp"],
                           "immediate_win": ENGINE.winner(g["state"]) == state.current_player,
                           "nodes": 0, "cutoffs": 0})
    spec = AgentSpec(id=entry["id"], algo_type="dqn", algo_params={"checkpoint": entry["checkpoint"],
                     "sha256": entry["sha256"], "device": "cpu"}, policy_type="top_k", policy_params={"k": 1})
    snap = snapshot(state)
    hierarchy = None
    if type_q is not None:
        groups = [("pawn", "走子", [r for r in legal if r["action_index"] < 81]),
                  ("wall", "放墙", [r for r in legal if r["action_index"] >= 81])]
        hierarchy = {"selected_type": ("pawn" if chosen["action_index"] < 81 else "wall") if chosen else None,
            "selection": "type Q argmax, then action Q argmax within selected type",
            "types": [{"id": key, "label": label, "q": type_q[i].item(), "available": bool(group),
                       "legal_actions": len(group), "best_action": max(group, key=lambda r: r["value"]) if group else None}
                      for i, (key, label, group) in enumerate(groups)]}
    return {**snap, "provenance": {**PROVENANCE, "checkpoint_sha256": entry["sha256"]},
            "created_at": datetime.now(timezone.utc).isoformat(), "position_hash": explain.position_hash(snap["position"]),
            "algorithm": "dqn", "depth": 1, "agent_spec": asdict(spec), "model_id": model_id,
            "candidates": candidates, "selected_id": chosen["id"] if chosen else None, "complete": True,
            "neural": {"model": identity(entry, runner), "input": {"shape": list(x.shape), "values": x.detach().tolist(),
                       "views": input_views(x), "perspective": snap["position"]["turn"], "axes": "[x,y], a1 bottom-left; no board flip"},
                       "outputs": outputs, "layers": layers, "raw_best": raw_best, "hierarchy": hierarchy,
                       "value": next((math.tanh(l["activation"]["mean"]) for l in layers if l["name"] == "value_head"), None),
                       "legal_margin": ranked[0]["value"]-ranked[1]["value"] if len(ranked) > 1 else None,
                       "illegal_above_selected": sum(not r["legal"] and r["value"] > chosen["value"] for r in outputs) if chosen else 0},
            "stats": {"nodes": 0, "cutoffs": 0, "forward_passes": 1, "seconds": time.monotonic()-start}}


def search(position, model_id, simulations=50, c_puct=1.5):
    """Inspect the exact same deterministic PUCT implementation used by Arena."""
    from agent_system.runtime.search.puct import SearchConfig, TorchEvaluator, search_batch
    from agent_system.runtime.search import puct
    start = time.monotonic()
    state = build_state(position)
    entry, runner = get_runner(model_id)
    if runner.inference.metadata["eval_summary"].get("value_head_status") not in {"calibrated", "trained"}:
        raise ValueError("该模型没有经过真实对局校准的 Value Head。")
    with runner.lock:
        evaluator = TorchEvaluator(runner.network)
        result = search_batch([state], evaluator, SearchConfig(simulations=simulations, c_puct=c_puct))[0]
    data = result.details()
    for row in data["actions"]:
        action = decode_action_id(row["action_index"], state.current_player)
        row.update(id=action_id(action), label=notation(action), action=wire_action(action))
    return {**data, "model_id": model_id, "checkpoint_sha256": entry["sha256"],
            "perspective": position.turn,
            "value_semantics": "current_player_outcome", "c_puct": c_puct,
            "noise_fraction": 0., "seconds": time.monotonic()-start,
            "forward_batches": evaluator.stats["forward_batches"],
            "provenance": {**PROVENANCE, "search_source_sha256": hashlib.sha256(
                Path(puct.__file__).read_bytes()).hexdigest()}}


def branch(position, move, model_id):
    start = time.monotonic()
    state = build_state(position)
    action = move.engine_action()
    after = ENGINE.apply_action(state, action)  # Validate before differentiating any action.
    entry, runner = get_runner(model_id)
    index = encode_engine_action(action)
    with runner.lock:
        x, q, _, _, type_q = runner.forward(state, gradients=True, heads=True)
        legal = sorted(encode_engine_action(a) for a in ENGINE.legal_actions(state))
        others = [i for i in legal if i != index and (type_q is None or (i >= 81) == (index >= 81))]
        reference = max(others, key=lambda i: float(q[i].detach())) if others else None
        objectives = [("q", q[index], None)]
        if reference is not None:
            objectives.append(("margin", q[index]-q[reference], reference))
        if type_q is not None:
            kind = int(index >= 81)
            objectives.append(("type_q", type_q[kind], None))
            if any((i >= 81) != (index >= 81) for i in legal):
                objectives.append(("type_margin", type_q[kind]-type_q[1-kind], None))
        sensitivities = []
        for i, (name, objective, ref) in enumerate(objectives):
            grad, = torch.autograd.grad(objective, x, retain_graph=i < len(objectives)-1)
            stats(grad)
            sensitivities.append({"objective": name, "value": objective.detach().item(),
                                  "reference_index": ref, "reference_label": notation(decode_action_id(ref, state.current_player)) if ref is not None else None,
                                  "gradient": grad.tolist(), "views": input_views(grad)})
    after_snapshot = snapshot(after)
    return {"action_id": action_id(action), "action_index": index, "position_hash": explain.position_hash(to_position(state)),
            "model_id": model_id, "checkpoint_sha256": entry["sha256"], "algorithm": "dqn", "depth": 1,
            "value": q[index].detach().item(), "bound": "exact", "complete": True,
            "principal_variation": [{"action": wire_action(action), "label": notation(action), **after_snapshot}],
            "after": after_snapshot, "trace": {"nodes": [], "omitted": 0},
            "neural": {"sensitivities": sensitivities, "method": "input_gradient", "perspective": position.turn},
            "stats": {"nodes": 0, "cutoffs": 0, "forward_passes": 1, "seconds": time.monotonic()-start}}


def tensor(position, model_id, kind, layer, channel=0, input_channel=0, offset=0):
    if kind not in ("activation", "weight"):
        raise ValueError("不支持的张量类型。")
    state = build_state(position)
    entry, runner = get_runner(model_id)
    module = runner.modules.get(layer)
    if module is None:
        raise ValueError("层名称不存在。")
    with runner.lock:
        if kind == "activation":
            if input_channel:
                raise ValueError("激活切片不接受输入通道。")
            _, _, _, tensors = runner.forward(state, layer=layer)
            value = tensors[layer]
            if value.ndim == 3:
                if not 0 <= channel < value.shape[0]:
                    raise ValueError("激活通道越界。")
                value, layout = value[channel], "board_xy"
            else:
                if channel:
                    raise ValueError("向量激活不接受通道索引。")
                layout = "vector"
        else:
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                raise ValueError("该层没有可查看的权重。")
            if not 0 <= channel < module.weight.shape[0]:
                raise ValueError("输出单元越界。")
            value = module.weight[channel].detach()
            if value.ndim == 3:
                if not 0 <= input_channel < value.shape[0]:
                    raise ValueError("输入通道越界。")
                value, layout = value[input_channel], "matrix"
            else:
                if input_channel:
                    raise ValueError("Linear 权重使用 offset，不接受输入通道。")
                layout = "vector"
        full_stats = stats(value, True)
        shape, total = list(value.shape), value.numel()
        if layout == "vector":
            if not 0 <= offset < total:
                raise ValueError("向量偏移越界。")
            shown = value.reshape(-1)[offset:offset+WINDOW_SIZE].tolist()
            values = [shown[i:i+16] for i in range(0, len(shown), 16)]
        else:
            if offset:
                raise ValueError("二维切片不接受向量偏移。")
            values = value.tolist()
    return {"model_id": model_id, "checkpoint_sha256": entry["sha256"], "position_hash": explain.position_hash(to_position(state)),
            "kind": kind, "layer": layer, "channel": channel, "input_channel": input_channel,
            "layout": layout, "values": values, "shape": shape, "stats": full_stats,
            "window": {"offset": offset, "total": total, "shown": sum(len(row) for row in values)}}
