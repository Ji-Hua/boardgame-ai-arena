"""Small PUCT search shared by self-play, Arena and Decision Lab.

Independent roots can share a neural batch. Each tree still performs one
sequential simulation at a time: no virtual loss, tree reuse or transpositions.
Edge Q is always from its parent player's perspective. A repeated state on a
search path is adjudicated zero (not an engine terminal).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from quoridor_engine import RuleEngine

from agent_system.training.dqn.action_space import ACTION_COUNT, encode_engine_action
from agent_system.training.dqn.observation_cnn import encode_observation_cnn


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 50
    c_puct: float = 1.5
    max_depth: int = 128
    dirichlet_alpha: float = .15
    noise_fraction: float = 0.

    def __post_init__(self):
        if self.simulations < 1 or self.max_depth < 1 or not math.isfinite(self.c_puct) or self.c_puct <= 0:
            raise ValueError("Positive search budget, depth and c_puct required")
        if not 0 <= self.noise_fraction <= 1 or not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("Invalid root noise configuration")


class Node:
    def __init__(self, state, engine):
        self.state = state
        self.player = state.current_player
        self.terminal = engine.is_game_over(state)
        self.terminal_value = (1. if engine.winner(state) == self.player else -1.) if self.terminal else None
        self.obs = np.asarray(encode_observation_cnn(state), dtype=np.float32)
        # Encoder contains both pawns, all wall anchors/stocks and current goal.
        self.key = self.obs.tobytes()
        self.actions = sorted(engine.legal_actions(state), key=encode_engine_action) if not self.terminal else []
        self.ids = np.asarray([encode_engine_action(a) for a in self.actions], dtype=np.int64)
        self.legal = np.zeros(ACTION_COUNT, dtype=np.bool_)
        self.legal[self.ids] = True
        self.prior = None
        self.visits = np.zeros(len(self.ids), dtype=np.int64)
        self.value_sum = np.zeros(len(self.ids), dtype=np.float64)
        self.children = [None] * len(self.ids)
        self.value = self.terminal_value

    def expand(self, probabilities, value):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (ACTION_COUNT,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Invalid policy probabilities")
        if not np.isfinite(value) or abs(value) > 1.000001:
            raise ValueError("Leaf outcome must be finite and in [-1,1]")
        prior = probabilities[self.ids]
        if prior.sum() <= 0:
            raise ValueError("Policy has no mass on legal actions")
        self.prior = prior / prior.sum()
        self.value = float(value)

    def q(self):
        return np.divide(self.value_sum, self.visits, out=np.zeros_like(self.value_sum), where=self.visits > 0)


def backup(path, leaf_player, value):
    """Convert perspective across EACH edge, including actual terminal leaves."""
    player = leaf_player
    for node, index in reversed(path):
        if node.player != player:
            value = -value
        node.visits[index] += 1
        node.value_sum[index] += value
        player = node.player


class TorchEvaluator:
    def __init__(self, network, device="cpu", meter=None):
        if getattr(network, "value_head", None) is None:
            raise ValueError("MCTS requires an explicit calibrated/trained value head")
        self.network, self.device, self.meter = network, device, meter
        self.stats = Counter()

    def __call__(self, nodes):
        if not nodes:
            return []
        from contextlib import nullcontext
        start = time.monotonic()
        with torch.inference_mode(), (self.meter.measure("search_inference_seconds") if self.meter else nullcontext()):
            obs = torch.from_numpy(np.stack([n.obs for n in nodes])).to(self.device)
            legal = torch.from_numpy(np.stack([n.legal for n in nodes])).to(self.device)
            logits, value = self.network.forward_policy_value(obs)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(value).all()):
                raise ValueError("Non-finite policy/value output")
            probs = logits.masked_fill(~legal, -torch.inf).softmax(-1).cpu().numpy()
            values = value.cpu().numpy()
        self.stats.update(model_evaluations=len(nodes), forward_batches=1,
                          inference_wall_seconds=time.monotonic() - start)
        return list(zip(probs, values))


@dataclass
class SearchResult:
    root: Node
    stats: dict

    @property
    def distribution(self):
        result = np.zeros(ACTION_COUNT, dtype=np.float32)
        if self.root.visits.sum():
            result[self.root.ids] = self.root.visits / self.root.visits.sum()
        return result

    @property
    def ranked_indices(self):
        root = self.root
        return sorted(range(len(root.ids)), key=lambda i: (-root.visits[i], -root.prior[i], int(root.ids[i])))

    @property
    def action_id(self):
        if self.root.terminal:
            raise ValueError("Terminal root has no action")
        return int(self.root.ids[self.ranked_indices[0]])

    def details(self):
        r = self.root
        return {"root_value": r.value, "stats": self.stats,
                "selected_action_index": None if r.terminal else self.action_id,
                "actions": [{"action_index": int(r.ids[i]), "prior": float(r.prior[i]),
                    "visits": int(r.visits[i]), "visit_probability": float(self.distribution[r.ids[i]]),
                    "q": float(r.q()[i])} for i in self.ranked_indices] if not r.terminal else []}


def search_batch(states, evaluator, config=None, *, rngs=None, engine=None):
    config = config or SearchConfig()
    engine = engine or RuleEngine.standard()
    roots = [Node(s, engine) for s in states]
    counts = [Counter(simulations=0, model_evaluations=0, terminal_leaves=0,
                      repetition_leaves=0, depth_limit_leaves=0, depth_sum=0, max_depth=0) for _ in roots]
    active = [(i, n) for i, n in enumerate(roots) if not n.terminal]
    predictions = evaluator([n for _, n in active]) if active else []
    if len(predictions) != len(active):
        raise ValueError("Evaluator batch size mismatch")
    for (i, node), (prior, value) in zip(active, predictions):
        node.expand(prior, value)
        counts[i]["model_evaluations"] += 1
        if config.noise_fraction:
            if rngs is None:
                raise ValueError("Self-play noise requires explicit per-game RNGs")
            noise = rngs[i].dirichlet(np.full(len(node.ids), config.dirichlet_alpha))
            node.prior = (1 - config.noise_fraction) * node.prior + config.noise_fraction * noise
    for _ in range(config.simulations):
        pending = []
        for i, root in active:
            node, path, seen = root, [], {root.key}
            while True:
                u = config.c_puct * node.prior * math.sqrt(1 + int(node.visits.sum())) / (1 + node.visits)
                index = int(np.argmax(node.q() + u))
                path.append((node, index))
                if node.children[index] is None:
                    node.children[index] = Node(engine.apply_action(node.state, node.actions[index]), engine)
                node = node.children[index]
                reason = "terminal_leaves" if node.terminal else "repetition_leaves" if node.key in seen else "depth_limit_leaves" if len(path) >= config.max_depth else None
                if reason:
                    counts[i][reason] += 1
                    backup(path, node.player, node.terminal_value if node.terminal else 0.)
                    break
                seen.add(node.key)
                if node.prior is None:
                    pending.append((i, node, path))
                    break
            counts[i]["simulations"] += 1
            counts[i]["depth_sum"] += len(path)
            counts[i]["max_depth"] = max(counts[i]["max_depth"], len(path))
        predictions = evaluator([n for _, n, _ in pending]) if pending else []
        if len(predictions) != len(pending):
            raise ValueError("Evaluator batch size mismatch")
        for (i, node, path), (prior, value) in zip(pending, predictions):
            node.expand(prior, value)
            counts[i]["model_evaluations"] += 1
            backup(path, node.player, value)
    return [SearchResult(n, dict(c)) for n, c in zip(roots, counts)]
