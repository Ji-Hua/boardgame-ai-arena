"""Episode-counted CNN learning with online Minimax demonstrations.

One episode is a game from the standard board, with alternating learner seats.
The teacher labels learner decision states; a decaying behavior mixture visits
both teacher and CNN states. Only the CNN is saved and used for evaluation.
Targets combine same-player DDQN, complete-game returns and expert margins.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch import nn
from quoridor_engine import Player, RuleEngine

from agent_system.analysis.state import to_position, wire_action
from agent_system.runtime.service.agents.minimax_agent import score_position
from .action_space import ACTION_COUNT, decode_action_id, encode_engine_action
from .checkpoint import save_checkpoint
from .demonstrations import expert_margin, symmetry_key
from .model import CNNQNetwork, policy_argmax
from .observation_cnn import encode_observation_cnn
from .opponent import build_opponent
from .reward import RewardConfig, compute_reward_breakdown


def teacher_candidates(state, depth):
    """Use serving search and its first-maximum tie break, including all -inf.

    Search bounds and +/- infinity are never numerical learning targets.
    """
    candidates, context = score_position(state, depth)
    best = max(candidates, key=lambda c: c["score"])
    return candidates, encode_engine_action(best["action"]), context


def _observation(state):
    return np.asarray(encode_observation_cnn(state), dtype=np.float32)


def _mask(actions):
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    mask[[encode_engine_action(a) for a in actions]] = True
    return mask


def collect_teacher_episode(engine, net, opponent, rng, *, episode, depth,
                            max_plies, teacher_probability, epsilon,
                            gamma, reward_config, device):
    """Close the opponent reply even at a cap; never assign fake MC returns."""
    learner = Player.P1 if episode % 2 == 0 else Player.P2
    state = engine.initial_state()
    rows, actions, visited = [], [], set()
    stats = collections.Counter()

    def apply(action_id):
        nonlocal state
        visited.add(symmetry_key(_observation(state)))
        action = decode_action_id(action_id, state.current_player)
        actions.append(wire_action(action))
        state = engine.apply_action(state, action)
        stats["env_plies"] += 1

    def opponent_move():
        legal = [encode_engine_action(a) for a in engine.legal_actions(state)]
        apply(opponent.select_action_id(engine, state, legal, rng))

    if state.current_player != learner:
        opponent_move()
    while not engine.is_game_over(state) and stats["env_plies"] < max_plies:
        before = state
        started = time.monotonic()
        candidates, expert, search = teacher_candidates(state, depth)
        stats["teacher_seconds"] += time.monotonic() - started
        stats["teacher_queries"] += 1
        stats["teacher_nodes"] += search.nodes
        legal = _mask([c["action"] for c in candidates])
        obs = _observation(state)
        if rng.random() < epsilon:
            pawn = np.flatnonzero(legal[:81]).tolist()
            walls = (np.flatnonzero(legal[81:]) + 81).tolist()
            group = pawn if not walls or (pawn and rng.random() < .7) else walls
            chosen = rng.choice(group)
            stats["random_actions"] += 1
        elif rng.random() < teacher_probability:
            chosen = expert
            stats["teacher_actions"] += 1
        else:
            with torch.no_grad():
                q = net(torch.from_numpy(obs).to(device))
                chosen = int(policy_argmax(q, torch.from_numpy(legal).to(device)))
            stats["cnn_actions"] += 1
            stats["cnn_teacher_agreement"] += chosen == expert
        stats["executed_expert"] += chosen == expert
        apply(chosen)
        if not engine.is_game_over(state):
            opponent_move()
        terminal = engine.is_game_over(state)
        assert terminal or state.current_player == learner
        reward = compute_reward_breakdown(engine, before, state, learner,
            (1. if engine.winner(state) == learner else -1.) if terminal else 0.,
            reward_config).combined_reward
        rows.append({"obs": obs, "next_obs": _observation(state), "legal": legal,
            "next_mask": np.zeros(ACTION_COUNT, dtype=np.bool_) if terminal else _mask(engine.legal_actions(state)),
            "action": chosen, "expert": expert, "reward": reward, "done": terminal,
            "return": 0., "mc_valid": False})
    completed = engine.is_game_over(state)
    if completed:
        value = 0.
        for row in reversed(rows):
            value = row["reward"] + (0. if row["done"] else gamma * value)
            row["return"], row["mc_valid"] = value, True
    stats["completed_games"] = int(completed)
    stats["truncated_games"] = int(not completed)
    stats["learner_wins"] = int(completed and engine.winner(state) == learner)
    visited.add(symmetry_key(_observation(state)))
    trace = {"episode": episode + 1, "learner_seat": 1 if learner == Player.P1 else 2,
        "start": to_position(engine.initial_state()), "actions": actions,
        "completed": completed, "teacher_probability": teacher_probability, "epsilon": epsilon}
    return rows, stats, trace, visited


class TeacherReplay:
    """A bounded tensor replay; diagnostics never participate in this buffer."""
    def __init__(self, capacity, device):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.device = capacity, device
        self.tensors, self.size, self.cursor = {}, 0, 0

    def append(self, rows):
        for offset in range(0, len(rows), self.capacity):
            part = rows[offset:offset + self.capacity]
            values = {k: torch.from_numpy(np.asarray([r[k] for r in part])).to(self.device)
                      for k in part[0]}
            if not self.tensors:
                self.tensors = {k: torch.empty((self.capacity, *v.shape[1:]), dtype=v.dtype,
                                              device=self.device) for k, v in values.items()}
            indices = (torch.arange(len(part), device=self.device) + self.cursor) % self.capacity
            for k, value in values.items():
                self.tensors[k][indices] = value
            self.size = min(self.capacity, self.size + len(part))
            self.cursor = (self.cursor + len(part)) % self.capacity

    def sample(self, size, generator):
        indices = torch.randint(self.size, (size,), device=self.device, generator=generator)
        return {k: v[indices] for k, v in self.tensors.items()}


def teacher_update(net, target, optimizer, batch, *, gamma, margin,
                   margin_weight, grad_clip_norm):
    q = net(batch["obs"])
    taken = q.gather(1, batch["action"][:, None]).squeeze(1)
    with torch.no_grad():
        next_action = policy_argmax(net(batch["next_obs"]), batch["next_mask"])
        next_value = target(batch["next_obs"]).gather(1, next_action[:, None]).squeeze(1)
        td_target = batch["reward"].float() + gamma * torch.where(batch["done"], 0., next_value)
    td = nn.functional.smooth_l1_loss(taken, td_target)
    mc_terms = nn.functional.smooth_l1_loss(taken, batch["return"].float(), reduction="none")
    mc = (mc_terms * batch["mc_valid"]).sum() / batch["mc_valid"].sum().clamp_min(1)
    imitation = expert_margin(q, batch["legal"], batch["expert"], margin)
    loss = td + mc + margin_weight * imitation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
    if not torch.isfinite(norm):
        raise RuntimeError("Non-finite teacher training gradient")
    optimizer.step()
    return {"loss": loss.item(), "td_loss": td.item(), "mc_loss": mc.item(),
            "expert_margin": imitation.item(), "chosen_q_abs_max": taken.detach().abs().max().item()}


def train_teacher(cfg, config, output, callback):
    """Fresh training only; checkpoints are counted by games, not updates."""
    if cfg.warm_start_checkpoint or cfg.demonstration_dataset or cfg.cnn_hierarchical:
        raise ValueError("Teacher episode experiment requires a fresh flat CNN")
    depth = config["depth"]
    if depth not in (1, 2) or cfg.episodes < 1 or cfg.checkpoint_interval < 1:
        raise ValueError("Specify depth 1/2 and positive episode/checkpoint budgets")
    if cfg.buffer_capacity < cfg.warmup_size:
        raise ValueError("Replay must hold the warmup transitions")
    if cfg.max_steps_per_episode < 1 or cfg.log_interval < 1 or cfg.target_sync_interval < 1:
        raise ValueError("Ply, log and target-sync intervals must be positive")
    for probability in (config["teacher_probability_start"], config["teacher_probability_end"],
                        cfg.epsilon_start, cfg.epsilon_end):
        if not 0 <= probability <= 1:
            raise ValueError("Behavior probabilities must be in [0, 1]")
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    generator = torch.Generator(device=cfg.device).manual_seed(config["batch_seed"])
    net = CNNQNetwork(cnn_channels=cfg.cnn_channels, activation=cfg.cnn_activation).to(cfg.device)
    target = CNNQNetwork(cnn_channels=cfg.cnn_channels, activation=cfg.cnn_activation).to(cfg.device)
    target.load_state_dict(net.state_dict())
    digest = hashlib.sha256()
    for key, value in sorted(net.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    initialization = {"mode": "fresh", "seed": cfg.seed, "initial_weights_sha256": digest.hexdigest(),
                      "parent_checkpoint": None}
    (output / "initialization.json").write_text(json.dumps(initialization, indent=2))
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    replay = TeacherReplay(cfg.buffer_capacity, cfg.device)
    engine, opponent = RuleEngine.standard(), build_opponent("greedy")
    reward = RewardConfig(mode=cfg.reward_mode, distance_reward_weight=cfg.distance_reward_weight,
                          distance_delta_clip=cfg.distance_delta_clip)
    totals, visited = collections.Counter(), set()
    updates, metrics, checkpoints = 0, [], []
    started = time.monotonic()
    with gzip.open(output / "training-games.jsonl.gz", "wt") as traces:
        for episode in range(cfg.episodes):
            fraction = episode / max(1, cfg.episodes - 1)
            beta = config["teacher_probability_start"] + fraction * (config["teacher_probability_end"] - config["teacher_probability_start"])
            epsilon = cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)
            rows, stats, trace, keys = collect_teacher_episode(engine, net, opponent, rng,
                episode=episode, depth=depth, max_plies=cfg.max_steps_per_episode,
                teacher_probability=beta, epsilon=epsilon, gamma=cfg.gamma,
                reward_config=reward, device=cfg.device)
            replay.append(rows)
            totals.update(stats)
            visited.update(keys)
            traces.write(json.dumps(trace) + "\n")
            totals["learner_transitions"] += len(rows)
            if replay.size >= cfg.warmup_size:
                for _ in rows:  # One update per newly observed learner transition.
                    last = teacher_update(net, target, optimizer, replay.sample(cfg.batch_size, generator),
                        gamma=cfg.gamma, margin=config["margin"], margin_weight=config["margin_weight"],
                        grad_clip_norm=cfg.grad_clip_norm or 1.)
                    updates += 1
                    if updates % cfg.target_sync_interval == 0:
                        target.load_state_dict(net.state_dict())
            if (episode + 1) % cfg.log_interval == 0 or episode + 1 == cfg.episodes:
                metric = {"episode": episode + 1, "updates": updates, "replay_size": replay.size,
                    "teacher_probability": beta, "epsilon": epsilon, "totals": dict(totals),
                    "elapsed_seconds": time.monotonic() - started, **(last if updates else {})}
                metrics.append(metric)
                with (output / "teacher-metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)
            if (episode + 1) % cfg.checkpoint_interval == 0 or episode + 1 == cfg.episodes:
                path = output / "checkpoints" / f"ep{episode+1:05d}.pt"
                summary = {"training_objective": "same-player DDQN + complete-game MC + expert margin",
                    "teacher": f"minimax_d{depth}", "teacher_depth": depth,
                    "inference": "CNN masked argmax only", "initialization": initialization,
                    "episode_semantics": "one standard-board game; alternating learner seat; caps counted",
                    "training_counters": dict(totals), "optimizer_updates": updates,
                    "teacher_probability": beta, "epsilon": epsilon}
                save_checkpoint(path, net, model_arch="cnn", agent_id=cfg.agent_id,
                    algorithm="double_dqn", training_step=updates, episode_count=episode+1,
                    optimizer=optimizer, device=cfg.device, reward_mode=cfg.reward_mode,
                    distance_reward_weight=cfg.distance_reward_weight, distance_delta_clip=cfg.distance_delta_clip,
                    opponent="greedy", eval_summary=summary)
                checkpoints.append(str(path))
                callback(path, None)
    # All visited positions, including truncated games and states evicted from
    # replay, remain excluded from the later held-out opening book.
    (output / "training-state-keys.json").write_text(json.dumps(sorted(visited)))
    torch.save({"train": {k: v[:replay.size].cpu() for k, v in replay.tensors.items()},
                "metadata": {"gamma": cfg.gamma, "teacher_depth": depth, "episodes": cfg.episodes,
                             "replay_size": replay.size}}, output / "teacher-replay.pt")
    return {"episodes": cfg.episodes, "updates": updates, "totals": dict(totals),
            "initialization": initialization, "teacher_depth": depth, "metrics": metrics,
            "checkpoints": checkpoints, "training_unique_symmetry_groups": len(visited)}
