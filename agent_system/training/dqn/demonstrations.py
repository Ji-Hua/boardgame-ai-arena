"""CNN Double-DQN with Greedy demonstrations and Monte Carlo supervision.

This is a small demonstration-assisted experiment, not a full DQfD replica:
uniform replay, one-step DDQN, complete-game returns and a legal-action margin.
Greedy scores are action labels only; they are never used as numerical Q targets.
Inference remains the normal masked CNN argmax, with no teacher or win override.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from quoridor_engine import RuleEngine

from agent_system.runtime.service.agents.greedy_agent import score_position
from agent_system.analysis.state import to_position, wire_action
from .action_space import ACTION_COUNT, encode_engine_action
from .checkpoint import load_checkpoint, save_checkpoint
from .model import CNNQNetwork, network_outputs, policy_argmax
from .observation_cnn import CNN_OBSERVATION_VERSION, encode_observation_cnn
from .reward import RewardConfig, compute_reward_breakdown


def symmetry_key(obs):
    """Group reflected positions together, including the 8x8 wall-anchor grid."""
    keys = []
    for fx, fy in ((False, False), (True, False), (False, True), (True, True)):
        transformed = obs.copy()
        axes = tuple(i for i, yes in enumerate((fx, fy)) if yes)
        if axes:
            for c in (0, 1, 6):
                transformed[c] = np.flip(obs[c], axes)
            for c in (2, 3):
                transformed[c] = 0
                transformed[c, :8, :8] = np.flip(obs[c, :8, :8], axes)
        keys.append(transformed.tobytes())
    return hashlib.sha256(min(keys)).hexdigest()


def expert_margin(q, legal, expert, margin=.1):
    penalties = torch.full_like(q, margin)
    penalties.scatter_(1, expert[:, None], 0.)
    best = (q + penalties).masked_fill(~legal, -torch.inf).max(1).values
    return (best - q.gather(1, expert[:, None]).squeeze(1)).mean()


def demonstration_objectives(q, type_q, batch, margin=.1):
    taken = q.gather(1, batch["action"][:, None]).squeeze(1)
    mc = nn.functional.smooth_l1_loss(taken, batch["return"].float())
    if type_q is None:
        return mc, expert_margin(q, batch["legal"], batch["expert"], margin)
    expert_type = (batch["expert"] >= 81).long()
    executed_type = (batch["action"] >= 81).long()
    type_taken = type_q.gather(1, executed_type[:, None]).squeeze(1)
    mc = .5 * (mc + nn.functional.smooth_l1_loss(type_taken, batch["return"].float()))
    available = torch.stack((batch["legal"][:, :81].any(1), batch["legal"][:, 81:].any(1)), 1)
    group = (torch.arange(q.shape[-1], device=q.device) >= 81).long()
    conditional = batch["legal"] & (group == expert_type[:, None])
    expert_loss = .5 * (expert_margin(q, conditional, batch["expert"], margin) +
                       expert_margin(type_q, available, expert_type, margin))
    return mc, expert_loss


def collect_demonstrations(config, reward_config, gamma, folder):
    engine, rng = RuleEngine.standard(), random.Random(config["data_seed"])
    rows, traces = [], []
    completed = truncated = 0
    started = time.monotonic()
    for episode in range(config["episodes"]):
        state = engine.initial_state()
        prefix = rng.randrange(config.get("random_prefix_max", 16) + 1)
        trajectory, actions = [], []
        for ply in range(config.get("max_plies", 160)):
            candidates = score_position(state)
            winning = [c for c in candidates if engine.is_game_over(c["state"])]
            teacher = winning[0] if winning else max(candidates, key=lambda c: c["score"])
            chosen = teacher
            if ply < prefix or rng.random() < config.get("exploration", .15):
                pawn = [c for c in candidates if c["action"].kind == "MovePawn"]
                walls = [c for c in candidates if c["action"].kind != "MovePawn"]
                group = pawn if not walls or rng.random() < .7 else walls
                chosen = rng.choice(group)
            mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
            mask[[encode_engine_action(c["action"]) for c in candidates]] = True
            trajectory.append((state, encode_engine_action(chosen["action"]),
                               encode_engine_action(teacher["action"]), mask))
            actions.append(wire_action(chosen["action"]))
            state = chosen["state"]
            if engine.is_game_over(state):
                break
        # Monte Carlo targets require actual completed games. Never pretend
        # a time limit is a loss/draw; count and exclude incomplete trajectories.
        if not engine.is_game_over(state):
            truncated += 1
            continue
        completed += 1
        returns, game_rows = {}, []
        for i in range(len(trajectory)-1, -1, -1):
            before, action, teacher, mask = trajectory[i]
            terminal = i + 2 >= len(trajectory)
            after = state if terminal else trajectory[i+2][0]
            if not terminal:
                assert after.current_player == before.current_player
            term_reward = (1. if engine.winner(state) == before.current_player else -1.) if terminal else 0.
            reward = compute_reward_breakdown(engine, before, after, before.current_player,
                                              term_reward, reward_config).combined_reward
            returns[i] = reward + (0. if terminal else gamma * returns[i+2])
            obs = np.asarray(encode_observation_cnn(before), dtype=np.float32)
            next_obs = np.asarray(encode_observation_cnn(after), dtype=np.float32)
            next_mask = np.zeros(ACTION_COUNT, dtype=np.bool_) if terminal else trajectory[i+2][3]
            game_rows.append({"obs": obs, "next_obs": next_obs, "action": action,
                "expert": teacher, "legal": mask, "next_mask": next_mask,
                "reward": reward, "done": terminal, "return": returns[i],
                "episode": episode, "key": symmetry_key(obs)})
        rows.extend(reversed(game_rows))
        traces.append({"episode": episode, "start": to_position(engine.initial_state()), "actions": actions})
        if (episode+1) % 100 == 0:
            print(f"[demonstrations] {episode+1}/{config['episodes']} games, {len(rows)} transitions", flush=True)
    # Episode split, then remove any validation state (or reflection) that
    # occurs in training. Common openings must not inflate validation accuracy.
    train_keys = {r["key"] for r in rows if r["episode"] % 10 != 0}
    train_rows = [r for r in rows if r["episode"] % 10 != 0]
    valid_rows = [r for r in rows if r["episode"] % 10 == 0 and r["key"] not in train_keys]
    if not train_rows or not valid_rows:
        raise ValueError("Demonstration split has no training or independent validation states")
    keys = ("obs", "next_obs", "action", "expert", "legal", "next_mask", "reward", "done", "return")
    def tensors(items):
        return {k: torch.from_numpy(np.asarray([r[k] for r in items])) for k in keys}
    dataset = {"train": tensors(train_rows), "validation": tensors(valid_rows),
               "metadata": {"completed_games": completed, "truncated_games_excluded": truncated,
                   "training_transitions": len(train_rows), "validation_transitions": len(valid_rows),
                   "training_unique_symmetry_groups": len(train_keys), "generation_seconds": time.monotonic()-started,
                   "config": config, "gamma": gamma, "validation": "held-out episodes; no shared reflection group"}}
    torch.save(dataset, folder / "demonstrations.pt")
    with gzip.open(folder / "demonstration-games.json.gz", "wt") as stream:
        json.dump(traces, stream)
    (folder / "dataset.json").write_text(json.dumps(dataset["metadata"], indent=2))
    return dataset


def train_demonstrations(cfg, config, output, callback):
    torch.manual_seed(cfg.seed)
    reward_config = RewardConfig(mode=cfg.reward_mode, distance_reward_weight=cfg.distance_reward_weight,
                                 distance_delta_clip=cfg.distance_delta_clip)
    if config.get("dataset"):
        dataset = torch.load(config["dataset"], map_location="cpu", weights_only=True)
        if dataset["metadata"]["gamma"] != cfg.gamma:
            raise ValueError("Demonstration gamma mismatch")
        (output / "dataset.json").write_text(json.dumps({**dataset["metadata"], "source_dataset": config["dataset"]}, indent=2))
    else:
        dataset = collect_demonstrations(config, reward_config, cfg.gamma, output)
    train = {k: v.to(cfg.device) for k, v in dataset["train"].items()}
    valid = {k: v.to(cfg.device) for k, v in dataset["validation"].items()}
    net = CNNQNetwork(cnn_channels=cfg.cnn_channels, activation=cfg.cnn_activation, hierarchical=cfg.cnn_hierarchical).to(cfg.device)
    target = CNNQNetwork(cnn_channels=cfg.cnn_channels, activation=cfg.cnn_activation, hierarchical=cfg.cnn_hierarchical).to(cfg.device)
    initialization = {"mode": "fresh", "seed": cfg.seed}
    if cfg.warm_start_checkpoint:
        parent = load_checkpoint(cfg.warm_start_checkpoint, expected_obs_version=CNN_OBSERVATION_VERSION)
        if parent.model_arch != "cnn":
            raise ValueError("Demonstration warm start requires a CNN checkpoint")
        # Preserve all parent weights, while allowing the configured nonlinearity
        # to change. Optimizer and update counters intentionally start fresh.
        net.load_state_dict(parent.network.state_dict(), strict=True)
        with Path(cfg.warm_start_checkpoint).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        initialization = {"mode": "weights_only", "seed": cfg.seed,
            "parent_checkpoint": str(Path(cfg.warm_start_checkpoint).resolve()),
            "parent_sha256": digest, "parent_checkpoint_id": parent.checkpoint_id,
            "parent_episode_count": parent.episode_count, "parent_training_step": parent.training_step,
            "parent_activation": parent.network.activation, "activation": cfg.cnn_activation}
    (output / "initialization.json").write_text(json.dumps(initialization, indent=2))
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    batch_generator = None
    if "batch_seed" in config:
        batch_generator = torch.Generator(device=cfg.device).manual_seed(config["batch_seed"])
    metrics = []
    for step in range(1, config["updates"]+1):
        indices = torch.randint(len(train["obs"]), (cfg.batch_size,), device=cfg.device, generator=batch_generator)
        batch = {k: v[indices] for k, v in train.items()}
        q, type_q = network_outputs(net, batch["obs"])
        taken = q.gather(1, batch["action"][:, None]).squeeze(1)
        with torch.no_grad():
            next_q_online, next_type_q = network_outputs(net, batch["next_obs"])
            next_action = policy_argmax(next_q_online, batch["next_mask"], next_type_q)
            next_q = target(batch["next_obs"]).gather(1, next_action[:, None]).squeeze(1)
            td_target = batch["reward"].float() + cfg.gamma * torch.where(batch["done"], 0., next_q)
        td = nn.functional.smooth_l1_loss(taken, td_target)
        if type_q is not None:
            type_taken = type_q.gather(1, (batch["action"] >= 81).long()[:, None]).squeeze(1)
            td = .5 * (td + nn.functional.smooth_l1_loss(type_taken, td_target))
        mc, margin = demonstration_objectives(q, type_q, batch, config.get("margin", .1))
        loss = td + mc + config.get("margin_weight", 1.) * margin
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm or 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite demonstration training gradient")
        optimizer.step()
        if step % cfg.target_sync_interval == 0:
            target.load_state_dict(net.state_dict())
        if step % 200 == 0:
            with torch.no_grad():
                vi = torch.arange(min(2048, len(valid["obs"])), device=cfg.device)
                vq, vt = network_outputs(net, valid["obs"][vi])
                accuracy = (policy_argmax(vq, valid["legal"][vi], vt) == valid["expert"][vi]).float().mean().item()
            row = {"update": step, "loss": loss.item(), "td_loss": td.item(), "mc_loss": mc.item(),
                   "expert_margin": margin.item(), "validation_expert_accuracy": accuracy,
                   "chosen_q_abs_max": taken.abs().max().item()}
            metrics.append(row)
            with (output / "demo-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % config["checkpoint_updates"] == 0 or step == config["updates"]:
            path = output / "checkpoints" / f"update{step:06d}.pt"
            save_checkpoint(path, net, model_arch="cnn", agent_id=cfg.agent_id,
                algorithm="double_dqn", training_step=step, episode_count=dataset["metadata"]["completed_games"],
                optimizer=optimizer, device=cfg.device, reward_mode=cfg.reward_mode,
                distance_reward_weight=cfg.distance_reward_weight, distance_delta_clip=cfg.distance_delta_clip,
                opponent="greedy_demonstrations", eval_summary={"training_objective": "DDQN + MC return + expert margin",
                    "teacher": "Greedy with immediate-win labels", "inference": "CNN masked argmax only",
                    "initialization": initialization})
            callback(path, None)
    return {"dataset": dataset["metadata"], "initialization": initialization,
            "updates": config["updates"], "metrics": metrics}
