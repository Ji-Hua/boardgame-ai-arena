"""Controlled DAgger, PPO and feature-transfer DDQN experiments.

Uses the existing engine, CNN, teachers, DDQN update and checkpoint callback.
Every transition spans a learner action and the opponent reply, so rewards,
values and bootstraps always use the same player's perspective.
"""
from __future__ import annotations

import collections
from contextlib import contextmanager
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from quoridor_engine import Player, RuleEngine
from quoridor_engine import calculation

from agent_system.analysis.state import to_position, wire_action
from .action_space import ACTION_COUNT, decode_action_id, encode_engine_action
from .checkpoint import load_checkpoint, save_checkpoint
from .demonstrations import symmetry_key
from .distillation import policy_loss, reflect_batch
from .model import CNNQNetwork, policy_argmax
from .observation_cnn import encode_observation_cnn
from .opponent import GreedyOpponent
from .teacher_training import TeacherReplay, teacher_candidates
from .trainer import train_step


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def weights_hash(module):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                  for p in module.parameters())).hexdigest()


class ActorCritic(nn.Module):
    """The unchanged flat CNN actor plus a training-only scalar value head."""
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.value = nn.Linear(actor.dense_width, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs):
        hidden = self.actor.head[1](self.actor.head[0](self.actor.conv(obs).flatten(1)))
        return self.actor.head[2](hidden), self.value(hidden).squeeze(-1)


def masked_distribution(logits, legal):
    if not bool(legal.any(-1).all()):
        raise ValueError("A policy distribution requires at least one legal action")
    return torch.distributions.Categorical(logits=logits.masked_fill(~legal, -torch.inf))


def gae(rewards, values, terminated, bootstrap, gamma, lam):
    """One episode, including a possible time-limit truncation.

    True terminals zero the bootstrap; a time limit retains the critic value.
    Separate calls per episode prevent advantages leaking between games.
    """
    advantage = np.zeros(len(rewards), dtype=np.float32)
    carry, following = 0., float(bootstrap)
    for t in reversed(range(len(rewards))):
        continuation = 1. - float(terminated[t])
        delta = rewards[t] + gamma * continuation * following - values[t]
        carry = delta + gamma * lam * continuation * carry
        advantage[t] = carry
        following = values[t]
    return advantage, advantage + np.asarray(values, dtype=np.float32)


def ppo_objective(logits, value, batch, *, clip, value_weight, entropy_weight):
    distribution = masked_distribution(logits, batch["legal"])
    logp = distribution.log_prob(batch["action"])
    log_ratio = logp - batch["logp"]
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * batch["advantage"],
        ratio.clamp(1. - clip, 1. + clip) * batch["advantage"])
    actor_loss = -surrogate.mean()
    value_loss = .5 * (value - batch["return"]).square().mean()
    entropy = distribution.entropy().mean()
    loss = actor_loss + value_weight * value_loss - entropy_weight * entropy
    return loss, {"policy_loss": actor_loss.detach(), "value_loss": value_loss.detach(),
        "entropy": entropy.detach(), "approx_kl": ((ratio - 1) - log_ratio).mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > clip).float().mean().detach()}


class ComputeMeter:
    """CUDA event elapsed intervals, separated from teacher CPU and evaluation.

    Event elapsed time includes scheduling/contention, not exclusive GPU kernel
    accounting. CPU fallback is recorded only as wall time in test runs.
    """
    def __init__(self, device):
        self.device = device
        self.seconds = collections.Counter()

    @contextmanager
    def measure(self, category):
        cuda = str(self.device).startswith("cuda")
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if cuda:
                end.record()
                end.synchronize()
                self.seconds[category] += start.elapsed_time(end) / 1000


def observation(state):
    return np.asarray(encode_observation_cnn(state), dtype=np.float32)


def mask(engine, state):
    result = np.zeros(ACTION_COUNT, dtype=np.bool_)
    if not engine.is_game_over(state):
        result[[encode_engine_action(a) for a in engine.legal_actions(state)]] = True
    return result


def random_action(engine, state, rng):
    legal = engine.legal_actions(state)
    pawn = [a for a in legal if a.kind == "MovePawn"]
    walls = [a for a in legal if a.kind != "MovePawn"]
    return rng.choice(pawn if not walls or rng.random() < .7 else walls)


def shortest_path_features(engine, state, learner):
    """Authoritative Rust shortest-path features in one fixed learner view."""
    learner_distance = calculation.shortest_path_len(state, learner, engine.topology)
    opponent_distance = calculation.shortest_path_len(state, learner.opponent(), engine.topology)
    if learner_distance is None or opponent_distance is None:
        raise ValueError("Legal Quoridor state has no route to a goal")
    learner_distance, opponent_distance = float(learner_distance), float(opponent_distance)
    return {"self_distance": learner_distance, "opponent_distance": opponent_distance,
            "path_difference": opponent_distance - learner_distance}


def potential(features, *, alpha, beta, terminal=False):
    """Potential is zero on real terminals, as required for policy invariance."""
    if terminal:
        return 0.
    return -alpha * features["self_distance"] + beta * features["path_difference"]


def potential_shaping_reward(before, after, *, alpha, beta, gamma, terminal):
    """Return F(s,s') and its two potentials for a same-learner transition."""
    previous = potential(before, alpha=alpha, beta=beta)
    following = potential(after, alpha=alpha, beta=beta, terminal=terminal)
    return gamma * following - previous, previous, following


def discounted_shaping_sum(rewards, final_potential, gamma):
    """Numerical aid for tests and traces: sum gamma^t F_t plus residual."""
    return sum((gamma ** t) * reward for t, reward in enumerate(rewards)) + \
        (gamma ** len(rewards)) * final_potential


def collect_games(net, config, *, first_episode, count, device, meter,
                  opponent_pool=(), epsilon=0.):
    """Batched learner inference with canonical engine and existing Greedy.

    DAgger behavior is deterministic; a repeated exact state with a deterministic
    opponent is a draw. PPO is stochastic and never declares repeats terminal.
    All nonterminal caps close the opponent reply and retain their bootstrap.
    """
    engine, greedy = RuleEngine.standard(), GreedyOpponent()
    stats, visited, games = collections.Counter(), set(), []
    is_ppo = isinstance(net, ActorCritic)
    dagger = config["method"] == "dagger"
    shaping = config.get("reward_shaping", {})
    alpha, beta = float(shaping.get("alpha", 0.)), float(shaping.get("beta", 0.))
    if alpha < 0 or beta < 0 or not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("Reward shaping alpha and beta must be finite and non-negative")
    for key in ("terminal_reward_sum", "auxiliary_reward_sum", "auxiliary_reward_abs_sum",
                "auxiliary_reward_sq_sum", "auxiliary_reward_max_abs", "auxiliary_positive",
                "auxiliary_negative", "auxiliary_zero", "self_path_delta_sum",
                "path_difference_delta_sum", "pawn_actions", "wall_actions"):
        stats[key] = 0
    for index in range(first_episode, first_episode + count):
        rng = random.Random(config["rollout_seed"] + index * 104729)
        state = engine.initial_state()
        prefix = []
        for _ in range(rng.randrange(config["random_prefix_max"] + 1)):
            visited.add(symmetry_key(observation(state)))
            action = random_action(engine, state, rng)
            prefix.append(wire_action(action))
            state = engine.apply_action(state, action)
            if engine.is_game_over(state):
                raise ValueError("Training prefix reached a terminal state; choose a shorter prefix")
        stats["prefix_plies"] += len(prefix)
        games.append({"state": state, "seat": Player.P1 if index % 2 == 0 else Player.P2,
            "rng": rng, "episode": index + 1, "rows": [], "actions": [], "prefix": prefix,
            "opening": to_position(state), "plies": 0, "seen": set(), "finished": False,
            "opponent": (index // 2) % len(opponent_pool) if opponent_pool else None})

    def apply(game, action_id):
        state = game["state"]
        visited.add(symmetry_key(observation(state)))
        action = decode_action_id(int(action_id), state.current_player)
        game["actions"].append(wire_action(action))
        game["state"] = engine.apply_action(state, action)
        game["plies"] += 1
        stats["env_plies"] += 1

    def opponent_moves(active):
        groups = collections.defaultdict(list)
        for game in active:
            if not engine.is_game_over(game["state"]) and game["state"].current_player != game["seat"]:
                groups[game["opponent"]].append(game)
        for key, group in groups.items():
            if key is None:
                for game in group:
                    state = game["state"]
                    chosen = greedy.select_action_id(engine, state,
                        [encode_engine_action(a) for a in engine.legal_actions(state)], game["rng"])
                    apply(game, chosen)
            else:
                opponent = opponent_pool[key][1]
                with torch.no_grad(), meter.measure("opponent_inference_seconds"):
                    x = torch.from_numpy(np.stack([observation(g["state"]) for g in group])).to(device)
                    legal = torch.from_numpy(np.stack([mask(engine, g["state"]) for g in group])).to(device)
                    actions = masked_distribution(opponent(x), legal).sample().cpu().tolist()
                for game, chosen in zip(group, actions):
                    apply(game, chosen)

    opponent_moves(games)
    for game in games:
        if engine.is_game_over(game["state"]):
            # A valid near-goal prefix may let the opponent finish immediately.
            # Count the game; there is no learner decision to train on.
            game.update(finished=True, termination="goal", outcome=-1.)
            stats["completed_games"] += 1
            stats["learner_losses"] += 1
    while active := [g for g in games if not g["finished"]]:
        for game in active:
            if engine.is_game_over(game["state"]):
                raise ValueError("Opponent won before the first learner decision")
            assert game["state"].current_player == game["seat"]
        obs = np.stack([observation(g["state"]) for g in active])
        legal = np.stack([mask(engine, g["state"]) for g in active])
        with torch.no_grad(), meter.measure("learner_inference_seconds"):
            x, lm = torch.from_numpy(obs).to(device), torch.from_numpy(legal).to(device)
            if is_ppo:
                logits, value = net(x)
                distribution = masked_distribution(logits, lm)
                actions = distribution.sample()
                logps, values = distribution.log_prob(actions).cpu().tolist(), value.cpu().tolist()
            else:
                actions = policy_argmax(net(x), lm)
                logps, values = [0.] * len(active), [0.] * len(active)
            actions = actions.cpu().tolist()
        for i, game in enumerate(active):
            state = game["state"]
            game["seen"].add(observation(state).tobytes())
            chosen = actions[i]
            row = {"obs": obs[i], "legal": legal[i], "action": chosen,
                   "logp": logps[i], "value": values[i],
                   "action_kind": "pawn" if chosen < 81 else "wall",
                   "path_before": shortest_path_features(engine, state, game["seat"])}
            if dagger:
                before = time.monotonic()
                _, expert, context = teacher_candidates(state, 2)
                stats["teacher_seconds"] += time.monotonic() - before
                stats["minimax_search_calls"] += 1
                stats["teacher_nodes"] += context.nodes
                row["expert"] = expert
                stats["behavior_teacher_matches"] += chosen == expert
            elif not is_ppo and game["rng"].random() < epsilon:
                chosen = encode_engine_action(random_action(engine, state, game["rng"]))
                row["action"] = chosen
                stats["epsilon_actions"] += 1
            game["rows"].append(row)
            apply(game, chosen)
        opponent_moves(active)
        for game in active:
            state, row = game["state"], game["rows"][-1]
            terminal = engine.is_game_over(state)
            repeated = dagger and not terminal and observation(state).tobytes() in game["seen"]
            truncated = not terminal and not repeated and game["plies"] >= config["max_plies"]
            terminal_reward = (1. if engine.winner(state) == game["seat"] else -1.) if terminal else 0.
            path_after = shortest_path_features(engine, state, game["seat"])
            auxiliary_reward, phi_before, phi_after = potential_shaping_reward(
                row["path_before"], path_after, alpha=alpha, beta=beta, gamma=config["gamma"], terminal=terminal)
            reward = terminal_reward + auxiliary_reward
            row.update(reward=reward, terminal_reward=terminal_reward, auxiliary_reward=auxiliary_reward,
                potential_before=phi_before, potential_after=phi_after, path_after=path_after,
                done=terminal or repeated,
                next_obs=observation(state), next_mask=mask(engine, state))
            stats["terminal_reward_sum"] += terminal_reward
            stats["auxiliary_reward_sum"] += auxiliary_reward
            stats["auxiliary_reward_abs_sum"] += abs(auxiliary_reward)
            stats["auxiliary_reward_sq_sum"] += auxiliary_reward ** 2
            stats["auxiliary_reward_max_abs"] = max(stats["auxiliary_reward_max_abs"], abs(auxiliary_reward))
            stats["auxiliary_positive"] += auxiliary_reward > 0
            stats["auxiliary_negative"] += auxiliary_reward < 0
            stats["auxiliary_zero"] += auxiliary_reward == 0
            stats["self_path_delta_sum"] += path_after["self_distance"] - row["path_before"]["self_distance"]
            stats["path_difference_delta_sum"] += path_after["path_difference"] - row["path_before"]["path_difference"]
            stats[row["action_kind"] + "_actions"] += 1
            assert terminal or state.current_player == game["seat"]
            if terminal or repeated or truncated:
                game["finished"] = True
                game["termination"] = "goal" if terminal else "deterministic_cycle" if repeated else "time_limit"
                game["outcome"] = reward if terminal or repeated else None
                stats["completed_games" if terminal else "cycle_draws" if repeated else "truncated_games"] += 1
                stats["learner_wins"] += terminal and reward == 1
                stats["learner_losses"] += terminal and reward == -1
    if is_ppo:
        pending = [g for g in games if g["rows"] and not g["rows"][-1]["done"]]
        if pending:
            with torch.no_grad(), meter.measure("learner_inference_seconds"):
                x = torch.from_numpy(np.stack([g["rows"][-1]["next_obs"] for g in pending])).to(device)
                bootstrap = net(x)[1].cpu().tolist()
            for game, value in zip(pending, bootstrap):
                game["bootstrap"] = value
                # A time limit remains nonterminal: the critic's shaped value
                # carries the residual Phi(s_T), never a fabricated terminal 0.
                game["bootstrap_potential"] = game["rows"][-1]["potential_after"]
        for game in games:
            rows = game["rows"]
            advantage, returns = gae([r["reward"] for r in rows], [r["value"] for r in rows],
                [r["done"] for r in rows], game.get("bootstrap", 0.), config["gamma"], config["gae_lambda"])
            for row, adv, ret in zip(rows, advantage, returns):
                row["advantage"], row["return"] = adv, ret
    traces = [{"episode": g["episode"], "learner_seat": 1 if g["seat"] == Player.P1 else 2,
        "opponent": opponent_pool[g["opponent"]][0] if opponent_pool else "greedy",
        "prefix": g["prefix"], "opening": g["opening"], "actions": g["actions"],
        "termination": g["termination"], "outcome": g["outcome"],
        "shaping": {"terminal_sum": sum(r["terminal_reward"] for r in g["rows"]),
                    "auxiliary_sum": sum(r["auxiliary_reward"] for r in g["rows"]),
                    "auxiliary_abs_sum": sum(abs(r["auxiliary_reward"]) for r in g["rows"]),
                    "start": g["rows"][0]["path_before"] if g["rows"] else None,
                    "end": g["rows"][-1]["path_after"] if g["rows"] else None,
                    "final_potential": g["rows"][-1]["potential_after"] if g["rows"] else 0.,
                    "bootstrap_potential": g.get("bootstrap_potential", 0.)}} for g in games]
    rows = [row for game in games for row in game["rows"]]
    stats["episodes"], stats["learner_steps"] = len(games), len(rows)
    return rows, stats, traces, visited


def initialize(cfg, config):
    torch.manual_seed(cfg.seed)
    fresh = CNNQNetwork(cnn_channels=cfg.cnn_channels, activation=cfg.cnn_activation)
    initialization = config["initialization"]
    parent_path = Path(config["parent_checkpoint"])
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if digest != config["parent_sha256"]:
        raise ValueError("Frozen parent hash mismatch")
    parent = load_checkpoint(parent_path).network
    if (fresh.cnn_channels, fresh.dense_width, fresh.activation, fresh.hierarchical) != (
        parent.cnn_channels, parent.dense_width, parent.activation, parent.hierarchical):
        raise ValueError("Research CNN architecture must match the frozen parent")
    if initialization == "policy":
        fresh.load_state_dict(parent.state_dict())
    elif initialization == "conv_only":
        fresh.conv.load_state_dict(parent.conv.state_dict())
    elif initialization != "random":
        raise ValueError("Unknown initialization")
    info = {"mode": initialization, "parent_sha256": digest, "seed": cfg.seed,
        "initial_weights_sha256": weights_hash(fresh), "conv_sha256": weights_hash(fresh.conv),
        "head_sha256": weights_hash(fresh.head), "optimizer": "new; no inherited optimizer state",
        "actor_parameters": fresh.parameter_count()}
    return fresh.to(cfg.device), info


def tensors(rows, keys, device):
    result = {k: torch.from_numpy(np.asarray([r[k] for r in rows])).to(device) for k in keys}
    return {k: v.float() if v.dtype == torch.float64 else v for k, v in result.items()}


def sample_demo(data, size, device, generator, step):
    indices = torch.randint(len(data["obs"]), (size,), device=device, generator=generator)
    special = data.get("special")
    if special is not None and len(special):
        n = size // 4
        indices[:n] = special[torch.randint(len(special), (n,), device=device, generator=generator)]
    return reflect_batch(data["obs"][indices], data["legal"][indices], data["expert"][indices],
                         bool(step % 4 & 1), bool(step % 4 & 2))


def train_research(cfg, config, output, callback):
    method = config["method"]
    if method not in {"dagger", "ppo", "ppo_imitation", "selfplay_ppo", "double_dqn"}:
        raise ValueError("Unknown research method")
    if config["initialization"] == "conv_only" and method != "double_dqn":
        raise ValueError("Conv-only transfer is reserved for DDQN")
    if cfg.episodes % config["games_per_batch"]:
        raise ValueError("Episode budget must contain complete rollout batches")
    actor, initialization = initialize(cfg, config)
    ppo = method in {"ppo", "ppo_imitation", "selfplay_ppo"}
    net = ActorCritic(actor).to(cfg.device) if ppo else actor
    initialization["trainable_parameters"] = sum(p.numel() for p in net.parameters())
    write_json(output / "initialization.json", initialization)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.)
    generator = torch.Generator(device=cfg.device).manual_seed(config["batch_seed"])
    demo_generator = torch.Generator(device=cfg.device).manual_seed(config["batch_seed"] + 177)
    meter, totals = ComputeMeter(cfg.device), collections.Counter()
    for key in ("episodes", "env_plies", "prefix_plies", "learner_steps", "minimax_search_calls",
                "teacher_nodes", "teacher_seconds", "gradient_updates", "training_wall_seconds"):
        totals[key] = 0
    original = None
    if method in {"dagger", "ppo_imitation"}:
        dataset = torch.load(config["dataset"], map_location="cpu", weights_only=True)["train"]
        original = {"obs": dataset["obs"].to(cfg.device), "legal": dataset["legal"].to(cfg.device),
                    "expert": dataset["expert_d2"].to(cfg.device)}
        original["special"] = (dataset["winning"].any(1) |
            (dataset["expert_d1"] != dataset["expert_d2"])).nonzero().flatten().to(cfg.device)
    pool, pool_history = [], []
    if method == "selfplay_ppo":
        for row in config["opponents"]:
            path = Path(row["checkpoint"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Self-play opponent hash mismatch")
            opponent = load_checkpoint(path).network.to(cfg.device).eval()
            opponent.requires_grad_(False)
            pool.append((row["id"], opponent))
            pool_history.append(row)
    target = copy.deepcopy(actor).eval() if method == "double_dqn" else None
    replay = TeacherReplay(config.get("replay_capacity", 60000), cfg.device) if target else None
    aggregate, aggregate_keys, all_keys = [], set(), set()
    metrics = []
    with gzip.open(output / "training-games.jsonl.gz", "wt") as traces:
        for batch_number, first in enumerate(range(0, cfg.episodes, config["games_per_batch"]), 1):
            started = time.monotonic()
            fraction = first / max(1, cfg.episodes - config["games_per_batch"])
            epsilon = config.get("epsilon_start", .3) * (1 - fraction) + config.get("epsilon_end", .05) * fraction
            rows, counts, games, keys = collect_games(net, {**config, "gamma": cfg.gamma},
                first_episode=first, count=config["games_per_batch"], device=cfg.device,
                meter=meter, opponent_pool=pool, epsilon=epsilon)
            for key, value in counts.items():
                # A per-batch maximum is not additive across rollout batches.
                totals[key] = max(totals[key], value) if key == "auxiliary_reward_max_abs" else totals[key] + value
            all_keys.update(keys)
            for game in games:
                traces.write(json.dumps(game) + "\n")
            traces.flush()
            logs = collections.defaultdict(list)
            if method == "dagger":
                for row in rows:
                    key = symmetry_key(row["obs"])
                    if key not in aggregate_keys:
                        aggregate_keys.add(key)
                        aggregate.append({k: row[k] for k in ("obs", "legal", "expert")})
                aggregated = tensors(aggregate, ("obs", "legal", "expert"), cfg.device)
                with meter.measure("optimizer_seconds"):
                    for _ in range(config["updates_per_batch"]):
                        step = totals["gradient_updates"]
                        half = cfg.batch_size // 2
                        batches = [sample_demo(d, n, cfg.device, generator, step)
                            for d, n in ((original, half), (aggregated, cfg.batch_size - half))]
                        obs, legal, expert = [torch.cat([b[i] for b in batches]) for i in range(3)]
                        loss = policy_loss(actor(obs), legal, expert)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
                        if not torch.isfinite(norm):
                            raise RuntimeError("Non-finite DAgger gradient")
                        optimizer.step()
                        totals["gradient_updates"] += 1
                        logs["cross_entropy"].append(loss.item())
                logs["aggregate_positions"].append(len(aggregate))
            elif ppo:
                data = tensors(rows, ("obs", "legal", "action", "logp", "advantage", "return", "value"), cfg.device)
                advantage = data["advantage"]
                data["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                with meter.measure("optimizer_seconds"):
                    for epoch in range(config["epochs"]):
                        indices = torch.randperm(len(rows), device=cfg.device, generator=generator)
                        early_stop = False
                        for ix in indices.split(cfg.batch_size):
                            batch = {k: v[ix] for k, v in data.items()}
                            logits, value = net(batch["obs"])
                            loss, stats = ppo_objective(logits, value, batch, clip=config["clip"],
                                value_weight=config["value_weight"], entropy_weight=config["entropy_weight"])
                            # Stop BEFORE another update once the old-policy KL exceeds the cap.
                            if stats["approx_kl"].item() > config["target_kl"]:
                                totals["kl_early_stops"] += 1
                                early_stop = True
                                break
                            if method == "ppo_imitation":
                                obs, legal, expert = sample_demo(original, cfg.batch_size, cfg.device,
                                    demo_generator, totals["gradient_updates"])
                                ce = policy_loss(actor(obs), legal, expert)
                                loss = loss + config["imitation_weight"] * ce
                                logs["imitation_ce"].append(ce.item())
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            norm = nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
                            if not torch.isfinite(norm):
                                raise RuntimeError("Non-finite PPO gradient")
                            optimizer.step()
                            totals["gradient_updates"] += 1
                            for key, value in stats.items():
                                logs[key].append(float(value))
                        if early_stop:
                            break
                variance = data["return"].var(unbiased=False)
                logs["old_value_explained_variance"].append(float(1 -
                    (data["return"] - data["value"]).var(unbiased=False) / variance.clamp_min(1e-8)))
            else:
                replay.append([{k: r[k] for k in ("obs", "action", "reward", "next_obs", "done", "next_mask")} for r in rows])
                number = max(1, math.ceil(len(rows) * config["updates_per_transition"]))
                with meter.measure("optimizer_seconds"):
                    for _ in range(number):
                        batch = replay.sample(cfg.batch_size, generator)
                        batch["reward"] = batch["reward"].float()
                        result = train_step(actor, target, optimizer, batch, gamma=cfg.gamma,
                            grad_clip_norm=cfg.grad_clip_norm, algorithm="double_dqn")
                        if not math.isfinite(result.loss) or not math.isfinite(result.grad_norm):
                            raise RuntimeError("Non-finite DDQN update")
                        totals["gradient_updates"] += 1
                        if totals["gradient_updates"] % cfg.target_sync_interval == 0:
                            target.load_state_dict(actor.state_dict())
                            totals["target_syncs"] += 1
                        for key in ("loss", "q_max_abs", "mean_target_q"):
                            logs[key].append(getattr(result, key))
                logs["epsilon"].append(epsilon)
            totals["training_wall_seconds"] += time.monotonic() - started
            episode = totals["episodes"]
            metric = {"episode": episode, "batch": batch_number, "totals": dict(totals),
                "gpu_cuda_event_seconds": dict(meter.seconds), "rollout": dict(counts),
                **{key: sum(v) / len(v) for key, v in logs.items()}}
            metrics.append(metric)
            with (output / "research-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
            print(json.dumps(metric, allow_nan=False), flush=True)
            if episode in config["checkpoint_episodes"] or episode == cfg.episodes:
                path = output / "checkpoints" / f"ep{episode:05d}.pt"
                semantics = "action_values" if method == "double_dqn" else "policy_logits"
                save_checkpoint(path, actor, model_arch="cnn", agent_id=cfg.agent_id,
                    algorithm="policy_distillation" if method == "dagger" else method,
                    training_step=totals["gradient_updates"], episode_count=episode,
                    optimizer=None if ppo else optimizer, device=cfg.device,
                    reward_mode="potential_shaping" if any(config.get("reward_shaping", {}).values()) else "terminal",
                    opponent="frozen_history" if pool else "greedy",
                    eval_summary={"method": method, "output_semantics": semantics,
                        "parent_sha256": config["parent_sha256"], "initialization": initialization,
                        "resource_counts": dict(totals), "gpu_cuda_event_seconds": dict(meter.seconds),
                        "reward_shaping": config.get("reward_shaping", {"alpha": 0., "beta": 0.}),
                        "teacher_depth": 2 if method in {"dagger", "ppo_imitation"} else None,
                        "inference": "unchanged CNN legal masked argmax; no teacher, value head or search"})
                # The critic and actor-critic optimizer are training state, not serving models.
                training_state = output / "training-state"
                training_state.mkdir(exist_ok=True)
                torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "target": target.state_dict() if target else None,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if str(cfg.device).startswith("cuda") else [],
                    "batch_rng": generator.get_state(), "demo_rng": demo_generator.get_state(), "totals": dict(totals),
                    "pool": pool_history}, training_state / path.name)
                write_json(output / "training-state-keys.json", sorted(all_keys))
                # Checkpoint loading in evaluation constructs networks and consumes
                # CPU RNG; keep diagnosis from altering future training sampling.
                with torch.random.fork_rng(devices=[torch.device(cfg.device)] if str(cfg.device).startswith("cuda") else []):
                    callback(path, None)
                if pool and episode != cfg.episodes:
                    snapshot = copy.deepcopy(actor).eval().requires_grad_(False)
                    identifier = cfg.agent_id + "_" + path.stem
                    pool.append((identifier, snapshot))
                    pool_history.append({"id": identifier, "checkpoint": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "after_episode": episode})
                    write_json(output / "opponent-pool.json", pool_history)
    if aggregate:
        torch.save({"train": tensors(aggregate, ("obs", "legal", "expert"), "cpu"),
            "metadata": {"episodes": cfg.episodes, "teacher_depth": 2}}, output / "dagger-aggregate.pt")
    write_json(output / "opponent-pool.json", pool_history)
    write_json(output / "training-state-keys.json", sorted(all_keys))
    return {"method": method, "episodes": cfg.episodes, "updates": totals["gradient_updates"],
        "initialization": initialization, "totals": dict(totals),
        "gpu_cuda_event_seconds": dict(meter.seconds), "unique_visited_symmetry_groups": len(all_keys),
        "aggregate_positions": len(aggregate), "opponent_pool": pool_history,
        "output_semantics": "action_values" if method == "double_dqn" else "policy_logits",
        "reward_shaping": config.get("reward_shaping", {"alpha": 0., "beta": 0.})}
