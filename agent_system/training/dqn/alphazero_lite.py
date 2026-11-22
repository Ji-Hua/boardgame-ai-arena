"""Outcome calibration and small-scale MCTS policy/value learning.

No teacher labels, reward shaping or evaluation games enter the replay. The
value head is frozen-policy calibrated before ANY MCTS is used. Draw targets
are explicit experimental adjudications (threefold repetition / ply limit).
"""
from __future__ import annotations

from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from quoridor_engine import Player, RuleEngine

from agent_system.analysis.state import to_position, wire_action
from agent_system.runtime.search.puct import SearchConfig, TorchEvaluator, search_batch
from .action_space import decode_action_id, encode_engine_action
from .checkpoint import load_checkpoint, save_checkpoint
from .demonstrations import symmetry_key
from .distillation import reflect_batch, reflection_permutation
from .model import CNNQNetwork
from .opponent import GreedyOpponent
from .policy_research import ComputeMeter, observation, mask, write_json


def actor_hash(network):
    return hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes()
        for name, t in network.state_dict().items() if not name.startswith("value_head."))).hexdigest()


def prefix_action(engine, state, rng):
    """Sample with .7 pawn probability over nonterminal prefix actions.

    A random opening is a starting position, not a completed training game.
    Quoridor walls cannot end a game; a winning pawn destination can be filtered
    without repeatedly sampling or inventing a terminal state to train on.
    """
    goal = 8 if state.current_player == Player.P1 else 0
    actions = [a for a in engine.legal_actions(state) if not (a.kind == "MovePawn" and a.target_y == goal)]
    pawns = [a for a in actions if a.kind == "MovePawn"]
    walls = [a for a in actions if a.kind != "MovePawn"]
    if not actions:
        return None
    return rng.choice(pawns if pawns and (not walls or rng.random() < .7) else walls)


def initialize(parent_path, parent_sha256):
    if hashlib.sha256(Path(parent_path).read_bytes()).hexdigest() != parent_sha256:
        raise ValueError("Parent checkpoint checksum mismatch")
    parent = load_checkpoint(parent_path).network
    if not isinstance(parent, CNNQNetwork) or parent.hierarchical:
        raise ValueError("AlphaZero-lite requires the existing flat CNN policy")
    if parent.value_head is not None:
        return parent
    network = CNNQNetwork(in_channels=parent.in_channels, cnn_channels=parent.cnn_channels,
        action_count=parent.action_count, dense_width=parent.dense_width,
        activation=parent.activation, value_head=True)
    missing, unexpected = network.load_state_dict(parent.state_dict(), strict=False)
    if set(missing) != {"value_head.weight", "value_head.bias"} or unexpected:
        raise ValueError("Unexpected parent parameter mismatch")
    assert actor_hash(network) == actor_hash(parent)
    return network


def reflect_examples(obs, legal, pi, flip_x, flip_y):
    dummy = torch.zeros(len(obs), dtype=torch.long, device=obs.device)
    obs, legal, _ = reflect_batch(obs, legal, dummy, flip_x, flip_y)
    return obs, legal, pi[:, reflection_permutation(flip_x, flip_y, obs.device)]


def policy_value_loss(logits, values, legal, pi, outcomes, value_weight=1.):
    if not bool(legal.any(-1).all()) or not bool(torch.isfinite(pi).all()) or bool((pi < 0).any()):
        raise ValueError("Invalid legal mask or visit targets")
    if bool((pi.masked_select(~legal) != 0).any()) or not torch.allclose(pi.sum(-1), torch.ones_like(outcomes), atol=1e-5):
        raise ValueError("Visit targets must normalize over legal actions only")
    # Avoid 0 * -inf when illegal targets are zero.
    logp = logits.masked_fill(~legal, -torch.inf).log_softmax(-1).masked_fill(~legal, 0.)
    policy = -(pi * logp).sum(-1).mean()
    value = (values - outcomes).square().mean()
    return policy + value_weight * value, policy, value


def collect(network, config, first, count, device, meter, *, calibration=False):
    """Batched independent games; each recorded row has its own current player."""
    engine, greedy = RuleEngine.standard(), GreedyOpponent()
    games, keys, totals = [], set(), Counter()
    evaluator = TorchEvaluator(network, device, meter)
    search_config = SearchConfig(**config["search"])
    for index in range(first, first + count):
        seed = config["rollout_seed"] + index * 104729
        rng, nrng = random.Random(seed), np.random.default_rng(seed)
        state, prefix = engine.initial_state(), []
        for _ in range(rng.randrange(config["random_prefix_max"] + 1)):
            keys.add(symmetry_key(observation(state)))
            action = prefix_action(engine, state, rng)
            if action is None:
                break
            prefix.append(wire_action(action))
            state = engine.apply_action(state, action)
            if engine.is_game_over(state):
                raise ValueError("Random prefix reached a terminal state")
        totals["prefix_plies"] += len(prefix)
        games.append({"state": state, "episode": index + 1, "rng": rng, "nrng": nrng,
            "prefix": prefix, "opening": to_position(state), "rows": [], "actions": [],
            "seen": Counter({observation(state).tobytes(): 1}), "done": False,
            "greedy_seat": (Player.P1 if index // 2 % 2 == 0 else Player.P2) if calibration and index % 2 == 0 else None})
    while active := [g for g in games if not g["done"]]:
        obs = np.stack([observation(g["state"]) for g in active])
        legal = np.stack([mask(engine, g["state"]) for g in active])
        if calibration:
            with torch.inference_mode(), meter.measure("calibration_rollout_seconds"):
                logits = network(torch.from_numpy(obs).to(device))
                lm = torch.from_numpy(legal).to(device)
                probabilities = (logits / config.get("calibration_temperature", .8)).masked_fill(~lm, -torch.inf).softmax(-1).cpu().numpy()
            totals.update(model_evaluations=len(active), forward_batches=1)
            results = None
        else:
            results = search_batch([g["state"] for g in active], evaluator, search_config,
                                   rngs=[g["nrng"] for g in active], engine=engine)
        for i, game in enumerate(active):
            state = game["state"]
            keys.add(symmetry_key(obs[i]))
            if calibration:
                pi = probabilities[i].astype(np.float64)
                pi /= pi.sum()
                chosen = int(game["nrng"].choice(len(pi), p=pi))
                if game["greedy_seat"] == state.current_player:
                    chosen = greedy.select_action_id(engine, state, np.flatnonzero(legal[i]).tolist(), game["rng"])
            else:
                result = results[i]
                pi = result.distribution
                if len(game["actions"]) < config["temperature_plies"]:
                    sampling = pi.astype(np.float64)
                    chosen = int(game["nrng"].choice(len(pi), p=sampling / sampling.sum()))
                else:
                    chosen = result.action_id
                for key, value in result.stats.items():
                    if key == "max_depth":
                        totals["max_search_depth"] = max(totals["max_search_depth"], value)
                    elif key != "model_evaluations":
                        totals[key] += value
            row = {"obs": obs[i], "legal": legal[i], "pi": pi.astype(np.float32),
                   "player": state.current_player, "episode": game["episode"]}
            game["rows"].append(row)
            action = decode_action_id(chosen, state.current_player)
            game["actions"].append(wire_action(action))
            state = game["state"] = engine.apply_action(state, action)
            totals["env_plies"] += 1
            key = observation(state).tobytes()
            game["seen"][key] += 1
            terminal = engine.is_game_over(state)
            repeated = game["seen"][key] >= 3
            capped = len(game["actions"]) >= config["max_plies"]
            if terminal or repeated or capped:
                winner = engine.winner(state) if terminal else None
                game["done"] = True
                game["termination"] = "goal" if terminal else "threefold_repetition" if repeated else "ply_limit"
                game["winner"] = (1 if winner == Player.P1 else 2) if terminal else None
                for row in game["rows"]:
                    row["outcome"] = (1. if row["player"] == winner else -1.) if terminal else 0.
                totals[game["termination"]] += 1
    rows, traces = [], []
    for game in games:
        rows.extend(game["rows"])
        traces.append({k: game[k] for k in ("episode", "prefix", "opening", "actions", "termination", "winner")})
    totals.update(evaluator.stats)
    totals["games"] += len(games)
    totals["training_examples"] += len(rows)
    return rows, totals, traces, keys


def row_tensors(rows):
    return {key: torch.from_numpy(np.asarray([r[key] for r in rows], dtype=dtype))
            for key, dtype in (("obs", np.float32), ("legal", np.bool_), ("pi", np.float32),
                               ("outcome", np.float32), ("episode", np.int64))}


def value_metrics(pred, target, train_mean):
    return {"mse": float((pred - target).square().mean()), "zero_mse": float(target.square().mean()),
            "train_mean_mse": float((target - train_mean).square().mean()),
            "prediction_mean": float(pred.mean()), "target_mean": float(target.mean()),
            "sign_accuracy_nondraw": float(((pred[target != 0] > 0) == (target[target != 0] > 0)).float().mean()) if bool((target != 0).any()) else None}


def calibrate(network, data, config, device, meter):
    # Validation games are wholly withheld, not a random split of nearby states.
    validation = data["episode"] % 5 == 0
    if not bool(validation.any()) or not bool((~validation).any()):
        raise ValueError("Value calibration needs both training and validation games")
    before = actor_hash(network)
    for p in network.parameters():
        p.requires_grad_(False)
    for p in network.value_head.parameters():
        p.requires_grad_(True)
    x, targets = data["obs"], data["outcome"].to(device)
    features = []
    with torch.inference_mode(), meter.measure("calibration_features_seconds"):
        for fx, fy in ((False, False), (True, False), (False, True), (True, True)):
            blocks = []
            for start in range(0, len(x), 512):
                subset = slice(start, start + 512)
                obs, _, _ = reflect_examples(x[subset].to(device), data["legal"][subset].to(device), data["pi"][subset].to(device), fx, fy)
                hidden = network.head[1](network.head[0](network.conv(obs).flatten(1)))
                blocks.append(hidden.detach())
            features.append(torch.cat(blocks))
    # Clone outside inference_mode so linear autograd can save these constants.
    features = [v.clone() for v in features]
    train_ids = torch.where(~validation)[0].to(device)
    val_ids = torch.where(validation)[0].to(device)
    mean = float(targets[train_ids].mean())
    optimizer = torch.optim.Adam(network.value_head.parameters(), lr=config["calibration_lr"])
    best, selected_epoch, history = float("inf"), 0, []
    best_state = copy.deepcopy(network.value_head.state_dict())
    updates = 0
    for epoch in range(config["calibration_epochs"] + 1):
        if epoch:
            order = train_ids[torch.randperm(len(train_ids), device=device)]
            with meter.measure("calibration_update_seconds"):
                for offset in range(0, len(order), 512):
                    ids = order[offset:offset + 512]
                    hidden = features[(updates % 4)][ids]
                    pred = network.value_head(hidden).squeeze(-1).tanh()
                    loss = (pred - targets[ids]).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.value_head.parameters(), 1.)
                    optimizer.step()
                    updates += 1
        with torch.no_grad():
            prediction = network.value_head(features[0][val_ids]).squeeze(-1).tanh()
            metrics = value_metrics(prediction, targets[val_ids], mean)
        history.append({"epoch": epoch, **metrics})
        if metrics["mse"] < best:
            best, selected_epoch = metrics["mse"], epoch
            best_state = copy.deepcopy(network.value_head.state_dict())
    network.value_head.load_state_dict(best_state)
    fallback = None
    # A failed fit must not turn arbitrary predictions into search evidence.
    reference = min(history[0]["zero_mse"], history[0]["train_mean_mse"])
    if selected_epoch == 0 or best >= reference:
        fallback = "training_outcome_mean" if history[0]["train_mean_mse"] < history[0]["zero_mse"] else "zero"
        with torch.no_grad():
            network.value_head.weight.zero_()
            network.value_head.bias.fill_(float(np.arctanh(np.clip(mean, -.999, .999))) if fallback == "training_outcome_mean" else 0.)
    with torch.no_grad():
        final = value_metrics(network.value_head(features[0][val_ids]).squeeze(-1).tanh(), targets[val_ids], mean)
    assert actor_hash(network) == before, "Calibration modified the policy"
    for p in network.parameters():
        p.requires_grad_(True)
    return {"selected_epoch": selected_epoch, "fallback": fallback, "heldout": final,
        "train_examples": len(train_ids), "validation_examples": len(val_ids),
        "train_games": len(data["episode"][~validation].unique()),
        "validation_games": len(data["episode"][validation].unique()),
        "gradient_updates": updates, "actor_sha256": before, "history": history}


def train_alphazero(cfg, config, output, checkpoint_callback):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = random.Random(config["batch_seed"])
    device, meter = cfg.device, ComputeMeter(cfg.device)
    network = initialize(config["parent_checkpoint"], config["parent_sha256"]).to(device).eval()
    initial_actor = actor_hash(network)
    calibration_mode = config["phase"] == "calibration"
    if not calibration_mode:
        parent = load_checkpoint(config["parent_checkpoint"])
        if parent.eval_summary.get("value_head_status") not in {"calibrated", "trained"}:
            raise ValueError("Self-play must start from an outcome-calibrated value head")
    write_json(output / "initialization.json", {"parent_sha256": config["parent_sha256"],
        "actor_sha256": initial_actor, "value_head_zero_at_start": calibration_mode,
        "parameters": sum(p.numel() for p in network.parameters())})
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg.lr, weight_decay=config.get("weight_decay", 1e-4))
    totals, visited, replay, updates, saved = Counter(), set(), None, 0, []
    start = time.monotonic()
    diagnosis_seconds = 0.
    (output / "training-data").mkdir()
    metrics_path = output / "az-metrics.jsonl"
    calibration_rows = []

    def save(episode, calibration_report=None):
        nonlocal diagnosis_seconds
        path = output / "checkpoints" / f"ep{episode:05d}.pt"
        summary = {"output_semantics": "policy_logits", "value_semantics": "current_player_terminal_outcome",
            "value_head_status": "calibrated" if calibration_mode else "trained",
            "search": {**config["search"], "noise_fraction": 0.}, "parent_sha256": config["parent_sha256"],
            "initial_actor_sha256": initial_actor, "totals": dict(totals), "gradient_updates": updates,
            "value_calibration": calibration_report}
        save_checkpoint(path, network, model_arch="cnn", algorithm="value_calibration" if calibration_mode else "alphazero_lite",
            agent_id=cfg.agent_id, training_step=updates, episode_count=episode, optimizer=optimizer,
            device=device, reward_mode="terminal", eval_summary=summary)
        saved.append(str(path))
        before = time.monotonic()
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith("cuda") else []):
            checkpoint_callback(path, summary)
        diagnosis_seconds += time.monotonic() - before

    with gzip.open(output / "training-games.jsonl.gz", "wt") as games_file, metrics_path.open("w") as metrics_file:
        for first in range(0, cfg.episodes, config["games_per_batch"]):
            count = min(config["games_per_batch"], cfg.episodes - first)
            # Stop exactly at prespecified checkpoint boundaries.
            if any(first < cp < first + count for cp in config["checkpoint_episodes"]):
                raise ValueError("Checkpoint episodes must align with collection batches")
            rows, counters, traces, keys = collect(network, config, first, count, device, meter, calibration=calibration_mode)
            for key, value in counters.items():
                if key == "max_search_depth":
                    totals[key] = max(totals[key], value)
                else:
                    totals[key] += value
            visited.update(keys)
            for trace in traces:
                games_file.write(json.dumps(trace) + "\n")
            games_file.flush()
            batch = row_tensors(rows)
            torch.save(batch, output / "training-data" / f"games{first+1:05d}-{first+count:05d}.pt")
            policy_loss_mean, value_loss_mean = 0., 0.
            if calibration_mode:
                calibration_rows.append(batch)
            else:
                replay = batch if replay is None else {k: torch.cat((replay[k], v))[-config["replay_capacity"]:] for k, v in batch.items()}
                with meter.measure("joint_update_seconds"):
                    for _ in range(config["updates_per_batch"]):
                        ids = torch.randint(len(replay["obs"]), (cfg.batch_size,))
                        x, legal, pi = [replay[k][ids].to(device) for k in ("obs", "legal", "pi")]
                        x, legal, pi = reflect_examples(x, legal, pi, rng.random() < .5, rng.random() < .5)
                        target = replay["outcome"][ids].to(device)
                        logits, value = network.forward_policy_value(x)
                        loss, pl, vl = policy_value_loss(logits, value, legal, pi, target, config.get("value_weight", 1.))
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
                        optimizer.step()
                        policy_loss_mean += float(pl.detach()) / config["updates_per_batch"]
                        value_loss_mean += float(vl.detach()) / config["updates_per_batch"]
                        updates += 1
            episode = first + count
            report = None
            if calibration_mode and episode == cfg.episodes:
                data = {k: torch.cat([b[k] for b in calibration_rows]) for k in batch}
                report = calibrate(network, data, config, device, meter)
                updates = report["gradient_updates"]
                totals["calibration_gradient_updates"] = updates
                write_json(output / "calibration.json", report)
            row = {"episode": episode, "updates": updates, "totals": dict(totals),
                "policy_loss": policy_loss_mean, "value_loss": value_loss_mean,
                "replay_size": len(replay["obs"]) if replay is not None else 0,
                "gpu_cuda_event_seconds": dict(meter.seconds), "seconds": time.monotonic() - start - diagnosis_seconds}
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), flush=True)
            if episode in config["checkpoint_episodes"]:
                save(episode, report)
        write_json(output / "training-state-keys.json", sorted(visited))
    if not calibration_mode:
        torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(), "replay": replay,
            "updates": updates, "torch_rng": torch.random.get_rng_state(), "batch_rng": rng.getstate()}, output / "training-state.pt")
    return {"status": "complete", "episodes": cfg.episodes, "gradient_updates": updates, "totals": dict(totals),
        "checkpoints": saved, "actor_sha256": actor_hash(network), "initial_actor_sha256": initial_actor,
        "gpu_cuda_event_seconds": dict(meter.seconds), "training_seconds": time.monotonic() - start - diagnosis_seconds,
        "diagnosis_seconds": diagnosis_seconds, "total_wall_seconds": time.monotonic() - start}
