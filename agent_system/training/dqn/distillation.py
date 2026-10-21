"""Flat CNN policy distillation on shared, diverse Minimax trajectories.

This is supervised policy learning, not Q-learning. Outputs are logits; neither
alpha-beta bounds nor game returns are numerical targets. Runtime still uses
only the CNN and the engine's legal-action mask.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import collections
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

from agent_system.analysis.state import to_position, wire_action
from agent_system.runtime.service.agents.minimax_agent import score_position
from .action_space import ACTION_COUNT, encode_engine_action
from .checkpoint import save_checkpoint
from .demonstrations import symmetry_key
from .model import CNNQNetwork
from .observation_cnn import encode_observation_cnn


def reflection_permutation(flip_x, flip_y, device="cpu"):
    """Map old action indices to reflected ones (9x9 pawns, 8x8 walls)."""
    ids = []
    for offset, size in ((0, 9), (81, 8), (145, 8)):
        for x in range(size):
            for y in range(size):
                nx, ny = size - 1 - x if flip_x else x, size - 1 - y if flip_y else y
                ids.append(offset + nx * size + ny)
    return torch.tensor(ids, dtype=torch.long, device=device)


def reflect_batch(obs, legal, expert, flip_x, flip_y):
    axes = tuple(i + 1 for i, yes in enumerate((flip_x, flip_y)) if yes)
    if not axes:
        return obs, legal, expert
    result = obs.clone()
    for channel in (0, 1, 6):
        result[:, channel] = obs[:, channel].flip(axes)
    for channel in (2, 3):
        result[:, channel] = 0
        result[:, channel, :8, :8] = obs[:, channel, :8, :8].flip(axes)
    permutation = reflection_permutation(flip_x, flip_y, obs.device)
    # Reflections are involutions, so this gather is also the inverse map.
    return result, legal[:, permutation], permutation[expert]


def _collect_game(job):
    episode, config = job
    engine = RuleEngine.standard()
    rng = random.Random(config["seed"] + episode * 104729)
    state = engine.initial_state()
    prefix = rng.randrange(config["random_prefix_max"] + 1)
    behavior_depth = 1 + episode % 2
    rows, actions = [], []
    for ply in range(config["max_plies"]):
        d1, _ = score_position(state, 1)
        d2, _ = score_position(state, 2)
        teachers = [max(candidates, key=lambda c: c["score"]) for candidates in (d1, d2)]
        chosen = teachers[behavior_depth - 1]
        if ply < prefix or rng.random() < config["exploration"]:
            pawns = [c for c in d1 if c["action"].kind == "MovePawn"]
            walls = [c for c in d1 if c["action"].kind != "MovePawn"]
            group = pawns if not walls or rng.random() < config["random_pawn_probability"] else walls
            if ply < prefix and group is pawns:
                x, _ = state.pawn_pos(state.current_player)
                lateral = [c for c in pawns if c["action"].target_x != x]
                if lateral and rng.random() < .6:
                    group = lateral
            chosen = rng.choice(group)
        legal = np.zeros(ACTION_COUNT, dtype=np.bool_)
        wins = np.zeros(ACTION_COUNT, dtype=np.bool_)
        for candidate in d1:
            index = encode_engine_action(candidate["action"])
            legal[index] = True
            wins[index] = engine.winner(candidate["state"]) == state.current_player
        obs = np.asarray(encode_observation_cnn(state), dtype=np.float32)
        rows.append({"obs": obs, "legal": legal, "winning": wins,
                     "expert_d1": encode_engine_action(teachers[0]["action"]),
                     "expert_d2": encode_engine_action(teachers[1]["action"])})
        actions.append(wire_action(chosen["action"]))
        state = chosen["state"]
        if engine.is_game_over(state):
            break
    return rows, {"episode": episode, "behavior_depth": behavior_depth,
        "random_prefix": prefix, "actions": actions, "final_position": to_position(state),
        "completed": engine.is_game_over(state)}


def collect_shared_dataset(config, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, seen, counts, files, endings = [], set(), collections.Counter(), collections.Counter(), collections.Counter()
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool, gzip.open(folder / "games.jsonl.gz", "wt") as traces:
        # Ordered map plus per-episode RNG makes data independent of scheduling.
        for episode, (game_rows, trace) in enumerate(pool.map(_collect_game,
                ((i, config) for i in range(config["episodes"])), chunksize=4), 1):
            traces.write(json.dumps(trace) + "\n")
            counts["completed_games" if trace["completed"] else "truncated_games"] += 1
            counts["env_plies"] += len(game_rows)
            for row in game_rows:
                key = symmetry_key(row["obs"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                x, _ = np.argwhere(row["obs"][0] == 1)[0]
                files[int(x)] += 1
                counts["no_wall_positions"] += not row["obs"][2:4].any()
                counts["teacher_disagreements"] += row["expert_d1"] != row["expert_d2"]
                if row["winning"].any():
                    endings[int(x)] += 1
            if episode % 100 == 0:
                print(json.dumps({"games": episode, "unique_positions": len(rows), "counts": dict(counts)}), flush=True)
    tensors = {key: torch.from_numpy(np.asarray([row[key] for row in rows])) for key in rows[0]}
    metadata = {"config": config, "episodes": config["episodes"], **counts,
        "training_positions": len(rows), "unique_symmetry_groups": len(seen),
        "pawn_files": dict(files), "immediate_win_files": dict(endings),
        "split": "All 2000 sampled episodes used for training; no internal validation split. Development matches are separate.",
        "sampling": "Both seats; mixed D1/D2 behavior; random prefix with lateral moves; random pawn/wall exploration throughout",
        "teachers": "Both runtime Minimax depths on every shared position; first maximum; no numerical search-bound targets",
        "generation_seconds": time.monotonic() - started}
    torch.save({"train": tensors, "validation": {k: v[:0] for k, v in tensors.items()}, "metadata": metadata}, folder / "shared.pt")
    (folder / "dataset.json").write_text(json.dumps(metadata, indent=2))
    (folder / "training-state-keys.json").write_text(json.dumps(sorted(seen)))
    return metadata


def policy_loss(logits, legal, expert):
    if not bool(legal.gather(1, expert[:, None]).all()):
        raise ValueError("Expert labels must be legal")
    return nn.functional.cross_entropy(logits.masked_fill(~legal, -torch.inf), expert)


def train_distillation(cfg, config, output, callback):
    if cfg.model_arch != "cnn" or cfg.cnn_hierarchical or cfg.warm_start_checkpoint:
        raise ValueError("Policy distillation requires a fresh flat CNN")
    if config["depth"] not in (1, 2):
        raise ValueError("Teacher depth must be 1 or 2")
    dataset = torch.load(config["dataset"], map_location="cpu", weights_only=True)
    if dataset["metadata"]["episodes"] != cfg.episodes:
        raise ValueError("Configured episode budget does not match sampled games")
    metadata = {**dataset["metadata"], "source_dataset": config["dataset"],
                "source_sha256": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest()}
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2))
    data = {k: v.to(cfg.device) for k, v in dataset["train"].items()}
    torch.manual_seed(cfg.seed)
    net = CNNQNetwork(cnn_channels=cfg.cnn_channels, activation=cfg.cnn_activation).to(cfg.device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()
    initialization = {"mode": "fresh", "seed": cfg.seed, "initial_weights_sha256": initial_hash}
    (output / "initialization.json").write_text(json.dumps(initialization, indent=2))
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    generator = torch.Generator(device=cfg.device).manual_seed(config["batch_seed"])
    special = (data["winning"].any(1) | (data["expert_d1"] != data["expert_d2"])).nonzero().flatten()
    metrics = []
    for step in range(1, config["updates"] + 1):
        indices = torch.randint(len(data["obs"]), (cfg.batch_size,), device=cfg.device, generator=generator)
        n_special = int(cfg.batch_size * config.get("tactical_fraction", .25))
        if special.numel() and n_special:
            indices[:n_special] = special[torch.randint(len(special), (n_special,), device=cfg.device, generator=generator)]
        obs, legal, expert = (data["obs"][indices], data["legal"][indices], data[f"expert_d{config['depth']}"][indices])
        # Four batch symmetries cycled independently of the deterministic samples.
        obs, legal, expert = reflect_batch(obs, legal, expert, bool(step % 4 & 1), bool(step % 4 & 2))
        logits = net(obs)
        loss = policy_loss(logits, legal, expert)
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm or 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite policy gradient")
        optimizer.step()
        if step % 200 == 0:
            row = {"update": step, "policy_cross_entropy": loss.item(),
                "training_batch_accuracy": (logits.masked_fill(~legal, -torch.inf).argmax(1) == expert).float().mean().item(),
                "max_abs_logit": logits.detach().abs().max().item()}
            metrics.append(row)
            with (output / "distillation-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step in config["checkpoint_updates"] or step == config["updates"]:
            path = output / "checkpoints" / f"update{step:06d}.pt"
            save_checkpoint(path, net, model_arch="cnn", agent_id=cfg.agent_id,
                algorithm="policy_distillation", training_step=step, episode_count=cfg.episodes,
                optimizer=optimizer, device=cfg.device, opponent="shared_minimax_demonstrations",
                eval_summary={"training_objective": "legal-action cross entropy", "output_semantics": "policy_logits",
                    "teacher_depth": config["depth"], "inference": "CNN masked argmax only; no teacher, search or win override",
                    "dataset_sha256": metadata["source_sha256"], "sampled_games": cfg.episodes,
                    "initialization": initialization})
            callback(path, None)
    return {"episodes": cfg.episodes, "updates": config["updates"], "dataset": metadata,
            "initialization": initialization, "metrics": metrics, "output_semantics": "policy_logits"}
