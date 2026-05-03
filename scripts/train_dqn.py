#!/usr/bin/env python3
"""Minimal DQN training script for Quoridor.

Runs a self-contained DQN training loop:
  - learner-vs-random-opponent (alternating P1/P2 each episode)
  - epsilon-greedy exploration with linear decay
  - replay buffer → Bellman train step → target network hard sync
  - periodic checkpoint saves
  - simple evaluation against random at the end

Usage:
    uv run python scripts/train_dqn.py                   # all defaults
    uv run python scripts/train_dqn.py --episodes 50
    uv run python scripts/train_dqn.py --help

Default scale is intentionally small (smoke run, ~100 episodes).
For a real training run, increase --episodes and --buffer-capacity.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Import DQN MVP modules
# ---------------------------------------------------------------------------
from agent_system.training.dqn.action_space import (
    ACTION_COUNT,
    decode_action_id,
    legal_action_ids as _legal_ids,
    legal_action_mask as _legal_mask,
)
from agent_system.training.dqn.checkpoint import save_checkpoint
from agent_system.training.dqn.env import QuoridorEnv
from agent_system.training.dqn.evaluator import (
    DQNCheckpointAgent,
    evaluate_vs_random,
    evaluate_vs_opponent,
    evaluate_vs_opponent_with_replays,
)
from agent_system.training.dqn.model import (
    DEFAULT_HIDDEN_SIZE,
    CNN_DEFAULT_CHANNELS,
    QNetwork,
    CNNQNetwork,
    build_q_network,
    select_epsilon_greedy_action,
)
from agent_system.training.dqn.observation import OBSERVATION_SIZE, encode_observation
from agent_system.training.dqn.observation import OBSERVATION_VERSION as _OBS_V1_VERSION
from agent_system.training.dqn.observation_cnn import (
    CNN_OBSERVATION_VERSION as _OBS_CNN_VERSION,
    CNN_OBSERVATION_SHAPE as _CNN_OBS_SHAPE,
    CNN_OBSERVATION_SIZE as _CNN_OBS_SIZE,
    encode_observation_cnn,
)
from agent_system.training.dqn.opponent import MixedOpponent, build_mixed_opponent, build_opponent
from agent_system.training.dqn.replay_buffer import ReplayBuffer
from agent_system.training.dqn.reward import RewardConfig, RewardBreakdown, compute_reward_breakdown, compute_terminal_reward
from agent_system.training.dqn.trainer import TrainStepResult, sync_target_network, train_step


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Episode budget
    episodes: int = 100
    # Safety guard: max env steps per episode
    max_steps_per_episode: int = 3000
    # Replay buffer
    buffer_capacity: int = 10_000
    warmup_size: int = 256      # min transitions before first optimizer step
    # Batch / optimizer
    batch_size: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    # Exploration (linear decay over learner decision steps)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000  # learner decisions over which to decay
    # Target network
    target_sync_interval: int = 200    # optimizer steps between hard syncs
    # Checkpointing
    checkpoint_dir: str = "agent_system/training/artifacts/dqn/run_001/checkpoints"
    checkpoint_interval: int = 50      # episodes between checkpoint saves
    # Agent identity (stored in checkpoint metadata)
    agent_id: str = "dqn_sanity"
    # Observation version ("v1" = dqn_obs_v1, "v2" = dqn_obs_v2 board-flip)
    obs_version: str = "v1"
    # Evaluation
    eval_games: int = 20
    eval_max_steps: int = 3000
    eval_interval: int = 0            # evaluate every N episodes (0 = only at end)
    eval_replay_sample_every: int = 100  # save full replay every N eval games per opponent (0 = disable)
    # Reproducibility
    seed: int = 42
    # Gradient clipping (None or <= 0 disables clipping)
    grad_clip_norm: float | None = None
    # Device
    device: str = "auto"              # "auto" | "cpu" | "cuda"
    # Logging
    log_interval: int = 10            # episodes between progress prints
    # Training opponent ("random_legal" | "minimax" | "mixed")
    opponent: str = "random_legal"
    opponent_depth: int = 2           # depth for minimax opponent (ignored for random_legal)
    # Mixed-opponent weights (used only when opponent="mixed"; ignored otherwise)
    opponent_mix_dummy: float = 0.0
    opponent_mix_random: float = 0.0
    opponent_mix_minimax_d1: float = 0.0
    opponent_mix_minimax_d2: float = 0.0
    opponent_mix_minimax_d3: float = 0.0
    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE])
    # Model architecture: "mlp" (default, backward-compatible) or "cnn"
    model_arch: str = "mlp"
    # CNN-specific: conv layer output channels (ignored for mlp)
    cnn_channels: list[int] = field(default_factory=lambda: list(CNN_DEFAULT_CHANNELS))
    # Reward shaping
    reward_mode: str = "terminal"     # "terminal" | "distance_delta"
    distance_reward_weight: float = 0.01
    distance_delta_clip: float = 2.0
    # Algorithm
    algorithm: str = "dqn"            # "dqn" | "double_dqn"
    # Structured opponent configs (set by parse_args; derived from new-style YAML or legacy flat args)
    train_opponent_cfg: dict = field(default_factory=dict)
    eval_opponent_cfgs: list[dict] = field(default_factory=list)
    # Episode-based curriculum schedule (list of {from_episode, [to_episode], opponent} dicts)
    # When set, overrides train_opponent_cfg for each episode range.
    train_opponent_schedule: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    - ``"auto"``  → CUDA if available, else CPU.
    - ``"cuda"``  → CUDA; raises ``RuntimeError`` if CUDA is not available.
    - ``"cpu"``   → CPU.
    - ``"cuda:N"``→ specific CUDA device; raises if unavailable.

    Raises
    ------
    RuntimeError:
        When the caller explicitly requested CUDA but it is unavailable.
    ValueError:
        When ``device_arg`` is not a recognised string.
    """
    d = device_arg.strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda" or d.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device_arg}' was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch or use --device auto / --device cpu."
            )
        return torch.device(d)
    raise ValueError(
        f"Unrecognised device string '{device_arg}'. "
        "Expected 'auto', 'cpu', 'cuda', or 'cuda:N'."
    )


@dataclass
class TrainResult:
    config: TrainConfig
    total_env_steps: int = 0
    total_episodes: int = 0
    total_optimizer_steps: int = 0
    total_target_syncs: int = 0
    total_illegal_actions: int = 0
    total_truncated_episodes: int = 0   # episodes that hit max_steps_per_episode
    # Reward distribution (counts of transitions stored in buffer by reward sign)
    total_pos_rewards: int = 0    # reward = +1 (learner wins)
    total_neg_rewards: int = 0    # reward = -1 (opponent wins)
    total_zero_rewards: int = 0   # reward =  0 (non-terminal steps + truncations)
    total_terminal_transitions: int = 0  # done=True transitions stored
    # Per-episode records
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    recent_losses: list[float] = field(default_factory=list)
    recent_q_max: list[float] = field(default_factory=list)   # window of max|Q| per learner step
    # Batch-level diagnostic windows (filled from TrainStepResult)
    recent_q_min: list[float] = field(default_factory=list)
    recent_q_mean: list[float] = field(default_factory=list)
    recent_target_mean: list[float] = field(default_factory=list)
    recent_td_error_max_abs: list[float] = field(default_factory=list)
    recent_batch_done_count: list[int] = field(default_factory=list)
    final_checkpoint_path: str = ""
    # Trained network param count (set after network is built in train())
    param_count: int = 0
    # Per-opponent episode tracking (populated when using mixed opponent)
    opponent_episode_counts: dict = field(default_factory=dict)  # label -> episode count
    opponent_wins: dict = field(default_factory=dict)            # label -> win count
    opponent_losses: dict = field(default_factory=dict)          # label -> loss count
    # Reward shaping diagnostics (populated only when reward_mode=distance_delta)
    total_terminal_reward: float = 0.0
    total_distance_reward: float = 0.0
    total_combined_reward: float = 0.0
    distance_reward_min: float = float("inf")
    distance_reward_max: float = float("-inf")
    clipped_delta_min: float = float("inf")
    clipped_delta_max: float = float("-inf")
    # Per-episode reward breakdown lists (parallel to episode_rewards)
    episode_terminal_rewards: list[float] = field(default_factory=list)
    episode_distance_rewards: list[float] = field(default_factory=list)
    # Evaluation
    eval_result: dict = field(default_factory=dict)
    eval_results: list[dict] = field(default_factory=list)    # periodic evaluation results


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------

def compute_epsilon(step: int, cfg: TrainConfig) -> float:
    """Linearly anneal epsilon from start to end over decay_steps."""
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def _update_diag_windows(result: TrainResult, step_result: TrainStepResult, window: int = 200) -> None:
    """Append batch-level diagnostics from step_result into rolling windows on result."""
    result.recent_q_min.append(step_result.q_min)
    result.recent_q_mean.append(step_result.q_mean)
    result.recent_target_mean.append(step_result.target_mean)
    result.recent_td_error_max_abs.append(step_result.td_error_max_abs)
    result.recent_batch_done_count.append(step_result.done_count)
    if len(result.recent_q_min) > window:
        result.recent_q_min.pop(0)
    if len(result.recent_q_mean) > window:
        result.recent_q_mean.pop(0)
    if len(result.recent_target_mean) > window:
        result.recent_target_mean.pop(0)
    if len(result.recent_td_error_max_abs) > window:
        result.recent_td_error_max_abs.pop(0)
    if len(result.recent_batch_done_count) > window:
        result.recent_batch_done_count.pop(0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a replay batch to the given device."""
    return {k: v.to(device) for k, v in batch.items()}


def train(cfg: TrainConfig) -> TrainResult:
    """Run the DQN training loop and return a TrainResult."""

    from quoridor_engine import Player, RuleEngine

    # Resolve device
    device = resolve_device(cfg.device)
    _resolved_device_str = str(device)           # e.g. "cuda:0" or "cpu"
    _cuda_device_name: str | None = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )

    # Select observation encoder and version based on model_arch / obs_version
    if cfg.model_arch == "cnn":
        _obs_version_str = _OBS_CNN_VERSION
        _encode = encode_observation_cnn
        _obs_shape = list(_CNN_OBS_SHAPE)
        _obs_size_display = _CNN_OBS_SIZE
    elif cfg.obs_version == "v2":
        from agent_system.training.dqn.observation_v2 import (
            OBSERVATION_VERSION as _obs_version_str,
            encode_observation_v2 as _encode,
        )
        _obs_shape = [OBSERVATION_SIZE]
        _obs_size_display = OBSERVATION_SIZE
    else:
        from agent_system.training.dqn.observation import (
            OBSERVATION_VERSION as _obs_version_str,
            encode_observation as _encode,
        )
        _obs_shape = [OBSERVATION_SIZE]
        _obs_size_display = OBSERVATION_SIZE

    # Seeding
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    engine = RuleEngine.standard()

    # Networks
    online_net = build_q_network(
        model_arch=cfg.model_arch,
        action_count=ACTION_COUNT,
        obs_size=OBSERVATION_SIZE,
        hidden_layers=cfg.hidden_layers,
        in_channels=_CNN_OBS_SHAPE[0] if cfg.model_arch == "cnn" else 7,
        cnn_channels=cfg.cnn_channels if cfg.model_arch == "cnn" else None,
    ).to(device)
    target_net = build_q_network(
        model_arch=cfg.model_arch,
        action_count=ACTION_COUNT,
        obs_size=OBSERVATION_SIZE,
        hidden_layers=cfg.hidden_layers,
        in_channels=_CNN_OBS_SHAPE[0] if cfg.model_arch == "cnn" else 7,
        cnn_channels=cfg.cnn_channels if cfg.model_arch == "cnn" else None,
    ).to(device)
    sync_target_network(online_net, target_net)
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)

    result = TrainResult(config=cfg)
    result.param_count = online_net.parameter_count()
    learner_decision_step = 0  # for epsilon decay
    optimizer_step = 0
    _first_batch_diag_done = False  # print batch tensor device once after first train step

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Run directory artifact files
    # ---------------------------------------------------------------------------
    run_dir = checkpoint_dir.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write resolved config.yaml so this run is always config-backed.
    import yaml as _yaml  # PyYAML — already in project deps (pyyaml>=6.0.3)
    import dataclasses as _dc
    _config_dict: dict = _dc.asdict(cfg)
    # Convert list-typed fields to plain lists (they already are, but be explicit)
    with open(run_dir / "config.yaml", "w") as _cfg_fh:
        _yaml.dump(_config_dict, _cfg_fh, default_flow_style=False, sort_keys=True)

    # Open structured metrics files for appending.
    _metrics_fh = open(run_dir / "metrics.jsonl", "a")
    _eval_fh = open(run_dir / "eval_results.jsonl", "a")
    _eval_replays_fh = open(run_dir / "eval_replays.jsonl", "a")

    # Build training opponent from structured cfg (always populated by parse_args)
    # When a schedule is present, pre-compile it and use per-episode selection.
    _compiled_schedule: list[tuple] = []
    _schedule_active: bool = bool(cfg.train_opponent_schedule)
    _current_schedule_label: str = ""  # tracks last logged phase label

    if _schedule_active:
        # Compile schedule: sort entries and pre-build (from, to, opponent, label) tuples
        _sorted_entries = sorted(cfg.train_opponent_schedule, key=lambda e: e["from_episode"])
        for _entry in _sorted_entries:
            _from, _to, _opp, _lbl = _build_schedule_entry(_entry)
            _compiled_schedule.append((_from, _to, _opp, _lbl))

        # Print schedule summary at startup
        print(f"  [train_opponent_schedule] {len(_compiled_schedule)} phase(s):")
        for _from, _to, _opp, _lbl in _compiled_schedule:
            _to_str = str(_to) if _to is not None else "end"
            print(f"    episode {_from}-{_to_str}: {_lbl}")

        # For summary/logging purposes, use a generic label
        opponent = None  # will be resolved per-episode
        opponent_label = "schedule"
    elif cfg.train_opponent_cfg:
        opponent, opponent_label = _build_train_opponent_from_cfg(cfg.train_opponent_cfg)
    elif cfg.opponent == "mixed":
        mix_entries: list[tuple[float, str, int]] = []
        if cfg.opponent_mix_dummy > 0:
            mix_entries.append((cfg.opponent_mix_dummy, "dummy", 1))
        if cfg.opponent_mix_random > 0:
            mix_entries.append((cfg.opponent_mix_random, "random_legal", 1))
        if cfg.opponent_mix_minimax_d1 > 0:
            mix_entries.append((cfg.opponent_mix_minimax_d1, "minimax", 1))
        if cfg.opponent_mix_minimax_d2 > 0:
            mix_entries.append((cfg.opponent_mix_minimax_d2, "minimax", 2))
        if cfg.opponent_mix_minimax_d3 > 0:
            mix_entries.append((cfg.opponent_mix_minimax_d3, "minimax", 3))
        if not mix_entries:
            raise ValueError(
                "--opponent mixed requires at least one non-zero mix weight "
                "(--opponent-mix-dummy, --opponent-mix-random, --opponent-mix-minimax-d1, "
                "--opponent-mix-minimax-d2, or --opponent-mix-minimax-d3)"
            )
        opponent = build_mixed_opponent(mix_entries)
        opponent_label = opponent.description()
    else:
        opponent = build_opponent(cfg.opponent, minimax_depth=cfg.opponent_depth)
        opponent_label = (
            f"minimax(depth={cfg.opponent_depth})"
            if cfg.opponent == "minimax"
            else cfg.opponent
        )

    print(f"DQN Training — {cfg.episodes} episodes | obs_version={_obs_version_str} | device={_resolved_device_str}")
    print(f"  model_arch={cfg.model_arch}")
    print(f"  observation_version={_obs_version_str}")
    print(f"  observation_shape={_obs_shape}")
    print(f"  obs_size={_obs_size_display}, action_count={ACTION_COUNT}")
    if cfg.model_arch == "cnn":
        print(f"  cnn_channels={cfg.cnn_channels}")
    else:
        print(f"  hidden_layers={cfg.hidden_layers}")
    print(f"  parameter_count={online_net.parameter_count()}")
    print(f"  algorithm={cfg.algorithm}")
    print(f"  buffer_capacity={cfg.buffer_capacity}, warmup={cfg.warmup_size}")
    print(f"  batch={cfg.batch_size}, lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  epsilon: {cfg.epsilon_start}→{cfg.epsilon_end} over {cfg.epsilon_decay_steps} steps")
    print(f"  opponent: {opponent_label}")
    print(f"  reward_mode: {cfg.reward_mode}"
          + (f" (weight={cfg.distance_reward_weight}, clip={cfg.distance_delta_clip})"
             if cfg.reward_mode == "distance_delta" else ""))
    print(f"  checkpoint_dir: {checkpoint_dir}")
    # Device diagnostics block — confirms actual GPU use at startup
    print(f"  [device] requested_device={cfg.device}")
    print(f"  [device] resolved_device={_resolved_device_str}")
    print(f"  [device] torch_cuda_available={torch.cuda.is_available()}")
    print(f"  [device] torch_version_cuda={torch.version.cuda}")
    if _cuda_device_name:
        print(f"  [device] cuda_device_name={_cuda_device_name}")
    # Confirm network placement
    _online_dev = next(online_net.parameters()).device
    _target_dev = next(target_net.parameters()).device
    print(f"  [device] online_net_device={_online_dev}")
    print(f"  [device] target_net_device={_target_dev}")
    print()

    # Build reward config
    reward_cfg = RewardConfig(
        mode=cfg.reward_mode,
        distance_reward_weight=cfg.distance_reward_weight,
        distance_delta_clip=cfg.distance_delta_clip,
    )

    t_start = time.time()

    for episode in range(cfg.episodes):
        # Alternate learner seat each episode
        learner_player = Player.P1 if episode % 2 == 0 else Player.P2

        # Episode-based schedule: resolve which phase applies (1-based episode numbering)
        if _schedule_active:
            _phase_result = _resolve_schedule_opponent(_compiled_schedule, episode + 1)
            if _phase_result is not None:
                opponent, opponent_label = _phase_result
            else:
                # No phase matches — keep last opponent (shouldn't happen if schedule covers all episodes)
                if opponent is None:
                    raise RuntimeError(
                        f"train_opponent_schedule does not cover episode {episode + 1} "
                        "and no fallback opponent is defined."
                    )
            # Log phase transitions
            if opponent_label != _current_schedule_label:
                if _current_schedule_label:
                    print(
                        f"[schedule] episode {episode + 1}: switching opponent phase "
                        f"{_current_schedule_label!r} → {opponent_label!r}",
                        flush=True,
                    )
                else:
                    print(
                        f"[schedule] episode {episode + 1}: starting with opponent phase {opponent_label!r}",
                        flush=True,
                    )
                _current_schedule_label = opponent_label

        # Per-episode opponent selection (mixed sampling or fixed single opponent)
        if isinstance(opponent, MixedOpponent):
            ep_opponent, ep_opponent_label = opponent.sample(rng)
        else:
            ep_opponent = opponent
            ep_opponent_label = opponent_label
        result.opponent_episode_counts[ep_opponent_label] = (
            result.opponent_episode_counts.get(ep_opponent_label, 0) + 1
        )

        state = engine.initial_state()
        done = False
        episode_reward = 0.0
        episode_terminal_reward = 0.0
        episode_distance_reward = 0.0
        episode_steps = 0
        episode_illegal = 0
        ep_learner_won = False
        ep_opponent_won = False

        # Deferred-push state: hold the learner's (obs, action_id, prev_state)
        # until after the opponent responds, so that:
        #   1. next_obs is always encoded from the LEARNER's perspective
        #      (learner is current_player again after the opponent acts)
        #   2. reward=-1 can be assigned when the opponent's next move wins
        #   3. prev_state (learner decision state) is available for distance shaping
        pending_obs: list[float] | None = None
        pending_action_id: int | None = None
        pending_prev_state: object | None = None

        while not done and episode_steps < cfg.max_steps_per_episode:
            current_player = state.current_player
            mask = _legal_mask(engine, state)
            legal_ids = [i for i, v in enumerate(mask) if v]

            if current_player == learner_player:
                # Learner turn — decide and record (do NOT push yet)
                obs = _encode(state)
                epsilon = compute_epsilon(learner_decision_step, cfg)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                online_net.eval()
                with torch.no_grad():
                    q_values = online_net(obs_tensor)
                online_net.train()

                action_id = select_epsilon_greedy_action(q_values, mask, epsilon=epsilon, rng=rng)
                if not mask[action_id]:
                    episode_illegal += 1

                pending_obs = obs
                pending_action_id = action_id
                pending_prev_state = state  # learner decision state for distance shaping
                learner_decision_step += 1

                # Track Q-value range for sanity monitoring
                q_abs_max = float(q_values.abs().max().item())
                result.recent_q_max.append(q_abs_max)
                if len(result.recent_q_max) > 200:
                    result.recent_q_max.pop(0)
            else:
                # Opponent turn: episode-level opponent policy
                action_id = ep_opponent.select_action_id(engine, state, legal_ids, rng)

            # Apply action
            engine_action = decode_action_id(action_id, current_player)
            next_state = engine.apply_action(state, engine_action)
            episode_steps += 1
            result.total_env_steps += 1

            done = engine.is_game_over(next_state)

            if current_player == learner_player and done:
                # Learner's move ended the game — learner wins.
                terminal_reward = 1.0
                breakdown = compute_reward_breakdown(
                    engine, pending_prev_state, next_state,
                    learner_player, terminal_reward, reward_cfg,
                )
                reward = breakdown.combined_reward
                episode_reward += reward
                episode_terminal_reward += breakdown.terminal_reward
                episode_distance_reward += breakdown.distance_reward
                next_obs = _encode(next_state)
                buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                pending_obs = None
                pending_prev_state = None
                result.total_pos_rewards += 1
                result.total_terminal_transitions += 1
                # Reward shaping diagnostics
                result.total_terminal_reward += breakdown.terminal_reward
                result.total_distance_reward += breakdown.distance_reward
                result.total_combined_reward += reward
                if breakdown.distance_reward != 0.0 or reward_cfg.mode == "distance_delta":
                    result.distance_reward_min = min(result.distance_reward_min, breakdown.distance_reward)
                    result.distance_reward_max = max(result.distance_reward_max, breakdown.distance_reward)
                if breakdown.clipped_delta is not None:
                    result.clipped_delta_min = min(result.clipped_delta_min, breakdown.clipped_delta)
                    result.clipped_delta_max = max(result.clipped_delta_max, breakdown.clipped_delta)
                ep_learner_won = True

                # Optimizer step
                if buffer.is_ready(cfg.warmup_size):
                    batch = _batch_to_device(buffer.sample(cfg.batch_size, rng=rng), device)
                    step_result = train_step(online_net, target_net, optimizer, batch, gamma=cfg.gamma, grad_clip_norm=cfg.grad_clip_norm, algorithm=cfg.algorithm)
                    optimizer_step += 1
                    result.total_optimizer_steps += 1
                    result.recent_losses.append(step_result.loss)
                    if len(result.recent_losses) > 200:
                        result.recent_losses.pop(0)
                    _update_diag_windows(result, step_result)
                    if not _first_batch_diag_done:
                        print(f"  [device] sample_batch_obs_device={batch['obs'].device}")
                        _first_batch_diag_done = True
                    if optimizer_step % cfg.target_sync_interval == 0:
                        sync_target_network(online_net, target_net)
                        result.total_target_syncs += 1

            elif current_player != learner_player and pending_obs is not None:
                # Opponent just acted — push the deferred learner transition now.
                # next_obs is encoded with the LEARNER as current_player because
                # after the opponent acts it is the learner's turn again (or the
                # game ended, in which case done=True handles the semantics).
                if done:
                    # Opponent won — learner loses
                    terminal_reward = -1.0
                    breakdown = compute_reward_breakdown(
                        engine, pending_prev_state, next_state,
                        learner_player, terminal_reward, reward_cfg,
                    )
                    reward = breakdown.combined_reward
                    next_obs = _encode(next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                    result.total_neg_rewards += 1
                    result.total_terminal_transitions += 1
                    ep_opponent_won = True
                else:
                    # Game continues — next_obs is from the learner's perspective ✓
                    terminal_reward = 0.0
                    breakdown = compute_reward_breakdown(
                        engine, pending_prev_state, next_state,
                        learner_player, terminal_reward, reward_cfg,
                    )
                    reward = breakdown.combined_reward
                    next_obs = _encode(next_state)
                    next_mask = _legal_mask(engine, next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, False, next_mask)
                    result.total_zero_rewards += 1
                episode_reward += reward
                episode_terminal_reward += breakdown.terminal_reward
                episode_distance_reward += breakdown.distance_reward
                pending_obs = None
                pending_prev_state = None
                # Reward shaping diagnostics
                result.total_terminal_reward += breakdown.terminal_reward
                result.total_distance_reward += breakdown.distance_reward
                result.total_combined_reward += reward
                if reward_cfg.mode == "distance_delta":
                    result.distance_reward_min = min(result.distance_reward_min, breakdown.distance_reward)
                    result.distance_reward_max = max(result.distance_reward_max, breakdown.distance_reward)
                if breakdown.clipped_delta is not None:
                    result.clipped_delta_min = min(result.clipped_delta_min, breakdown.clipped_delta)
                    result.clipped_delta_max = max(result.clipped_delta_max, breakdown.clipped_delta)

                # Optimizer step
                if buffer.is_ready(cfg.warmup_size):
                    batch = _batch_to_device(buffer.sample(cfg.batch_size, rng=rng), device)
                    step_result = train_step(online_net, target_net, optimizer, batch, gamma=cfg.gamma, grad_clip_norm=cfg.grad_clip_norm, algorithm=cfg.algorithm)
                    optimizer_step += 1
                    result.total_optimizer_steps += 1
                    result.recent_losses.append(step_result.loss)
                    if len(result.recent_losses) > 200:
                        result.recent_losses.pop(0)
                    _update_diag_windows(result, step_result)
                    if not _first_batch_diag_done:
                        print(f"  [device] sample_batch_obs_device={batch['obs'].device}")
                        _first_batch_diag_done = True
                    if optimizer_step % cfg.target_sync_interval == 0:
                        sync_target_network(online_net, target_net)
                        result.total_target_syncs += 1

            state = next_state

        # Flush any pending learner transition that was truncated by max_steps
        # (neither player won — treat as a draw with reward=0, done=False).
        if pending_obs is not None:
            terminal_reward = 0.0
            breakdown = compute_reward_breakdown(
                engine, pending_prev_state, state,
                learner_player, terminal_reward, reward_cfg,
            )
            flush_reward = breakdown.combined_reward
            next_obs = _encode(state)
            next_mask = _legal_mask(engine, state) if not done else [False] * ACTION_COUNT
            buffer.push(pending_obs, pending_action_id, flush_reward, next_obs, False, next_mask)
            result.total_zero_rewards += 1
            episode_reward += flush_reward
            episode_terminal_reward += breakdown.terminal_reward
            episode_distance_reward += breakdown.distance_reward
            pending_obs = None
            pending_prev_state = None
            result.total_terminal_reward += breakdown.terminal_reward
            result.total_distance_reward += breakdown.distance_reward
            result.total_combined_reward += flush_reward
            if reward_cfg.mode == "distance_delta":
                result.distance_reward_min = min(result.distance_reward_min, breakdown.distance_reward)
                result.distance_reward_max = max(result.distance_reward_max, breakdown.distance_reward)
            if breakdown.clipped_delta is not None:
                result.clipped_delta_min = min(result.clipped_delta_min, breakdown.clipped_delta)
                result.clipped_delta_max = max(result.clipped_delta_max, breakdown.clipped_delta)
            result.total_truncated_episodes += 1

        # Episode done — record per-opponent outcome
        if ep_learner_won:
            result.opponent_wins[ep_opponent_label] = (
                result.opponent_wins.get(ep_opponent_label, 0) + 1
            )
        elif ep_opponent_won:
            result.opponent_losses[ep_opponent_label] = (
                result.opponent_losses.get(ep_opponent_label, 0) + 1
            )

        result.total_episodes += 1
        result.total_illegal_actions += episode_illegal
        result.episode_rewards.append(episode_reward)
        result.episode_terminal_rewards.append(episode_terminal_reward)
        result.episode_distance_rewards.append(episode_distance_reward)
        result.episode_lengths.append(episode_steps)

        # Periodic checkpoint
        is_checkpoint_ep = (episode + 1) % cfg.checkpoint_interval == 0 or episode == cfg.episodes - 1
        if is_checkpoint_ep:
            ckpt_path = checkpoint_dir / f"ep{result.total_episodes:05d}_step{result.total_env_steps}.pt"
            save_checkpoint(
                ckpt_path,
                online_net,
                agent_id=cfg.agent_id,
                training_step=result.total_optimizer_steps,
                episode_count=result.total_episodes,
                optimizer=optimizer,
                obs_version=_obs_version_str,
                device=_resolved_device_str,
                algorithm=cfg.algorithm,
                model_arch=cfg.model_arch,
                cnn_channels=cfg.cnn_channels if cfg.model_arch == "cnn" else None,
                reward_mode=cfg.reward_mode,
                distance_reward_weight=cfg.distance_reward_weight,
                distance_delta_clip=cfg.distance_delta_clip,
                opponent=opponent_label,
            )
            result.final_checkpoint_path = str(ckpt_path)

        # Periodic mid-training evaluation
        is_eval_ep = (
            cfg.eval_interval > 0
            and ((episode + 1) % cfg.eval_interval == 0 or episode == cfg.episodes - 1)
            and result.final_checkpoint_path
        )
        if is_eval_ep:
            import time as _time
            # Compute replay sample indices for this eval batch
            _replay_sample_every = cfg.eval_replay_sample_every
            _sample_indices: set[int] | None = None
            if _replay_sample_every > 0:
                _sample_indices = {i for i in range(cfg.eval_games) if i % _replay_sample_every == 0}
            _replay_dir = run_dir / "replays" if _sample_indices else None
            _ckpt_name = Path(result.final_checkpoint_path).name

            if cfg.eval_opponent_cfgs:
                # New: evaluate against each configured eval opponent
                for _opp_cfg in cfg.eval_opponent_cfgs:
                    _opp_name = _opp_cfg.get("name") or _opp_cfg.get("type", "unknown")
                    _replay_meta = {
                        "run_id": cfg.agent_id,
                        "agent_id": cfg.agent_id,
                        "episode": result.total_episodes,
                        "checkpoint": _ckpt_name,
                        "opp_cfg": _opp_cfg,
                    }
                    eval_r = run_evaluation(
                        result.final_checkpoint_path, cfg, encoder=_encode,
                        eval_opp_cfg=_opp_cfg,
                        sample_indices=_sample_indices,
                        replay_dir=_replay_dir,
                        replay_metadata=_replay_meta,
                    )
                    eval_r["episode"] = result.total_episodes
                    eval_r["env_step"] = result.total_env_steps
                    result.eval_results.append(eval_r)
                    _eval_record: dict = {
                        "type": "eval",
                        "episode": result.total_episodes,
                        "step": result.total_env_steps,
                        "agent_id": cfg.agent_id,
                        "opponent_name": _opp_name,
                        "opponent_type": _opp_cfg.get("type", "unknown"),
                        "opponent_depth": _opp_cfg.get("depth"),
                        "eval_games": eval_r.get("num_games"),
                        "wins": eval_r.get("wins"),
                        "losses": eval_r.get("losses"),
                        "draws": eval_r.get("draws"),
                        "win_rate": eval_r.get("win_rate"),
                        "avg_game_length": eval_r.get("avg_game_length"),
                        "illegal_action_count": eval_r.get("illegal_action_count"),
                        "sampled_replay_count": eval_r.get("sampled_replay_count", 0),
                        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    }
                    _eval_fh.write(json.dumps(_eval_record) + "\n")
                    _eval_fh.flush()
                    # Write replay index entries
                    for _ridx in eval_r.get("_replay_index_entries", []):
                        _eval_replays_fh.write(json.dumps(_ridx) + "\n")
                    _eval_replays_fh.flush()
            else:
                # Legacy: single random_legal eval
                eval_r = run_evaluation(result.final_checkpoint_path, cfg, encoder=_encode)
                eval_r["episode"] = result.total_episodes
                eval_r["env_step"] = result.total_env_steps
                result.eval_results.append(eval_r)
                _eval_record = {
                    "type": "eval",
                    "episode": result.total_episodes,
                    "step": result.total_env_steps,
                    "agent_id": cfg.agent_id,
                    "opponent": eval_r.get("opponent_id"),
                    "eval_games": eval_r.get("num_games"),
                    "wins": eval_r.get("wins"),
                    "losses": eval_r.get("losses"),
                    "draws": eval_r.get("draws"),
                    "win_rate": eval_r.get("win_rate"),
                    "avg_game_length": eval_r.get("avg_game_length"),
                    "illegal_action_count": eval_r.get("illegal_action_count"),
                    "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                }
                _eval_fh.write(json.dumps(_eval_record) + "\n")
                _eval_fh.flush()

        # Logging
        if (episode + 1) % cfg.log_interval == 0 or episode == cfg.episodes - 1:
            recent_ep = min(cfg.log_interval, episode + 1)
            avg_reward = sum(result.episode_rewards[-recent_ep:]) / recent_ep
            avg_len = sum(result.episode_lengths[-recent_ep:]) / recent_ep
            avg_loss = (
                sum(result.recent_losses[-50:]) / len(result.recent_losses[-50:])
                if result.recent_losses
                else float("nan")
            )
            avg_q_max = (
                sum(result.recent_q_max[-200:]) / len(result.recent_q_max[-200:])
                if result.recent_q_max
                else float("nan")
            )
            eps = compute_epsilon(learner_decision_step, cfg)
            print(
                f"  ep {episode+1:>4}/{cfg.episodes}"
                f" | steps {result.total_env_steps:>7}"
                f" | opt_steps {result.total_optimizer_steps:>5}"
                f" | eps {eps:.3f}"
                f" | buf {len(buffer):>6}"
                f" | avg_rew {avg_reward:+.3f}"
                f" | avg_len {avg_len:>5.0f}"
                f" | avg_loss {avg_loss:.4f}"
                f" | avg_q_max {avg_q_max:.3f}"
                f" | +r {result.total_pos_rewards:>4}"
                f" | -r {result.total_neg_rewards:>4}"
            )
            # Reward shaping breakdown line (distance_delta mode only)
            if cfg.reward_mode == "distance_delta" and result.episode_terminal_rewards:
                avg_term = sum(result.episode_terminal_rewards[-recent_ep:]) / recent_ep
                avg_dist = sum(result.episode_distance_rewards[-recent_ep:]) / recent_ep
                avg_comb = avg_term + avg_dist
                print(
                    f"    [reward]"
                    f" avg_term={avg_term:+.4f}"
                    f" avg_dist={avg_dist:+.4f}"
                    f" avg_comb={avg_comb:+.4f}"
                    f" | dist_min={result.distance_reward_min:+.4f}"
                    f" dist_max={result.distance_reward_max:+.4f}"
                )
            # Diagnostic log line with batch-level Q/target/td-error stats
            if result.recent_q_mean:
                avg_q_mean = sum(result.recent_q_mean[-100:]) / len(result.recent_q_mean[-100:])
                avg_q_min = sum(result.recent_q_min[-100:]) / len(result.recent_q_min[-100:])
                avg_tgt_mean = sum(result.recent_target_mean[-100:]) / len(result.recent_target_mean[-100:])
                avg_td_err_max = sum(result.recent_td_error_max_abs[-100:]) / len(result.recent_td_error_max_abs[-100:])
                avg_done_ct = sum(result.recent_batch_done_count[-100:]) / len(result.recent_batch_done_count[-100:])
                print(
                    f"    [diag]"
                    f" q_mean={avg_q_mean:+.3f}"
                    f" q_min={avg_q_min:+.3f}"
                    f" tgt_mean={avg_tgt_mean:+.3f}"
                    f" td_max_abs={avg_td_err_max:.3f}"
                    f" batch_done_ct={avg_done_ct:.2f}"
                    f" | trunc_eps={result.total_truncated_episodes}"
                )

            # Emit structured metrics record to metrics.jsonl
            import time as _time
            _metrics_record: dict = {
                "type": "train",
                "episode": result.total_episodes,
                "step": result.total_env_steps,
                "epsilon": round(eps, 6),
                "loss": round(avg_loss, 6) if avg_loss == avg_loss else None,  # NaN → None
                "avg_reward": round(avg_reward, 6),
                "avg_episode_length": round(avg_len, 2),
                "buffer_size": len(buffer),
                "opt_steps": result.total_optimizer_steps,
                "avg_q_max": round(avg_q_max, 6) if avg_q_max == avg_q_max else None,
                "pos_rewards": result.total_pos_rewards,
                "neg_rewards": result.total_neg_rewards,
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            }
            # Add training opponent info for charting / debug
            if _schedule_active:
                # For schedule, emit the current phase label + per-opponent breakdown
                _metrics_record["train_opponent_phase"] = _current_schedule_label
            _metrics_record["train_opponent_name"] = ep_opponent_label
            _metrics_fh.write(json.dumps(_metrics_record) + "\n")
            _metrics_fh.flush()

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  total env steps:     {result.total_env_steps}")
    print(f"  total episodes:      {result.total_episodes}")
    print(f"  optimizer steps:     {result.total_optimizer_steps}")
    print(f"  target syncs:        {result.total_target_syncs}")
    print(f"  illegal actions:     {result.total_illegal_actions}")
    print(f"  truncated episodes:  {result.total_truncated_episodes}")
    print(f"  pos rewards (+1):    {result.total_pos_rewards}")
    print(f"  neg rewards (-1):    {result.total_neg_rewards}")
    print(f"  zero rewards (0):    {result.total_zero_rewards}")
    print(f"  terminal transitions:{result.total_terminal_transitions}")
    print(f"  checkpoint:          {result.final_checkpoint_path}")
    # Reward shaping summary
    n_trans = result.total_pos_rewards + result.total_neg_rewards + result.total_zero_rewards
    avg_term = result.total_terminal_reward / n_trans if n_trans > 0 else 0.0
    avg_dist = result.total_distance_reward / n_trans if n_trans > 0 else 0.0
    avg_comb = result.total_combined_reward / n_trans if n_trans > 0 else 0.0
    print(f"  reward_mode:         {cfg.reward_mode}")
    print(f"  total_terminal_rew:  {result.total_terminal_reward:+.4f}")
    print(f"  total_distance_rew:  {result.total_distance_reward:+.4f}")
    print(f"  total_combined_rew:  {result.total_combined_reward:+.4f}")
    print(f"  avg_terminal_rew:    {avg_term:+.6f}")
    print(f"  avg_distance_rew:    {avg_dist:+.6f}")
    print(f"  avg_combined_rew:    {avg_comb:+.6f}")
    if cfg.reward_mode == "distance_delta":
        dist_min = result.distance_reward_min if result.distance_reward_min != float("inf") else 0.0
        dist_max = result.distance_reward_max if result.distance_reward_max != float("-inf") else 0.0
        cdelta_min = result.clipped_delta_min if result.clipped_delta_min != float("inf") else 0.0
        cdelta_max = result.clipped_delta_max if result.clipped_delta_max != float("-inf") else 0.0
        print(f"  distance_rew_min:    {dist_min:+.6f}")
        print(f"  distance_rew_max:    {dist_max:+.6f}")
        print(f"  clipped_delta_min:   {cdelta_min:+.6f}")
        print(f"  clipped_delta_max:   {cdelta_max:+.6f}")
    if len(result.opponent_episode_counts) > 1:
        print(f"  opponent episode counts:")
        for lbl, cnt in sorted(result.opponent_episode_counts.items()):
            wins = result.opponent_wins.get(lbl, 0)
            losses = result.opponent_losses.get(lbl, 0)
            pct = 100.0 * cnt / result.total_episodes
            print(f"    {lbl}: {cnt} eps ({pct:.1f}%) | wins={wins} losses={losses}")

    _metrics_fh.close()
    _eval_fh.close()
    _eval_replays_fh.close()

    return result


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    cfg: "TrainConfig",
    encoder=None,
    eval_opp_cfg: dict | None = None,
    sample_indices: "set[int] | None" = None,
    replay_dir: "Path | None" = None,
    replay_metadata: "dict | None" = None,
) -> dict:
    """Load checkpoint and run evaluation.

    Parameters
    ----------
    checkpoint_path:
        Path to the checkpoint ``.pt`` file.
    cfg:
        Active :class:`TrainConfig` (used for ``eval_games``, ``eval_max_steps``, ``seed``).
    encoder:
        Observation encoder callable.  Inferred from the checkpoint if ``None``.
    eval_opp_cfg:
        If ``None``, evaluates against a uniform-random opponent (legacy).
        If a dict with ``type``/``depth``/``name`` keys, evaluates against
        the specified opponent using :func:`evaluate_vs_opponent`.
    sample_indices:
        Game indices (0-based) for which to record full replays.  ``None``
        means no replay recording.
    replay_dir:
        Directory where replay JSON files will be written.  Required when
        *sample_indices* is non-empty.
    replay_metadata:
        Extra metadata forwarded to
        :func:`~agent_system.training.dqn.replay_writer.record_game`
        (``run_id``, ``agent_id``, ``episode``, ``checkpoint``, ``opp_cfg``).
    """
    from quoridor_engine import RuleEngine

    _opp_name: str
    if eval_opp_cfg is None:
        _opp_name = "random_legal"
    else:
        _opp_name = eval_opp_cfg.get("name") or eval_opp_cfg.get("type", "random_legal")

    print(f"\nRunning evaluation vs {_opp_name} from: {checkpoint_path}")
    engine = RuleEngine.standard()
    _expected_obs_version = None
    if encoder is not None:
        import torch as _torch
        raw = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _expected_obs_version = raw.get("observation_version")
    agent = DQNCheckpointAgent.from_path(checkpoint_path, expected_obs_version=_expected_obs_version)
    eval_rng = random.Random(cfg.seed + 1)

    if eval_opp_cfg is None:
        # Legacy: evaluate vs uniform-random
        result = evaluate_vs_random(
            agent=agent,
            engine=engine,
            num_games=cfg.eval_games,
            max_steps=cfg.eval_max_steps,
            rng=eval_rng,
            encoder=encoder,
            opponent_id="random_legal",
        )
        _replays = []
    else:
        # New: evaluate vs specified opponent
        _eval_opp = _build_eval_opponent_from_cfg(eval_opp_cfg)
        if sample_indices:
            # Record replays for selected games
            _rkw = replay_metadata.copy() if replay_metadata else {}
            _rkw.setdefault("opp_cfg", eval_opp_cfg)
            result, _replays = evaluate_vs_opponent_with_replays(
                agent=agent,
                engine=engine,
                opponent=_eval_opp,
                num_games=cfg.eval_games,
                max_steps=cfg.eval_max_steps,
                rng=eval_rng,
                encoder=encoder,
                opponent_id=_opp_name,
                sample_indices=sample_indices,
                replay_kwargs=_rkw,
            )
        else:
            result = evaluate_vs_opponent(
                agent=agent,
                engine=engine,
                opponent=_eval_opp,
                num_games=cfg.eval_games,
                max_steps=cfg.eval_max_steps,
                rng=eval_rng,
                encoder=encoder,
                opponent_id=_opp_name,
            )
            _replays = []

    # Write replay files and build index entries
    _replay_index_entries: list[dict] = []
    if _replays and replay_dir is not None:
        from agent_system.training.dqn.replay_writer import (
            write_replay, replay_filename, replay_index_entry,
        )
        replay_dir = Path(replay_dir)
        replay_dir.mkdir(parents=True, exist_ok=True)
        for _rp in _replays:
            _fname = replay_filename(_rp.episode, _opp_name, _rp.game_index)
            _fpath = replay_dir / _fname
            write_replay(_rp, _fpath)
            _replay_index_entries.append(
                replay_index_entry(_rp, f"replays/{_fname}")
            )

    print(f"  games:          {result.num_games}")
    print(f"  wins:           {result.wins}")
    print(f"  losses:         {result.losses}")
    print(f"  draws:          {result.draws}")
    print(f"  win_rate:       {result.win_rate:.3f}")
    print(f"  avg_game_len:   {result.avg_game_length:.1f}")
    print(f"  illegal_acts:   {result.illegal_action_count}")
    if _replay_index_entries:
        print(f"  replays saved:  {len(_replay_index_entries)}")

    ret: dict = {
        "checkpoint_id": result.checkpoint_id,
        "opponent_id": result.opponent_id,
        "num_games": result.num_games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_game_length": result.avg_game_length,
        "illegal_action_count": result.illegal_action_count,
        "sampled_replay_count": len(_replay_index_entries),
        "_replay_index_entries": _replay_index_entries,
    }
    if eval_opp_cfg is not None:
        ret["opponent_name"] = _opp_name
        ret["opponent_type"] = eval_opp_cfg.get("type", "unknown")
        ret["opponent_depth"] = eval_opp_cfg.get("depth")
    return ret


# ---------------------------------------------------------------------------
# Loaded-checkpoint rollout verification
# ---------------------------------------------------------------------------

def run_checkpoint_rollout(checkpoint_path: str, max_steps: int = 3000, encoder=None) -> dict:
    """Load checkpoint and run a single full rollout to verify legal play."""
    from quoridor_engine import Player, RuleEngine

    print(f"\nRunning single rollout from checkpoint: {checkpoint_path}")
    engine = RuleEngine.standard()
    _expected_obs_version = None
    if encoder is not None:
        import torch as _torch
        raw = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _expected_obs_version = raw.get("observation_version")
    agent = DQNCheckpointAgent.from_path(checkpoint_path, expected_obs_version=_expected_obs_version)
    _encoder = encoder if encoder is not None else encode_observation

    rng = random.Random(0)
    state = engine.initial_state()
    done = False
    steps = 0
    illegal_count = 0

    while not done and steps < max_steps:
        current_player = state.current_player
        mask = _legal_mask(engine, state)
        obs = _encoder(state)

        if current_player == Player.P1:
            action_id = agent.select_action(obs, mask)
            if not mask[action_id]:
                illegal_count += 1
        else:
            legal_ids = [i for i, v in enumerate(mask) if v]
            action_id = rng.choice(legal_ids)

        engine_action = decode_action_id(action_id, current_player)
        state = engine.apply_action(state, engine_action)
        steps += 1
        done = engine.is_game_over(state)

    outcome = "terminal" if done else "max_steps"
    winner = engine.winner(state) if done else None

    print(f"  rollout steps:    {steps}")
    print(f"  outcome:          {outcome}")
    print(f"  winner:           {winner}")
    print(f"  illegal_actions:  {illegal_count}")

    return {
        "steps": steps,
        "outcome": outcome,
        "winner": str(winner) if winner else None,
        "illegal_action_count": illegal_count,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str) -> tuple[dict, dict]:
    """Load a YAML config file.

    Returns
    -------
    (flat_defaults, raw_yaml)
        *flat_defaults* is a dict suitable for ``argparse.set_defaults``.
        List-valued keys ``hidden_layers`` and ``cnn_channels`` are converted
        to comma-separated strings.  Nested ``train_opponent`` and
        ``evaluation`` sections are translated into flat defaults where
        possible (so legacy CLI flags still work) and omitted from the flat
        dict otherwise.
        *raw_yaml* is the full parsed YAML dict, used by ``parse_args`` to
        build the structured ``train_opponent_cfg`` / ``eval_opponent_cfgs``.
    """
    import yaml  # PyYAML — already in project deps (pyyaml>=6.0.3)

    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: --config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path) as fh:
            raw: dict = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        print(f"ERROR: failed to parse YAML config {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    flat: dict = {}
    # Copy all non-structured scalar/list keys to flat defaults.
    for k, v in raw.items():
        if k in ("train_opponent", "evaluation", "train_opponent_schedule"):
            continue
        if k in ("hidden_layers", "cnn_channels") and isinstance(v, list):
            flat[k] = ",".join(str(x) for x in v)
        else:
            flat[k] = v

    # Translate train_opponent section → flat argparse defaults.
    train_opp = raw.get("train_opponent")
    if train_opp:
        opp_type = train_opp.get("type", "random_legal")
        flat["opponent"] = opp_type
        if opp_type == "minimax":
            flat["opponent_depth"] = train_opp.get("depth", 2)
        elif opp_type == "mixed":
            for item in train_opp.get("opponents", []):
                t = item.get("type", "")
                w = item.get("weight", 0.0)
                d = item.get("depth", 2)
                if t == "random_legal":
                    flat["opponent_mix_random"] = w
                elif t == "dummy":
                    flat["opponent_mix_dummy"] = w
                elif t == "minimax":
                    if d == 1:
                        flat["opponent_mix_minimax_d1"] = w
                    elif d == 2:
                        flat["opponent_mix_minimax_d2"] = w
                    elif d == 3:
                        flat["opponent_mix_minimax_d3"] = w

    # Translate evaluation section → flat argparse defaults.
    evaluation = raw.get("evaluation")
    if evaluation:
        if "interval" in evaluation:
            flat["eval_interval"] = evaluation["interval"]
        if "games_per_opponent" in evaluation:
            flat["eval_games"] = evaluation["games_per_opponent"]
        if "replay_sample_every" in evaluation:
            flat["eval_replay_sample_every"] = evaluation["replay_sample_every"]

    return flat, raw


def _flat_args_to_train_opponent_cfg(args: object) -> dict:
    """Convert legacy flat argparse args to a normalized ``train_opponent_cfg`` dict."""
    opponent = getattr(args, "opponent", "random_legal")
    if opponent == "mixed":
        opponents: list[dict] = []
        if getattr(args, "opponent_mix_dummy", 0.0) > 0:
            opponents.append({"name": "dummy", "type": "dummy",
                               "weight": args.opponent_mix_dummy})  # type: ignore[attr-defined]
        if getattr(args, "opponent_mix_random", 0.0) > 0:
            opponents.append({"name": "random_legal", "type": "random_legal",
                               "weight": args.opponent_mix_random})  # type: ignore[attr-defined]
        if getattr(args, "opponent_mix_minimax_d1", 0.0) > 0:
            opponents.append({"name": "minimax_d1", "type": "minimax", "depth": 1,
                               "weight": args.opponent_mix_minimax_d1})  # type: ignore[attr-defined]
        if getattr(args, "opponent_mix_minimax_d2", 0.0) > 0:
            opponents.append({"name": "minimax_d2", "type": "minimax", "depth": 2,
                               "weight": args.opponent_mix_minimax_d2})  # type: ignore[attr-defined]
        if getattr(args, "opponent_mix_minimax_d3", 0.0) > 0:
            opponents.append({"name": "minimax_d3", "type": "minimax", "depth": 3,
                               "weight": args.opponent_mix_minimax_d3})  # type: ignore[attr-defined]
        return {"type": "mixed", "opponents": opponents}
    elif opponent == "minimax":
        return {"type": "minimax", "depth": getattr(args, "opponent_depth", 2)}
    else:
        return {"type": opponent}


def _build_train_opponent_from_cfg(opp_cfg: dict):
    """Build a training opponent (and label) from a ``train_opponent_cfg`` dict."""
    opp_type = opp_cfg.get("type", "random_legal")
    if opp_type == "mixed":
        mix_entries: list[tuple[float, str, int]] = []
        for item in opp_cfg.get("opponents", []):
            w = item.get("weight", 0.0)
            if w <= 0:
                continue
            t = item.get("type", "random_legal")
            d = item.get("depth", 2)
            mix_entries.append((w, t, d))
        if not mix_entries:
            raise ValueError(
                "train_opponent type=mixed has no entries with positive weight"
            )
        from agent_system.training.dqn.opponent import MixedOpponent as _MixedOpponent
        opponent = build_mixed_opponent(mix_entries)
        label = opponent.description()
    elif opp_type == "minimax":
        d = opp_cfg.get("depth", 2)
        opponent = build_opponent("minimax", minimax_depth=d)
        label = f"minimax(depth={d})"
    else:
        opponent = build_opponent(opp_type)
        label = opp_type
    return opponent, label


def _build_eval_opponent_from_cfg(opp_cfg: dict):
    """Build a single eval opponent from an eval opponent config dict."""
    opp_type = opp_cfg.get("type", "random_legal")
    depth = opp_cfg.get("depth", 2)
    return build_opponent(opp_type, minimax_depth=depth)


# ---------------------------------------------------------------------------
# Opponent schedule helpers
# ---------------------------------------------------------------------------

def _validate_opponent_schedule(schedule: list[dict]) -> None:
    """Validate a ``train_opponent_schedule`` list.

    Checks performed:
    - At least one entry.
    - Each entry has ``from_episode`` (positive int) and ``opponent`` dict.
    - Optional ``to_episode`` is >= ``from_episode`` when present.
    - No overlapping episode ranges.
    - Opponent types are valid.
    - Mixed opponents have at least one positive weight.
    - Minimax depths are 1, 2 or 3.

    Raises ``ValueError`` with a descriptive message on any violation.
    """
    if not schedule:
        raise ValueError("train_opponent_schedule must contain at least one entry.")

    # Validate individual entries
    for i, entry in enumerate(schedule):
        if "from_episode" not in entry:
            raise ValueError(f"train_opponent_schedule[{i}]: missing required key 'from_episode'.")
        if "opponent" not in entry:
            raise ValueError(f"train_opponent_schedule[{i}]: missing required key 'opponent'.")
        fe = entry["from_episode"]
        if not isinstance(fe, int) or fe < 1:
            raise ValueError(
                f"train_opponent_schedule[{i}]: 'from_episode' must be a positive integer, got {fe!r}."
            )
        te = entry.get("to_episode")
        if te is not None:
            if not isinstance(te, int) or te < fe:
                raise ValueError(
                    f"train_opponent_schedule[{i}]: 'to_episode' ({te!r}) must be an integer >= from_episode ({fe})."
                )
        # Validate opponent config
        opp = entry["opponent"]
        if not isinstance(opp, dict):
            raise ValueError(
                f"train_opponent_schedule[{i}]: 'opponent' must be a dict, got {type(opp).__name__}."
            )
        opp_type = opp.get("type", "")
        if opp_type not in ("random_legal", "dummy", "minimax", "mixed"):
            raise ValueError(
                f"train_opponent_schedule[{i}]: unsupported opponent type {opp_type!r}. "
                "Expected one of: random_legal, dummy, minimax, mixed."
            )
        if opp_type == "minimax":
            d = opp.get("depth", 2)
            if d not in (1, 2, 3):
                raise ValueError(
                    f"train_opponent_schedule[{i}]: minimax depth must be 1, 2 or 3, got {d!r}."
                )
        if opp_type == "mixed":
            sub_opps = opp.get("opponents", [])
            if not sub_opps:
                raise ValueError(
                    f"train_opponent_schedule[{i}]: mixed opponent has no 'opponents' list."
                )
            total_w = sum(item.get("weight", 0.0) for item in sub_opps)
            if total_w <= 0:
                raise ValueError(
                    f"train_opponent_schedule[{i}]: mixed opponent has no entries with positive weight."
                )
            for j, item in enumerate(sub_opps):
                st = item.get("type", "")
                if st not in ("random_legal", "dummy", "minimax"):
                    raise ValueError(
                        f"train_opponent_schedule[{i}].opponents[{j}]: unsupported type {st!r}."
                    )
                if st == "minimax":
                    sd = item.get("depth", 2)
                    if sd not in (1, 2, 3):
                        raise ValueError(
                            f"train_opponent_schedule[{i}].opponents[{j}]: minimax depth must be 1, 2 or 3, got {sd!r}."
                        )

    # Check for overlapping ranges
    sorted_entries = sorted(schedule, key=lambda e: e["from_episode"])
    for i in range(len(sorted_entries) - 1):
        curr = sorted_entries[i]
        nxt = sorted_entries[i + 1]
        curr_to = curr.get("to_episode", float("inf"))
        if nxt["from_episode"] <= curr_to:
            raise ValueError(
                f"train_opponent_schedule: overlapping ranges between entry "
                f"[from={curr['from_episode']}, to={curr.get('to_episode', '∞')}] and "
                f"[from={nxt['from_episode']}, to={nxt.get('to_episode', '∞')}]."
            )


def _build_schedule_entry(entry: dict) -> tuple:
    """Build (from_ep, to_ep, opponent, label) for one schedule entry dict."""
    from_ep = entry["from_episode"]
    to_ep = entry.get("to_episode")           # None means "until end of training"
    opp, label = _build_train_opponent_from_cfg(entry["opponent"])
    return from_ep, to_ep, opp, label


def _resolve_schedule_opponent(
    compiled_schedule: list[tuple],
    episode_1based: int,
) -> tuple | None:
    """Return ``(opponent, label)`` for the given 1-based episode number.

    *compiled_schedule* is a list of ``(from_ep, to_ep_or_None, opponent, label)`` tuples,
    sorted ascending by ``from_ep``.

    Returns ``None`` if no matching phase is found (caller should fall back to
    ``train_opponent_cfg``).
    """
    for from_ep, to_ep, opp, label in compiled_schedule:
        if episode_1based < from_ep:
            continue
        if to_ep is None or episode_1based <= to_ep:
            return opp, label
    return None


def parse_args() -> TrainConfig:
    # --- Phase 1: pre-parse to extract --config before the full parse ---
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()

    yaml_defaults: dict = {}
    _raw_yaml: dict = {}
    if _pre_args.config:
        yaml_defaults, _raw_yaml = _load_yaml_config(_pre_args.config)

    p = argparse.ArgumentParser(description="DQN training for Quoridor")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file. Values from YAML are used as defaults "
                        "and can be overridden by explicit CLI arguments.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps-per-episode", type=int, default=3000)
    p.add_argument("--buffer-capacity", type=int, default=10_000)
    p.add_argument("--warmup-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=5_000)
    p.add_argument("--target-sync-interval", type=int, default=200)
    p.add_argument("--checkpoint-dir", type=str,
                   default="agent_system/training/artifacts/dqn/run_001/checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=50)
    p.add_argument("--agent-id", type=str, default="dqn_sanity")
    p.add_argument("--obs-version", type=str, default="v1", choices=["v1", "v2"],
                   help="Observation encoder version (v1 = dqn_obs_v1, v2 = dqn_obs_v2 board-flip)")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=0,
                   help="Evaluate every N episodes during training (0 = only at end)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--device", type=str, default="auto",
                   help="Device to train on: auto (default), cpu, or cuda")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="Max gradient norm for clipping (omit or 0 to disable)")
    p.add_argument("--opponent", type=str, default="random_legal",
                   choices=["random_legal", "dummy", "minimax", "mixed"],
                   help="Training opponent policy (default: random_legal)")
    p.add_argument("--opponent-depth", type=int, default=2,
                   help="Search depth for minimax opponent (default: 2, ignored for random_legal/mixed)")
    p.add_argument("--opponent-mix-random", type=float, default=0.0,
                   help="Weight for random_legal in mixed opponent (used only with --opponent mixed)")
    p.add_argument("--opponent-mix-minimax-d1", type=float, default=0.0,
                   help="Weight for minimax depth=1 in mixed opponent (used only with --opponent mixed)")
    p.add_argument("--opponent-mix-dummy", type=float, default=0.0,
                   help="Weight for dummy (first-legal) in mixed opponent (used only with --opponent mixed)")
    p.add_argument("--opponent-mix-minimax-d2", type=float, default=0.0,
                   help="Weight for minimax depth=2 in mixed opponent (used only with --opponent mixed)")
    p.add_argument("--opponent-mix-minimax-d3", type=float, default=0.0,
                   help="Weight for minimax depth=3 in mixed opponent (used only with --opponent mixed)")
    p.add_argument("--hidden-layers", type=str, default="",
                   help="Comma-separated hidden layer widths, e.g. '512,512' or '512,512,256'. "
                        "Defaults to '256,256' (matches DEFAULT_HIDDEN_SIZE).")
    p.add_argument("--reward-mode", type=str, default="terminal",
                   choices=["terminal", "distance_delta"],
                   help="Reward mode: 'terminal' (default, sparse +1/-1) or "
                        "'distance_delta' (terminal + distance-advantage shaping)")
    p.add_argument("--distance-reward-weight", type=float, default=0.01,
                   help="Scale factor for distance shaping reward (default: 0.01, "
                        "used only with --reward-mode distance_delta)")
    p.add_argument("--distance-delta-clip", type=float, default=2.0,
                   help="Symmetric clip bound for distance advantage delta (default: 2.0, "
                        "used only with --reward-mode distance_delta)")
    p.add_argument("--algorithm", type=str, default="dqn",
                   choices=["dqn", "double_dqn"],
                   help="Training algorithm: 'dqn' (default) or 'double_dqn' (Double DQN)")
    p.add_argument("--model-arch", type=str, default="mlp",
                   choices=["mlp", "cnn"],
                   help="Model architecture: 'mlp' (default, MLP Q-network) or 'cnn' "
                        "(CNN Q-network with dqn_obs_cnn_v1 encoder)")
    p.add_argument("--cnn-channels", type=str, default="",
                   help="Comma-separated CNN conv-layer channel widths, e.g. '32,64,64' "
                        "(only used with --model-arch cnn; defaults to '32,64,64')")

    # Inject YAML values as defaults before parsing.  Any argument the user
    # provides explicitly on the CLI will still override the YAML value
    # because argparse defaults have lower precedence than explicit CLI values.
    if yaml_defaults:
        p.set_defaults(**yaml_defaults)

    args = p.parse_args()

    # Parse hidden_layers from comma-separated string
    if args.hidden_layers.strip():
        try:
            _hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",") if x.strip()]
        except ValueError:
            import sys as _sys
            print(f"ERROR: --hidden-layers value '{args.hidden_layers}' is not valid. "
                  "Expected comma-separated integers, e.g. '256,256' or '512,512,256'.",
                  file=_sys.stderr)
            _sys.exit(1)
    else:
        _hidden_layers = [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE]

    # Parse cnn_channels from comma-separated string
    if args.cnn_channels.strip():
        try:
            _cnn_channels = [int(x.strip()) for x in args.cnn_channels.split(",") if x.strip()]
        except ValueError:
            import sys as _sys
            print(f"ERROR: --cnn-channels value '{args.cnn_channels}' is not valid. "
                  "Expected comma-separated integers, e.g. '32,64,64'.",
                  file=_sys.stderr)
            _sys.exit(1)
    else:
        _cnn_channels = list(CNN_DEFAULT_CHANNELS)

    # Build structured opponent configs from new-style YAML or legacy flat args.
    _schedule: list[dict] = _raw_yaml.get("train_opponent_schedule", [])

    # Conflict check: train_opponent and train_opponent_schedule are mutually exclusive.
    if "train_opponent" in _raw_yaml and _schedule:
        print(
            "ERROR: YAML config contains both 'train_opponent' and 'train_opponent_schedule'. "
            "These are mutually exclusive. Use 'train_opponent' for a static opponent or "
            "'train_opponent_schedule' for episode-based curriculum scheduling.",
            file=sys.stderr,
        )
        sys.exit(1)

    if _schedule:
        # Validate schedule early (before training starts) for fast feedback.
        try:
            _validate_opponent_schedule(_schedule)
        except ValueError as exc:
            print(f"ERROR: invalid train_opponent_schedule: {exc}", file=sys.stderr)
            sys.exit(1)
        _train_opp_cfg = {}   # schedule takes precedence; no static opponent needed
    elif "train_opponent" in _raw_yaml:
        _train_opp_cfg = _raw_yaml["train_opponent"]
    else:
        _train_opp_cfg = _flat_args_to_train_opponent_cfg(args)

    _eval_section = _raw_yaml.get("evaluation", {})
    _eval_opp_cfgs: list[dict] = _eval_section.get("opponents", [])

    return TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        buffer_capacity=args.buffer_capacity,
        warmup_size=args.warmup_size,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_interval=args.target_sync_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        agent_id=args.agent_id,
        obs_version=args.obs_version,
        eval_games=args.eval_games,
        eval_interval=args.eval_interval,
        eval_replay_sample_every=getattr(args, "eval_replay_sample_every", 100),
        seed=args.seed,
        log_interval=args.log_interval,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
        opponent=args.opponent,
        opponent_depth=args.opponent_depth,
        opponent_mix_dummy=args.opponent_mix_dummy,
        opponent_mix_random=args.opponent_mix_random,
        opponent_mix_minimax_d1=args.opponent_mix_minimax_d1,
        opponent_mix_minimax_d2=args.opponent_mix_minimax_d2,
        opponent_mix_minimax_d3=args.opponent_mix_minimax_d3,
        reward_mode=args.reward_mode,
        distance_reward_weight=args.distance_reward_weight,
        distance_delta_clip=args.distance_delta_clip,
        hidden_layers=_hidden_layers,
        algorithm=args.algorithm,
        model_arch=args.model_arch,
        cnn_channels=_cnn_channels,
        train_opponent_cfg=_train_opp_cfg,
        eval_opponent_cfgs=_eval_opp_cfgs,
        train_opponent_schedule=_schedule,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    # Select encoder for post-training rollout / evaluation
    if cfg.model_arch == "cnn":
        _post_encode = encode_observation_cnn
    elif cfg.obs_version == "v2":
        from agent_system.training.dqn.observation_v2 import encode_observation_v2 as _post_encode
    else:
        _post_encode = encode_observation

    # Resolve device here so it's available for the summary JSON below.
    _resolved_device_str = str(resolve_device(cfg.device))
    _cuda_device_name: str | None = (
        torch.cuda.get_device_name(torch.device(_resolved_device_str))
        if _resolved_device_str.startswith("cuda")
        else None
    )

    result = train(cfg)

    if not result.final_checkpoint_path:
        print("ERROR: No checkpoint was saved.", file=sys.stderr)
        sys.exit(1)

    # Rollout verification
    rollout = run_checkpoint_rollout(result.final_checkpoint_path, encoder=_post_encode)

    # Evaluation
    if cfg.eval_opponent_cfgs:
        # New: evaluate against each configured eval opponent
        import time as _ftime
        _run_dir_fe = Path(cfg.checkpoint_dir).parent
        _eval_results_list: list[dict] = []
        with open(_run_dir_fe / "eval_results.jsonl", "a") as _final_eval_fh:
            for _opp_cfg in cfg.eval_opponent_cfgs:
                eval_result = run_evaluation(result.final_checkpoint_path, cfg, encoder=_post_encode, eval_opp_cfg=_opp_cfg)
                result.eval_result = eval_result  # keep last for summary JSON
                _eval_results_list.append(eval_result)
                _opp_name_fe = _opp_cfg.get("name") or _opp_cfg.get("type", "unknown")
                _final_eval_record: dict = {
                    "type": "eval_final",
                    "episode": result.total_episodes,
                    "step": result.total_env_steps,
                    "agent_id": cfg.agent_id,
                    "opponent_name": _opp_name_fe,
                    "opponent_type": _opp_cfg.get("type", "unknown"),
                    "opponent_depth": _opp_cfg.get("depth"),
                    "eval_games": eval_result.get("num_games"),
                    "wins": eval_result.get("wins"),
                    "losses": eval_result.get("losses"),
                    "draws": eval_result.get("draws"),
                    "win_rate": eval_result.get("win_rate"),
                    "avg_game_length": eval_result.get("avg_game_length"),
                    "illegal_action_count": eval_result.get("illegal_action_count"),
                    "timestamp": _ftime.strftime("%Y-%m-%dT%H:%M:%SZ", _ftime.gmtime()),
                }
                _final_eval_fh.write(json.dumps(_final_eval_record) + "\n")
        eval_result = result.eval_result
    else:
        # Legacy: single random_legal eval
        eval_result = run_evaluation(result.final_checkpoint_path, cfg, encoder=_post_encode)
        result.eval_result = eval_result

        # Write final eval to eval_results.jsonl in the run directory
        _run_dir = Path(cfg.checkpoint_dir).parent
        with open(_run_dir / "eval_results.jsonl", "a") as _final_eval_fh:
            import time as _time
            _final_eval_record = {
                "type": "eval_final",
                "episode": result.total_episodes,
                "step": result.total_env_steps,
                "agent_id": cfg.agent_id,
                "opponent": eval_result.get("opponent_id"),
                "eval_games": eval_result.get("num_games"),
                "wins": eval_result.get("wins"),
                "losses": eval_result.get("losses"),
                "draws": eval_result.get("draws"),
                "win_rate": eval_result.get("win_rate"),
                "avg_game_length": eval_result.get("avg_game_length"),
                "illegal_action_count": eval_result.get("illegal_action_count"),
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            }
            _final_eval_fh.write(json.dumps(_final_eval_record) + "\n")

    # Reward averages (per-transition) for summary JSON
    _n_trans = result.total_pos_rewards + result.total_neg_rewards + result.total_zero_rewards
    avg_term = result.total_terminal_reward / _n_trans if _n_trans > 0 else 0.0
    avg_dist = result.total_distance_reward / _n_trans if _n_trans > 0 else 0.0
    avg_comb = result.total_combined_reward / _n_trans if _n_trans > 0 else 0.0

    # Summary JSON to stdout (for report generation)
    # Determine observation version and shape for the summary
    if cfg.model_arch == "cnn":
        _summary_obs_version = _OBS_CNN_VERSION
        _summary_obs_shape = list(_CNN_OBS_SHAPE)
    elif cfg.obs_version == "v2":
        from agent_system.training.dqn.observation_v2 import OBSERVATION_VERSION as _obs_v2_str
        _summary_obs_version = _obs_v2_str
        _summary_obs_shape = [OBSERVATION_SIZE]
    else:
        _summary_obs_version = _OBS_V1_VERSION
        _summary_obs_shape = [OBSERVATION_SIZE]

    summary = {
        "total_env_steps": result.total_env_steps,
        "total_episodes": result.total_episodes,
        "total_optimizer_steps": result.total_optimizer_steps,
        "total_target_syncs": result.total_target_syncs,
        "total_illegal_actions": result.total_illegal_actions,
        "total_truncated_episodes": result.total_truncated_episodes,
        "total_pos_rewards": result.total_pos_rewards,
        "total_neg_rewards": result.total_neg_rewards,
        "total_zero_rewards": result.total_zero_rewards,
        "total_terminal_transitions": result.total_terminal_transitions,
        "model_arch": cfg.model_arch,
        "observation_version": _summary_obs_version,
        "observation_shape": _summary_obs_shape,
        "hidden_layers": cfg.hidden_layers if cfg.model_arch == "mlp" else None,
        "cnn_channels": cfg.cnn_channels if cfg.model_arch == "cnn" else None,
        "parameter_count": result.param_count,
        "algorithm": cfg.algorithm,
        "is_double_dqn": cfg.algorithm == "double_dqn",
        "opponent": cfg.opponent,
        "requested_device": cfg.device,
        "resolved_device": _resolved_device_str,
        "cuda_device_name": _cuda_device_name,
        "avg_episode_reward": (
            sum(result.episode_rewards) / len(result.episode_rewards)
            if result.episode_rewards else 0.0
        ),
        "avg_episode_length": (
            sum(result.episode_lengths) / len(result.episode_lengths)
            if result.episode_lengths else 0.0
        ),
        "final_avg_loss": (
            sum(result.recent_losses[-50:]) / len(result.recent_losses[-50:])
            if result.recent_losses else None
        ),
        "final_avg_q_max": (
            sum(result.recent_q_max[-200:]) / len(result.recent_q_max[-200:])
            if result.recent_q_max else None
        ),
        "checkpoint_path": result.final_checkpoint_path,
        "opponent_episode_counts": result.opponent_episode_counts,
        "opponent_wins_by_label": result.opponent_wins,
        "opponent_losses_by_label": result.opponent_losses,
        "reward_mode": cfg.reward_mode,
        "distance_reward_weight": cfg.distance_reward_weight,
        "distance_delta_clip": cfg.distance_delta_clip,
        "total_terminal_reward": result.total_terminal_reward,
        "total_distance_reward": result.total_distance_reward,
        "total_combined_reward": result.total_combined_reward,
        "avg_terminal_reward": avg_term,
        "avg_distance_reward": avg_dist,
        "avg_combined_reward": avg_comb,
        "distance_reward_min": (
            result.distance_reward_min if result.distance_reward_min != float("inf") else None
        ),
        "distance_reward_max": (
            result.distance_reward_max if result.distance_reward_max != float("-inf") else None
        ),
        "clipped_delta_min": (
            result.clipped_delta_min if result.clipped_delta_min != float("inf") else None
        ),
        "clipped_delta_max": (
            result.clipped_delta_max if result.clipped_delta_max != float("-inf") else None
        ),
        "rollout": rollout,
        "evaluation": eval_result,
        "periodic_evals": result.eval_results,
    }

    print("\n--- JSON SUMMARY ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
