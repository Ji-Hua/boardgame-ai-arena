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
from agent_system.training.dqn.evaluator import DQNCheckpointAgent, evaluate_vs_random
from agent_system.training.dqn.model import (
    DEFAULT_HIDDEN_SIZE,
    QNetwork,
    select_epsilon_greedy_action,
)
from agent_system.training.dqn.observation import OBSERVATION_SIZE, encode_observation
from agent_system.training.dqn.observation import OBSERVATION_VERSION as _OBS_V1_VERSION
from agent_system.training.dqn.replay_buffer import ReplayBuffer
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
    # Reproducibility
    seed: int = 42
    # Logging
    log_interval: int = 10            # episodes between progress prints


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

def train(cfg: TrainConfig) -> TrainResult:
    """Run the DQN training loop and return a TrainResult."""

    from quoridor_engine import Player, RuleEngine

    # Select observation encoder based on configured version
    if cfg.obs_version == "v2":
        from agent_system.training.dqn.observation_v2 import (
            OBSERVATION_VERSION as _obs_version_str,
            encode_observation_v2 as _encode,
        )
    else:
        from agent_system.training.dqn.observation import (
            OBSERVATION_VERSION as _obs_version_str,
            encode_observation as _encode,
        )

    # Seeding
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    engine = RuleEngine.standard()

    # Networks
    online_net = QNetwork(
        hidden_size=DEFAULT_HIDDEN_SIZE,
        obs_size=OBSERVATION_SIZE,
        action_count=ACTION_COUNT,
    )
    target_net = QNetwork(
        hidden_size=DEFAULT_HIDDEN_SIZE,
        obs_size=OBSERVATION_SIZE,
        action_count=ACTION_COUNT,
    )
    sync_target_network(online_net, target_net)
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)

    result = TrainResult(config=cfg)
    learner_decision_step = 0  # for epsilon decay
    optimizer_step = 0

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"DQN Training — {cfg.episodes} episodes | obs_version={_obs_version_str}")
    print(f"  obs_size={OBSERVATION_SIZE}, action_count={ACTION_COUNT}")
    print(f"  buffer_capacity={cfg.buffer_capacity}, warmup={cfg.warmup_size}")
    print(f"  batch={cfg.batch_size}, lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  epsilon: {cfg.epsilon_start}→{cfg.epsilon_end} over {cfg.epsilon_decay_steps} steps")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print()

    t_start = time.time()

    for episode in range(cfg.episodes):
        # Alternate learner seat each episode
        learner_player = Player.P1 if episode % 2 == 0 else Player.P2

        state = engine.initial_state()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        episode_illegal = 0

        # Deferred-push state: hold the learner's (obs, action_id) until
        # after the opponent responds, so that:
        #   1. next_obs is always encoded from the LEARNER's perspective
        #      (learner is current_player again after the opponent acts)
        #   2. reward=-1 can be assigned when the opponent's next move wins
        pending_obs: list[float] | None = None
        pending_action_id: int | None = None

        while not done and episode_steps < cfg.max_steps_per_episode:
            current_player = state.current_player
            mask = _legal_mask(engine, state)
            legal_ids = [i for i, v in enumerate(mask) if v]

            if current_player == learner_player:
                # Learner turn — decide and record (do NOT push yet)
                obs = _encode(state)
                epsilon = compute_epsilon(learner_decision_step, cfg)

                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                online_net.eval()
                with torch.no_grad():
                    q_values = online_net(obs_tensor)
                online_net.train()

                action_id = select_epsilon_greedy_action(q_values, mask, epsilon=epsilon, rng=rng)
                if not mask[action_id]:
                    episode_illegal += 1

                pending_obs = obs
                pending_action_id = action_id
                learner_decision_step += 1

                # Track Q-value range for sanity monitoring
                q_abs_max = float(q_values.abs().max().item())
                result.recent_q_max.append(q_abs_max)
                if len(result.recent_q_max) > 200:
                    result.recent_q_max.pop(0)
            else:
                # Opponent turn: uniform random legal
                action_id = rng.choice(legal_ids)

            # Apply action
            engine_action = decode_action_id(action_id, current_player)
            next_state = engine.apply_action(state, engine_action)
            episode_steps += 1
            result.total_env_steps += 1

            done = engine.is_game_over(next_state)

            if current_player == learner_player and done:
                # Learner's move ended the game — learner wins (only the moving
                # player can win in Quoridor; winning requires reaching goal row
                # on your own turn).
                reward = 1.0
                episode_reward += reward
                next_obs = _encode(next_state)
                buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                pending_obs = None
                result.total_pos_rewards += 1
                result.total_terminal_transitions += 1

                # Optimizer step
                if buffer.is_ready(cfg.warmup_size):
                    batch = buffer.sample(cfg.batch_size, rng=rng)
                    step_result = train_step(online_net, target_net, optimizer, batch, gamma=cfg.gamma)
                    optimizer_step += 1
                    result.total_optimizer_steps += 1
                    result.recent_losses.append(step_result.loss)
                    if len(result.recent_losses) > 200:
                        result.recent_losses.pop(0)
                    _update_diag_windows(result, step_result)
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
                    reward = -1.0
                    next_obs = _encode(next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                    result.total_neg_rewards += 1
                    result.total_terminal_transitions += 1
                else:
                    # Game continues — next_obs is from the learner's perspective ✓
                    reward = 0.0
                    next_obs = _encode(next_state)
                    next_mask = _legal_mask(engine, next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, False, next_mask)
                    result.total_zero_rewards += 1
                episode_reward += reward
                pending_obs = None

                # Optimizer step
                if buffer.is_ready(cfg.warmup_size):
                    batch = buffer.sample(cfg.batch_size, rng=rng)
                    step_result = train_step(online_net, target_net, optimizer, batch, gamma=cfg.gamma)
                    optimizer_step += 1
                    result.total_optimizer_steps += 1
                    result.recent_losses.append(step_result.loss)
                    if len(result.recent_losses) > 200:
                        result.recent_losses.pop(0)
                    _update_diag_windows(result, step_result)
                    if optimizer_step % cfg.target_sync_interval == 0:
                        sync_target_network(online_net, target_net)
                        result.total_target_syncs += 1

            state = next_state

        # Flush any pending learner transition that was truncated by max_steps
        # (neither player won — treat as a draw with reward=0, done=False).
        if pending_obs is not None:
            next_obs = _encode(state)
            next_mask = _legal_mask(engine, state) if not done else [False] * ACTION_COUNT
            buffer.push(pending_obs, pending_action_id, 0.0, next_obs, False, next_mask)
            result.total_zero_rewards += 1
            pending_obs = None
            result.total_truncated_episodes += 1

        # Episode done
        result.total_episodes += 1
        result.total_illegal_actions += episode_illegal
        result.episode_rewards.append(episode_reward)
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
            )
            result.final_checkpoint_path = str(ckpt_path)

        # Periodic mid-training evaluation
        is_eval_ep = (
            cfg.eval_interval > 0
            and ((episode + 1) % cfg.eval_interval == 0 or episode == cfg.episodes - 1)
            and result.final_checkpoint_path
        )
        if is_eval_ep:
            eval_r = run_evaluation(result.final_checkpoint_path, cfg, encoder=_encode)
            eval_r["episode"] = result.total_episodes
            eval_r["env_step"] = result.total_env_steps
            result.eval_results.append(eval_r)

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

    return result


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def run_evaluation(checkpoint_path: str, cfg: TrainConfig, encoder=None) -> dict:
    """Load checkpoint and run simple evaluation against Random."""
    from quoridor_engine import RuleEngine

    print(f"\nRunning evaluation from: {checkpoint_path}")
    engine = RuleEngine.standard()
    # Derive the expected obs version from the encoder if one is provided.
    # Fall back to loading raw payload to read the stored version.
    _expected_obs_version = None
    if encoder is not None:
        import torch as _torch
        raw = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _expected_obs_version = raw.get("observation_version")
    agent = DQNCheckpointAgent.from_path(checkpoint_path, expected_obs_version=_expected_obs_version)
    eval_rng = random.Random(cfg.seed + 1)

    result = evaluate_vs_random(
        agent=agent,
        engine=engine,
        num_games=cfg.eval_games,
        max_steps=cfg.eval_max_steps,
        rng=eval_rng,
        encoder=encoder,
        opponent_id="random_legal",
    )

    print(f"  games:          {result.num_games}")
    print(f"  wins:           {result.wins}")
    print(f"  losses:         {result.losses}")
    print(f"  draws:          {result.draws}")
    print(f"  win_rate:       {result.win_rate:.3f}")
    print(f"  avg_game_len:   {result.avg_game_length:.1f}")
    print(f"  illegal_acts:   {result.illegal_action_count}")

    return {
        "checkpoint_id": result.checkpoint_id,
        "opponent_id": result.opponent_id,
        "num_games": result.num_games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_game_length": result.avg_game_length,
        "illegal_action_count": result.illegal_action_count,
    }


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

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="DQN training for Quoridor")
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
    args = p.parse_args()

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
        seed=args.seed,
        log_interval=args.log_interval,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    # Select encoder for post-training rollout / evaluation
    if cfg.obs_version == "v2":
        from agent_system.training.dqn.observation_v2 import encode_observation_v2 as _post_encode
    else:
        _post_encode = encode_observation

    result = train(cfg)

    if not result.final_checkpoint_path:
        print("ERROR: No checkpoint was saved.", file=sys.stderr)
        sys.exit(1)

    # Rollout verification
    rollout = run_checkpoint_rollout(result.final_checkpoint_path, encoder=_post_encode)

    # Evaluation
    eval_result = run_evaluation(result.final_checkpoint_path, cfg, encoder=_post_encode)
    result.eval_result = eval_result

    # Summary JSON to stdout (for report generation)
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
        "rollout": rollout,
        "evaluation": eval_result,
        "periodic_evals": result.eval_results,
    }

    print("\n--- JSON SUMMARY ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
