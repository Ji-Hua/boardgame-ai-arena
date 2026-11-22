"""Integration smoke test for the DQN training loop — Phase 5.

Verifies the end-to-end path:
    QuoridorEnv -> DQNPolicy (epsilon-greedy) -> ReplayBuffer -> train_step

This test does NOT verify learning quality; it only proves the machinery
runs without crashes and produces sensible diagnostics.

Coverage:
    - Transitions collected from RLEnv correctly (obs, reward, done, mask)
    - Replay buffer reaches trainable size
    - At least one optimizer step runs successfully
    - Loss is finite after the update
    - No illegal actions are selected during the collection phase
    - Full loop (collect + update + target sync) does not crash
    - Multiple episodes complete inside the smoke loop
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.optim as optim

from quoridor_engine import Player, RuleEngine

from agent_system.training.dqn.action_space import ACTION_COUNT
from agent_system.training.dqn.env import QuoridorEnv
from agent_system.training.dqn.model import DQNPolicy, QNetwork
from agent_system.training.dqn.observation import OBSERVATION_SIZE
from agent_system.training.dqn.replay_buffer import ReplayBuffer
from agent_system.training.dqn.trainer import (
    TrainStepResult,
    sync_target_network,
    train_step,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_env(learner: Player = Player.P1) -> QuoridorEnv:
    return QuoridorEnv(RuleEngine.standard(), learner_player=learner)


def _make_policy(seed: int = 0) -> DQNPolicy:
    torch.manual_seed(seed)
    return DQNPolicy(QNetwork())


def _collect_transitions(
    env: QuoridorEnv,
    policy: DQNPolicy,
    buf: ReplayBuffer,
    rng: random.Random,
    epsilon: float = 1.0,
    max_steps: int = 2000,
) -> dict:
    """Run one episode with epsilon-greedy policy, storing transitions.

    Returns stats: step_count, illegal_count, final_reward.
    """
    obs = env.reset()
    done = False
    step_count = 0
    illegal_count = 0
    final_reward = 0.0

    while not done and step_count < max_steps:
        mask = env.legal_action_mask()
        action_id = policy.select_action(obs, mask, epsilon=epsilon, rng=rng)
        if not mask[action_id]:
            illegal_count += 1
        next_obs, reward, done, _ = env.step(action_id)
        buf.push(
            obs=obs,
            action_id=action_id,
            reward=reward,
            next_obs=next_obs,
            done=done,
            next_legal_mask=env.legal_action_mask() if not done else [False] * ACTION_COUNT,
        )
        obs = next_obs
        step_count += 1
        final_reward = reward

    return {
        "step_count": step_count,
        "illegal_count": illegal_count,
        "final_reward": final_reward,
        "done": done,
    }


# ---------------------------------------------------------------------------
# TestSmokeCollectTransitions
# ---------------------------------------------------------------------------

class TestSmokeCollectTransitions:
    def test_episode_completes(self):
        env = _make_env()
        policy = _make_policy()
        buf = ReplayBuffer(capacity=2000)
        rng = random.Random(0)
        stats = _collect_transitions(env, policy, buf, rng, epsilon=1.0)
        assert stats["done"] is True

    def test_replay_buffer_fills_after_episode(self):
        env = _make_env()
        policy = _make_policy()
        buf = ReplayBuffer(capacity=2000)
        rng = random.Random(1)
        _collect_transitions(env, policy, buf, rng, epsilon=1.0)
        assert len(buf) > 0

    def test_no_illegal_actions_with_epsilon_one(self):
        env = _make_env()
        policy = _make_policy()
        buf = ReplayBuffer(capacity=2000)
        rng = random.Random(2)
        stats = _collect_transitions(env, policy, buf, rng, epsilon=1.0)
        assert stats["illegal_count"] == 0

    def test_obs_in_buffer_has_correct_size(self):
        env = _make_env()
        policy = _make_policy()
        buf = ReplayBuffer(capacity=200)
        rng = random.Random(3)
        _collect_transitions(env, policy, buf, rng, epsilon=1.0)
        assert len(buf) > 0
        batch = buf.sample(1, rng=rng)
        assert batch["obs"].shape == (1, OBSERVATION_SIZE)

    def test_next_mask_in_buffer_has_correct_size(self):
        env = _make_env()
        policy = _make_policy()
        buf = ReplayBuffer(capacity=200)
        rng = random.Random(4)
        _collect_transitions(env, policy, buf, rng, epsilon=1.0)
        batch = buf.sample(1, rng=rng)
        assert batch["next_mask"].shape == (1, ACTION_COUNT)


# ---------------------------------------------------------------------------
# TestSmokeTrainStep
# ---------------------------------------------------------------------------

class TestSmokeTrainStep:
    def _warmup_buffer(
        self,
        capacity: int = 500,
        min_size: int = 64,
        seed: int = 0,
    ) -> ReplayBuffer:
        """Collect enough transitions to enable a train step."""
        env = _make_env()
        policy = _make_policy(seed)
        buf = ReplayBuffer(capacity=capacity)
        rng = random.Random(seed)
        while not buf.is_ready(min_size):
            _collect_transitions(env, policy, buf, rng, epsilon=1.0)
        return buf

    def test_buffer_reaches_trainable_size(self):
        buf = self._warmup_buffer(min_size=32)
        assert len(buf) >= 32

    def test_train_step_runs_without_error(self):
        buf = self._warmup_buffer(min_size=32)
        torch.manual_seed(0)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        batch = buf.sample(32, rng=random.Random(0))
        result = train_step(online, target, optimizer, batch)
        assert isinstance(result, TrainStepResult)

    def test_train_step_loss_is_finite(self):
        buf = self._warmup_buffer(min_size=32)
        torch.manual_seed(0)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        batch = buf.sample(32, rng=random.Random(1))
        result = train_step(online, target, optimizer, batch)
        import math
        assert math.isfinite(result.loss)

    def test_optimizer_step_updates_online_net(self):
        buf = self._warmup_buffer(min_size=32)
        torch.manual_seed(0)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        params_before = [p.clone() for p in online.parameters()]
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        batch = buf.sample(32, rng=random.Random(2))
        train_step(online, target, optimizer, batch)
        changed = any(
            not torch.equal(b, p)
            for b, p in zip(params_before, online.parameters())
        )
        assert changed

    def test_target_sync_works_during_loop(self):
        buf = self._warmup_buffer(min_size=32)
        torch.manual_seed(0)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        batch = buf.sample(32, rng=random.Random(3))
        train_step(online, target, optimizer, batch)
        sync_target_network(online, target)
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert torch.equal(p_o, p_t)


# ---------------------------------------------------------------------------
# TestSmokeFullLoop
# ---------------------------------------------------------------------------

class TestSmokeFullLoop:
    """End-to-end smoke: collect -> train -> sync for N steps."""

    def test_full_loop_does_not_crash(self):
        """Run a minimal but complete DQN training loop without crashing."""
        rng = random.Random(42)
        torch.manual_seed(42)

        env = _make_env(Player.P1)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        buf = ReplayBuffer(capacity=1000)
        policy = DQNPolicy(online)

        batch_size = 32
        warmup = 64
        target_sync_every = 50   # optimizer steps between target sync
        max_train_steps = 5      # small for smoke test speed
        optimizer_steps = 0
        episodes = 0
        illegal_count = 0

        while optimizer_steps < max_train_steps:
            # Collect one full episode
            obs = env.reset()
            done = False
            while not done:
                mask = env.legal_action_mask()
                action_id = policy.select_action(obs, mask, epsilon=1.0, rng=rng)
                if not mask[action_id]:
                    illegal_count += 1
                next_obs, reward, done, _ = env.step(action_id)
                next_mask = (
                    env.legal_action_mask() if not done
                    else [False] * ACTION_COUNT
                )
                buf.push(obs, action_id, reward, next_obs, done, next_mask)
                obs = next_obs
            episodes += 1

            # Train when ready
            if buf.is_ready(warmup):
                batch = buf.sample(batch_size, rng=rng)
                result = train_step(online, target, optimizer, batch)
                optimizer_steps += 1
                assert import_math_isfinite(result.loss)
                if optimizer_steps % target_sync_every == 0:
                    sync_target_network(online, target)

        assert optimizer_steps == max_train_steps
        assert illegal_count == 0
        assert episodes >= 1

    def test_multiple_episodes_complete(self):
        """Prove that multiple episodes can run consecutively without error."""
        rng = random.Random(7)
        torch.manual_seed(7)
        env = _make_env()
        policy = _make_policy(7)
        buf = ReplayBuffer(capacity=500)

        completed = 0
        for _ in range(3):
            stats = _collect_transitions(env, policy, buf, rng, epsilon=1.0)
            if stats["done"]:
                completed += 1

        assert completed == 3
        assert len(buf) > 0


# ---------------------------------------------------------------------------
# Module-level helper (avoids importing math at class level)
# ---------------------------------------------------------------------------

def import_math_isfinite(val: float) -> bool:
    import math
    return math.isfinite(val)
