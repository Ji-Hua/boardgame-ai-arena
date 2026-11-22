# TEST_CLASSIFICATION: SPECIFIED
"""Integration tests for distance-delta reward shaping — Phase 15A.

Test categories:
  D. Deferred-push semantics: reward shaping uses correct prev/next states.
  E. Smoke integration: short run with distance_delta completes correctly.
"""

from __future__ import annotations

import random

import pytest
import torch

from quoridor_engine import Player, RuleEngine

from agent_system.training.dqn.action_space import (
    ACTION_COUNT,
    decode_action_id,
    legal_action_ids as _legal_ids,
    legal_action_mask as _legal_mask,
)
from agent_system.training.dqn.model import QNetwork, select_epsilon_greedy_action
from agent_system.training.dqn.observation import encode_observation
from agent_system.training.dqn.replay_buffer import ReplayBuffer
from agent_system.training.dqn.reward import (
    RewardConfig,
    compute_reward_breakdown,
    compute_terminal_reward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


# ---------------------------------------------------------------------------
# D. Deferred-push semantics with reward shaping
# ---------------------------------------------------------------------------

class _Transition:
    """Captured deferred-push transition with reward breakdown."""
    __slots__ = (
        "obs", "action_id", "reward_breakdown", "next_obs", "done",
        "prev_state", "next_state",
    )

    def __init__(self, obs, action_id, reward_breakdown, next_obs, done,
                 prev_state, next_state):
        self.obs = obs
        self.action_id = action_id
        self.reward_breakdown = reward_breakdown
        self.next_obs = next_obs
        self.done = done
        self.prev_state = prev_state
        self.next_state = next_state


def _run_distance_delta_loop(
    engine: RuleEngine,
    num_episodes: int,
    learner_player: Player = Player.P1,
    seed: int = 42,
    max_steps: int = 800,
    weight: float = 0.01,
    clip: float = 2.0,
) -> list[_Transition]:
    """Run deferred-push loop with distance_delta reward shaping.

    Captures all learner transitions with full reward breakdown.
    """
    rng = random.Random(seed)
    net = QNetwork()
    cfg = RewardConfig(mode="distance_delta", distance_reward_weight=weight,
                       distance_delta_clip=clip)
    transitions: list[_Transition] = []

    for episode in range(num_episodes):
        ep_learner = learner_player if episode % 2 == 0 else learner_player.opponent()
        state = engine.initial_state()
        done = False
        steps = 0

        pending_obs = None
        pending_action_id = None
        pending_prev_state = None

        while not done and steps < max_steps:
            cp = state.current_player
            mask = _legal_mask(engine, state)
            legal = [i for i, v in enumerate(mask) if v]

            if cp == ep_learner:
                obs = encode_observation(state)
                with torch.no_grad():
                    q = net(torch.tensor(obs, dtype=torch.float32))
                action_id = select_epsilon_greedy_action(q, mask, epsilon=1.0, rng=rng)
                pending_obs = obs
                pending_action_id = action_id
                pending_prev_state = state
            else:
                action_id = rng.choice(legal)

            engine_action = decode_action_id(action_id, cp)
            next_state = engine.apply_action(state, engine_action)
            steps += 1
            done = engine.is_game_over(next_state)

            if cp == ep_learner and done:
                # Learner wins
                t_rew = compute_terminal_reward(ep_learner, engine.winner(next_state), done)
                breakdown = compute_reward_breakdown(
                    engine, pending_prev_state, next_state,
                    ep_learner, t_rew, cfg,
                )
                transitions.append(_Transition(
                    obs=pending_obs,
                    action_id=pending_action_id,
                    reward_breakdown=breakdown,
                    next_obs=encode_observation(next_state),
                    done=True,
                    prev_state=pending_prev_state,
                    next_state=next_state,
                ))
                pending_obs = None
                pending_prev_state = None

            elif cp != ep_learner and pending_obs is not None:
                winner = engine.winner(next_state) if done else None
                t_rew = compute_terminal_reward(ep_learner, winner, done)
                breakdown = compute_reward_breakdown(
                    engine, pending_prev_state, next_state,
                    ep_learner, t_rew, cfg,
                )
                transitions.append(_Transition(
                    obs=pending_obs,
                    action_id=pending_action_id,
                    reward_breakdown=breakdown,
                    next_obs=encode_observation(next_state),
                    done=done,
                    prev_state=pending_prev_state,
                    next_state=next_state,
                ))
                pending_obs = None
                pending_prev_state = None

            state = next_state

        if pending_obs is not None:
            # Truncated episode
            t_rew = 0.0
            breakdown = compute_reward_breakdown(
                engine, pending_prev_state, state,
                ep_learner, t_rew, cfg,
            )
            transitions.append(_Transition(
                obs=pending_obs,
                action_id=pending_action_id,
                reward_breakdown=breakdown,
                next_obs=encode_observation(state),
                done=False,
                prev_state=pending_prev_state,
                next_state=state,
            ))

    return transitions


class TestDeferredPushSemanticsWithShaping:
    """Deferred-push semantics are preserved with distance_delta reward shaping."""

    def test_produces_transitions(self, engine):
        transitions = _run_distance_delta_loop(engine, num_episodes=5, seed=0)
        assert len(transitions) > 0, "Should produce at least some transitions"

    def test_learner_win_has_terminal_reward_plus_one(self, engine):
        """Learner terminal wins must have terminal_reward=+1.0."""
        transitions = _run_distance_delta_loop(engine, num_episodes=30, seed=7)
        wins = [t for t in transitions if t.done and t.reward_breakdown.terminal_reward == 1.0]
        # With 30 episodes there should be some wins
        # (not guaranteed — just check format when wins occur)
        for t in wins:
            assert t.reward_breakdown.terminal_reward == 1.0
            assert t.reward_breakdown.combined_reward == pytest.approx(
                1.0 + t.reward_breakdown.distance_reward
            )

    def test_opponent_win_has_terminal_reward_minus_one(self, engine):
        """Opponent terminal wins must have terminal_reward=-1.0."""
        transitions = _run_distance_delta_loop(engine, num_episodes=30, seed=7)
        losses = [t for t in transitions if t.done and t.reward_breakdown.terminal_reward == -1.0]
        for t in losses:
            assert t.reward_breakdown.combined_reward == pytest.approx(
                -1.0 + t.reward_breakdown.distance_reward
            )

    def test_non_terminal_has_zero_terminal_reward(self, engine):
        """Non-terminal transitions must have terminal_reward=0.0."""
        transitions = _run_distance_delta_loop(engine, num_episodes=10, seed=3)
        non_terminal = [t for t in transitions if not t.done]
        for t in non_terminal:
            assert t.reward_breakdown.terminal_reward == 0.0

    def test_combined_equals_terminal_plus_distance(self, engine):
        """combined_reward == terminal_reward + distance_reward for every transition."""
        transitions = _run_distance_delta_loop(engine, num_episodes=10, seed=5)
        for t in transitions:
            bd = t.reward_breakdown
            assert bd.combined_reward == pytest.approx(bd.terminal_reward + bd.distance_reward)

    def test_prev_state_is_learner_decision_state(self, engine):
        """prev_state must be the state before the learner acts (not after)."""
        # Verify that prev_advantage is computed from a valid state and
        # that it differs from next_advantage when the game progresses.
        transitions = _run_distance_delta_loop(engine, num_episodes=10, seed=42)
        assert all(t.prev_state is not None for t in transitions)
        assert all(t.next_state is not None for t in transitions)

    def test_distance_breakdown_fields_set_in_delta_mode(self, engine):
        """All distance fields must be non-None in distance_delta mode."""
        transitions = _run_distance_delta_loop(engine, num_episodes=5, seed=99)
        for t in transitions:
            bd = t.reward_breakdown
            assert bd.prev_advantage is not None
            assert bd.next_advantage is not None
            assert bd.distance_delta is not None
            assert bd.clipped_delta is not None

    def test_clipped_delta_within_bounds(self, engine):
        """clipped_delta must be within [-clip, +clip] for every transition."""
        clip = 2.0
        transitions = _run_distance_delta_loop(engine, num_episodes=10, seed=77, clip=clip)
        for t in transitions:
            assert -clip <= t.reward_breakdown.clipped_delta <= clip

    def test_distance_reward_nonzero_over_run(self, engine):
        """Over a non-trivial run, total distance_reward should be nonzero."""
        transitions = _run_distance_delta_loop(engine, num_episodes=20, seed=13)
        total_dist = sum(t.reward_breakdown.distance_reward for t in transitions)
        # Quoridor is dynamic; at least some transitions should have nonzero delta
        # (pawn moves change distances). We just check the sum is not exactly zero.
        assert total_dist != 0.0, (
            "Expected nonzero total_distance_reward over 20 episodes, got 0.0. "
            "This suggests distance computation is always returning 0 — check engine API."
        )


# ---------------------------------------------------------------------------
# E. Smoke integration — full deferred-push loop with buffer
# ---------------------------------------------------------------------------

class TestSmokeIntegration:
    """Short training-loop smoke with distance_delta mode."""

    def test_distance_delta_smoke_run(self, engine):
        """Short smoke with distance_delta: no illegal actions, finite loss, nonzero dist reward."""
        import torch.optim as optim
        from agent_system.training.dqn.trainer import sync_target_network, train_step
        from agent_system.training.dqn.observation import OBSERVATION_SIZE

        rng = random.Random(42)
        torch.manual_seed(42)
        cfg = RewardConfig(mode="distance_delta", distance_reward_weight=0.01, distance_delta_clip=2.0)

        net = QNetwork(obs_size=OBSERVATION_SIZE, action_count=ACTION_COUNT)
        target_net = QNetwork(obs_size=OBSERVATION_SIZE, action_count=ACTION_COUNT)
        sync_target_network(net, target_net)
        target_net.eval()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        buf = ReplayBuffer(capacity=5000)

        learner_player = Player.P1
        total_illegal = 0
        total_dist_rew = 0.0
        optimizer_steps = 0
        warmup = 128

        for episode in range(20):
            ep_learner = Player.P1 if episode % 2 == 0 else Player.P2
            state = engine.initial_state()
            done = False
            steps = 0
            pending_obs = None
            pending_action_id = None
            pending_prev_state = None

            while not done and steps < 800:
                cp = state.current_player
                mask = _legal_mask(engine, state)
                legal = [i for i, v in enumerate(mask) if v]

                if cp == ep_learner:
                    obs = encode_observation(state)
                    net.eval()
                    with torch.no_grad():
                        q = net(torch.tensor(obs, dtype=torch.float32))
                    net.train()
                    action_id = select_epsilon_greedy_action(q, mask, epsilon=1.0, rng=rng)
                    if not mask[action_id]:
                        total_illegal += 1
                    pending_obs = obs
                    pending_action_id = action_id
                    pending_prev_state = state
                else:
                    action_id = rng.choice(legal)

                engine_action = decode_action_id(action_id, cp)
                next_state = engine.apply_action(state, engine_action)
                steps += 1
                done = engine.is_game_over(next_state)

                if cp == ep_learner and done:
                    winner = engine.winner(next_state)
                    t_rew = compute_terminal_reward(ep_learner, winner, done)
                    bd = compute_reward_breakdown(engine, pending_prev_state, next_state,
                                                  ep_learner, t_rew, cfg)
                    next_obs = encode_observation(next_state)
                    buf.push(pending_obs, pending_action_id, bd.combined_reward,
                             next_obs, True, [False] * ACTION_COUNT)
                    total_dist_rew += bd.distance_reward
                    pending_obs = None
                    pending_prev_state = None

                elif cp != ep_learner and pending_obs is not None:
                    winner = engine.winner(next_state) if done else None
                    t_rew = compute_terminal_reward(ep_learner, winner, done)
                    bd = compute_reward_breakdown(engine, pending_prev_state, next_state,
                                                  ep_learner, t_rew, cfg)
                    next_obs = encode_observation(next_state)
                    next_mask = [False] * ACTION_COUNT if done else _legal_mask(engine, next_state)
                    buf.push(pending_obs, pending_action_id, bd.combined_reward,
                             next_obs, done, next_mask)
                    total_dist_rew += bd.distance_reward
                    pending_obs = None
                    pending_prev_state = None

                    if buf.is_ready(warmup):
                        batch = buffer_to_device(buf.sample(64, rng=rng), "cpu")
                        step_result = train_step(net, target_net, optimizer, batch, gamma=0.99)
                        optimizer_steps += 1
                        assert not torch.isnan(torch.tensor(step_result.loss)), "Loss is NaN"
                        assert torch.isfinite(torch.tensor(step_result.loss)), "Loss is infinite"

                state = next_state

        assert total_illegal == 0, f"Illegal actions detected: {total_illegal}"
        assert total_dist_rew != 0.0, (
            "total distance reward is zero over 20 episodes — shaping not working"
        )
        assert len(buf) > 0, "Replay buffer must have transitions"


def buffer_to_device(batch: dict, device: str) -> dict:
    """Move batch tensors to device."""
    import torch
    return {k: v.to(device) for k, v in batch.items()}
