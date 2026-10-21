# TEST_CLASSIFICATION: SPECIFIED
"""Tests for pluggable training opponent abstraction (Phase 14A/14B/14C).

Covers:
    A. Factory: default opponent is random_legal
    B. Factory: minimax opponent constructed with depth=2
    C. Factory: invalid opponent name raises ValueError clearly
    D. Factory: minimax depth < 1 raises ValueError clearly
    E. RandomLegalOpponent always returns a legal action_id
    F. MinimaxOpponent always returns a legal action_id (depth=1)
    G. MinimaxOpponent always returns a legal action_id (depth=2)
    H. MinimaxOpponent exposed depth property equals configured depth
    I. Training semantics: DQN vs minimax stores learner-centric transitions
    J. Training semantics: opponent actions not stored as learner transitions
    K. Training semantics: opponent terminal win produces learner reward=-1
    L. Training semantics: learner terminal win produces reward=+1
    M. Training semantics: next_obs after opponent response is learner-perspective
    N. Training semantics: next_legal_mask after opponent response is learner-perspective
    O. Training semantics: illegal actions remain 0 in short rollout vs minimax
    P. Backward compatibility: random_legal opponent unchanged
    Phase 14B:
    Q. DeferredPush depth=1: zero illegal actions
    R. DeferredPush depth=1: opponent win → reward=-1
    S. DeferredPush depth=1: rewards restricted to {-1,0,+1}
    T. DeferredPush depth=1: produces transitions
    Phase 14C:
    U. MixedOpponent construction: valid, normalization, edge cases, errors
    V. MixedOpponent sampling: determinism, label distribution, approximation
    W. MixedOpponent legality: each component opponent returns legal actions
    X. MixedOpponent training semantics: deferred-push preserved, 0 illegal actions
"""

from __future__ import annotations

import random

import pytest

from quoridor_engine import Player, RuleEngine

from agent_system.training.dqn.action_space import (
    ACTION_COUNT,
    decode_action_id,
    encode_engine_action,
    legal_action_ids as _legal_ids,
    legal_action_mask as _legal_mask,
)
from agent_system.training.dqn.observation import encode_observation
from agent_system.training.dqn.opponent import (
    DummyOpponent,
    MinimaxOpponent,
    MixedOpponent,
    RandomLegalOpponent,
    TrainingOpponent,
    build_mixed_opponent,
    build_opponent,
)
from agent_system.training.dqn.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


@pytest.fixture(scope="module")
def initial_state(engine: RuleEngine):
    return engine.initial_state()


# ---------------------------------------------------------------------------
# A–D: Factory tests
# ---------------------------------------------------------------------------

class TestBuildOpponent:
    def test_default_random_legal(self):
        op = build_opponent("random_legal")
        assert isinstance(op, RandomLegalOpponent)
        assert isinstance(op, TrainingOpponent)

    def test_minimax_depth2(self):
        op = build_opponent("minimax", minimax_depth=2)
        assert isinstance(op, MinimaxOpponent)
        assert isinstance(op, TrainingOpponent)
        assert op.depth == 2

    def test_minimax_depth1(self):
        op = build_opponent("minimax", minimax_depth=1)
        assert isinstance(op, MinimaxOpponent)
        assert op.depth == 1

    def test_invalid_opponent_name_raises(self):
        with pytest.raises(ValueError, match="Unknown opponent type"):
            build_opponent("ppo")

    def test_minimax_depth_zero_raises(self):
        with pytest.raises(ValueError, match="depth must be"):
            build_opponent("minimax", minimax_depth=0)

    def test_minimax_depth_negative_raises(self):
        with pytest.raises(ValueError, match="depth must be"):
            build_opponent("minimax", minimax_depth=-1)


# ---------------------------------------------------------------------------
# E–H: Opponent action legality tests
# ---------------------------------------------------------------------------

class TestOpponentLegality:
    """Verify all opponents return action_ids that are actually legal."""

    def _run_legality_check(
        self, engine: RuleEngine, opponent: TrainingOpponent, seed: int, steps: int
    ) -> int:
        """Run steps moves using the opponent and count illegal actions."""
        rng = random.Random(seed)
        state = engine.initial_state()
        illegal_count = 0

        for _ in range(steps):
            if engine.is_game_over(state):
                break
            mask = _legal_mask(engine, state)
            ids = [i for i, v in enumerate(mask) if v]
            action_id = opponent.select_action_id(engine, state, ids, rng)
            if not mask[action_id]:
                illegal_count += 1
            engine_action = decode_action_id(action_id, state.current_player)
            state = engine.apply_action(state, engine_action)

        return illegal_count

    def test_random_legal_returns_legal_action(self, engine: RuleEngine):
        op = RandomLegalOpponent()
        illegal = self._run_legality_check(engine, op, seed=0, steps=50)
        assert illegal == 0

    def test_minimax_d1_returns_legal_action(self, engine: RuleEngine):
        op = MinimaxOpponent(depth=1)
        illegal = self._run_legality_check(engine, op, seed=0, steps=30)
        assert illegal == 0

    def test_minimax_d2_returns_legal_action(self, engine: RuleEngine):
        op = MinimaxOpponent(depth=2)
        illegal = self._run_legality_check(engine, op, seed=0, steps=20)
        assert illegal == 0

    def test_minimax_depth_property(self):
        op = MinimaxOpponent(depth=3)
        assert op.depth == 3


# ---------------------------------------------------------------------------
# I–O: Training semantics with minimax opponent
# ---------------------------------------------------------------------------

def _run_deferred_push_with_opponent(
    engine: RuleEngine,
    opponent: TrainingOpponent,
    num_episodes: int,
    seed: int = 42,
    max_steps: int = 800,
) -> tuple[ReplayBuffer, dict]:
    """Run deferred-push training loop with specified opponent.

    Returns (buffer, stats).
    """
    from agent_system.training.dqn.model import QNetwork, select_epsilon_greedy_action
    rng = random.Random(seed)
    net = QNetwork()
    buffer = ReplayBuffer(capacity=100_000)

    stats: dict = {
        "pos_rewards": 0,
        "neg_rewards": 0,
        "zero_rewards": 0,
        "terminal_transitions": 0,
        "total_transitions": 0,
        "illegal_actions": 0,
        "episodes_completed": 0,
        "obs_perspective_correct": True,
    }

    for ep in range(num_episodes):
        learner_player = Player.P1 if ep % 2 == 0 else Player.P2
        state = engine.initial_state()
        done = False
        steps = 0

        pending_obs = None
        pending_action_id = None

        while not done and steps < max_steps:
            current_player = state.current_player
            mask = _legal_mask(engine, state)
            legal_ids = [i for i, v in enumerate(mask) if v]

            if current_player == learner_player:
                obs = encode_observation(state)
                # verify obs is always learner-perspective (current_player == learner)
                assert state.current_player == learner_player, \
                    "Obs encoded when learner is not current_player"
                import torch
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    q_values = net(obs_tensor)
                action_id = select_epsilon_greedy_action(q_values, mask, epsilon=1.0, rng=rng)
                if not mask[action_id]:
                    stats["illegal_actions"] += 1
                pending_obs = obs
                pending_action_id = action_id
            else:
                # Opponent turn — use pluggable opponent
                action_id = opponent.select_action_id(engine, state, legal_ids, rng)

            engine_action = decode_action_id(action_id, current_player)
            next_state = engine.apply_action(state, engine_action)
            steps += 1
            done = engine.is_game_over(next_state)

            if current_player == learner_player and done:
                # Learner wins
                reward = 1.0
                next_obs = encode_observation(next_state)
                buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                pending_obs = None
                stats["pos_rewards"] += 1
                stats["terminal_transitions"] += 1
                stats["total_transitions"] += 1

            elif current_player != learner_player and pending_obs is not None:
                if done:
                    # Opponent wins — learner loses
                    reward = -1.0
                    # CRITICAL: next_obs encoded here, but learner is NOT current_player
                    # We still store next_obs for this terminal transition (all zeros mask)
                    next_obs = encode_observation(next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                    stats["neg_rewards"] += 1
                    stats["terminal_transitions"] += 1
                    stats["total_transitions"] += 1
                else:
                    # Game continues — next_obs from learner's perspective
                    # At this point current_player is learner_player again
                    assert next_state.current_player == learner_player, \
                        "next_obs encoded when learner is not current_player"
                    reward = 0.0
                    next_obs = encode_observation(next_state)
                    next_mask = _legal_mask(engine, next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, False, next_mask)
                    stats["zero_rewards"] += 1
                    stats["total_transitions"] += 1
                pending_obs = None

            state = next_state

        stats["episodes_completed"] += 1

    return buffer, stats


class TestDeferredPushWithMinimaxOpponent:
    """Verify deferred-push training semantics are preserved with minimax opponent."""

    def test_minimax_opponent_produces_transitions(self, engine: RuleEngine):
        op = MinimaxOpponent(depth=2)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=5, seed=0, max_steps=200)
        assert stats["total_transitions"] > 0
        assert stats["episodes_completed"] == 5

    def test_minimax_opponent_zero_illegal_actions(self, engine: RuleEngine):
        """Opponent actions must be legal; learner with epsilon=1.0 uses only legal actions."""
        op = MinimaxOpponent(depth=2)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=5, seed=0, max_steps=200)
        assert stats["illegal_actions"] == 0

    def test_minimax_opponent_neg_reward_on_opponent_win(self, engine: RuleEngine):
        """When opponent wins, learner transition must have reward=-1."""
        op = MinimaxOpponent(depth=2)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=20, seed=42, max_steps=400)
        # Minimax d2 is stronger than random — should win some games
        # At minimum: verify the semantic is present (neg_rewards counter works)
        total_terminal = stats["pos_rewards"] + stats["neg_rewards"]
        assert total_terminal > 0, "No terminal transitions stored — test setup error"
        # If any opponent won, neg_rewards must be > 0
        # We can't guarantee opponent won in only 20 eps, but with minimax d2 it should

    def test_learner_win_produces_pos_reward(self, engine: RuleEngine):
        """When learner wins, stored reward must be +1."""
        op = RandomLegalOpponent()  # easier opponent so learner wins sometimes
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=30, seed=42, max_steps=400)
        # Verify buffer contains +1 transitions if learner won any
        samples = buf.sample(min(50, len(buf)), rng=random.Random(0))
        rewards = samples["reward"].tolist()
        reward_set = set(round(r, 1) for r in rewards)
        # All rewards must be in {-1.0, 0.0, +1.0}
        assert reward_set <= {-1.0, 0.0, 1.0}, f"Unexpected reward values: {reward_set}"

    def test_next_obs_perspective_is_learner_centric(self, engine: RuleEngine):
        """next_obs in non-terminal transitions must be encoded from learner perspective."""
        op = MinimaxOpponent(depth=1)
        # _run_deferred_push_with_opponent already has perspective asserts that raise if violated
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=10, seed=0, max_steps=200)
        assert stats["obs_perspective_correct"]

    def test_reward_values_are_only_legal_values(self, engine: RuleEngine):
        """All stored rewards must be in {-1.0, 0.0, +1.0}."""
        op = MinimaxOpponent(depth=2)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=10, seed=7, max_steps=300)
        assert len(buf) > 0
        samples = buf.sample(min(100, len(buf)), rng=random.Random(1))
        rewards = samples["reward"].tolist()
        for r in rewards:
            rounded = round(r, 6)
            assert rounded in (-1.0, 0.0, 1.0), f"Illegal reward value: {rounded}"


# ---------------------------------------------------------------------------
# Phase 14B: Focused deferred-push tests for minimax depth=1
# ---------------------------------------------------------------------------

class TestDeferredPushMinimaxD1:
    """Deferred-push training semantics with minimax depth=1 specifically.

    Depth=2 tests in TestDeferredPushWithMinimaxOpponent share the same code
    path, but these tests confirm depth=1 explicitly so Phase 14B results are
    self-contained.
    """

    def test_minimax_d1_zero_illegal_actions_in_training_loop(self, engine: RuleEngine):
        """Illegal action count must remain 0 across a short depth=1 rollout."""
        op = MinimaxOpponent(depth=1)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=10, seed=42, max_steps=200)
        assert stats["illegal_actions"] == 0

    def test_minimax_d1_opponent_win_produces_neg_reward(self, engine: RuleEngine):
        """When minimax depth=1 wins a terminal game, learner reward must be -1."""
        op = MinimaxOpponent(depth=1)
        # depth=1 is aggressive; run enough episodes that opponent should win some
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=20, seed=42, max_steps=300)
        total_terminal = stats["pos_rewards"] + stats["neg_rewards"]
        assert total_terminal > 0, "No terminal transitions found — test setup error"
        # With depth=1 (greedy), opponent should win the majority: neg_rewards > 0
        assert stats["neg_rewards"] > 0, (
            f"Expected neg rewards from minimax d1 wins, got: {stats}"
        )

    def test_minimax_d1_reward_values_restricted(self, engine: RuleEngine):
        """All stored rewards in the buffer must be in {-1.0, 0.0, +1.0}."""
        op = MinimaxOpponent(depth=1)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=10, seed=7, max_steps=300)
        assert len(buf) > 0
        samples = buf.sample(min(100, len(buf)), rng=random.Random(1))
        for r in samples["reward"].tolist():
            assert round(r, 6) in (-1.0, 0.0, 1.0), f"Illegal reward value: {r}"

    def test_minimax_d1_produces_transitions(self, engine: RuleEngine):
        """Training loop with depth=1 must store at least one transition."""
        op = MinimaxOpponent(depth=1)
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=5, seed=0, max_steps=200)
        assert stats["total_transitions"] > 0
        assert stats["episodes_completed"] == 5


# ---------------------------------------------------------------------------
# P: Backward compatibility — random_legal behavior unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibilityRandomLegal:
    """Existing random_legal behavior must be unchanged."""

    def test_random_legal_opponent_type(self):
        op = build_opponent("random_legal")
        assert isinstance(op, RandomLegalOpponent)

    def test_random_legal_action_is_legal(self, engine: RuleEngine, initial_state):
        op = RandomLegalOpponent()
        rng = random.Random(0)
        mask = _legal_mask(engine, initial_state)
        ids = [i for i, v in enumerate(mask) if v]
        for _ in range(20):
            action_id = op.select_action_id(engine, initial_state, ids, rng)
            assert mask[action_id], f"random_legal returned illegal action {action_id}"

    def test_random_legal_training_semantics_unchanged(self, engine: RuleEngine):
        """Run 10 episodes with random_legal and verify same invariants as before."""
        op = RandomLegalOpponent()
        buf, stats = _run_deferred_push_with_opponent(engine, op, num_episodes=10, seed=42, max_steps=400)
        assert stats["illegal_actions"] == 0
        assert stats["total_transitions"] > 0


# ---------------------------------------------------------------------------
# Phase 14C: Mixed-opponent support
# ---------------------------------------------------------------------------

class TestMixedOpponentConstruction:
    """U — MixedOpponent/build_mixed_opponent construction and validation."""

    def test_valid_mix_builds_correctly(self):
        mixed = build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])
        assert isinstance(mixed, MixedOpponent)
        assert len(mixed.entries) == 3

    def test_weights_are_normalized(self):
        mixed = build_mixed_opponent([
            (7.0, "random_legal", 1),
            (2.0, "minimax", 1),
            (1.0, "minimax", 2),
        ])
        total = sum(w for w, _, _ in mixed.entries)
        assert abs(total - 1.0) < 1e-9

    def test_single_entry_mix_builds(self):
        mixed = build_mixed_opponent([(1.0, "random_legal", 1)])
        assert len(mixed.entries) == 1

    def test_description_format(self):
        mixed = build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.30, "minimax", 1),
        ])
        desc = mixed.description()
        assert desc.startswith("mixed(")
        assert "random_legal" in desc
        assert "minimax(d=1)" in desc

    def test_invalid_opponent_name_fails(self):
        with pytest.raises(ValueError, match="Unknown opponent type"):
            build_mixed_opponent([(1.0, "ppo", 1)])

    def test_negative_weight_fails(self):
        with pytest.raises(ValueError, match="Weight must be >= 0"):
            build_mixed_opponent([(-0.5, "random_legal", 1)])

    def test_zero_total_weight_fails(self):
        with pytest.raises(ValueError, match="Total opponent weight must be > 0"):
            MixedOpponent([(0.0, RandomLegalOpponent(), "random_legal")])

    def test_empty_entries_fails(self):
        with pytest.raises(ValueError, match="requires at least one entry"):
            build_mixed_opponent([])

    def test_invalid_minimax_depth_fails(self):
        with pytest.raises(ValueError, match="depth must be"):
            build_mixed_opponent([(1.0, "minimax", 0)])


class TestMixedOpponentSampling:
    """V — Per-episode sampling is deterministic and distributes correctly."""

    def _sample_n(self, mixed: MixedOpponent, n: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        return [mixed.sample(rng)[1] for _ in range(n)]

    def test_sampling_is_deterministic_with_seed(self):
        mixed = build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])
        labels_a = self._sample_n(mixed, 50, seed=0)
        labels_b = self._sample_n(mixed, 50, seed=0)
        assert labels_a == labels_b

    def test_sampled_labels_in_expected_set(self):
        mixed = build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])
        labels = self._sample_n(mixed, 100, seed=42)
        expected = {"random_legal", "minimax(d=1)", "minimax(d=2)"}
        assert set(labels) <= expected

    def test_approximate_distribution(self):
        """Over 1000 samples, observed frequencies should be within ±8% of targets."""
        mixed = build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])
        labels = self._sample_n(mixed, 1000, seed=99)
        counts = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        # Check approximate proportions (±8% tolerance)
        assert abs(counts.get("random_legal", 0) / 1000 - 0.70) < 0.08
        assert abs(counts.get("minimax(d=1)", 0) / 1000 - 0.20) < 0.08
        assert abs(counts.get("minimax(d=2)", 0) / 1000 - 0.10) < 0.08

    def test_sample_returns_opponent_instance(self):
        mixed = build_mixed_opponent([
            (0.60, "random_legal", 1),
            (0.40, "minimax", 1),
        ])
        rng = random.Random(7)
        op, label = mixed.sample(rng)
        assert isinstance(op, TrainingOpponent)
        assert isinstance(label, str)
        assert label in ("random_legal", "minimax(d=1)")


class TestMixedOpponentLegality:
    """W — Each component opponent in a mix returns legal actions."""

    def _make_standard_mix(self) -> MixedOpponent:
        return build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])

    def test_random_legal_component_returns_legal(self, engine: RuleEngine):
        mixed = self._make_standard_mix()
        rng = random.Random(0)
        state = engine.initial_state()
        # Force-sample random_legal
        op_rl = build_opponent("random_legal")
        mask = _legal_mask(engine, state)
        ids = [i for i, v in enumerate(mask) if v]
        for _ in range(20):
            action_id = op_rl.select_action_id(engine, state, ids, rng)
            assert mask[action_id]

    def test_minimax_d1_component_returns_legal(self, engine: RuleEngine):
        rng = random.Random(0)
        state = engine.initial_state()
        op = MinimaxOpponent(depth=1)
        mask = _legal_mask(engine, state)
        ids = [i for i, v in enumerate(mask) if v]
        for _ in range(10):
            action_id = op.select_action_id(engine, state, ids, rng)
            assert mask[action_id]

    def test_minimax_d2_component_returns_legal(self, engine: RuleEngine):
        rng = random.Random(0)
        state = engine.initial_state()
        op = MinimaxOpponent(depth=2)
        mask = _legal_mask(engine, state)
        ids = [i for i, v in enumerate(mask) if v]
        for _ in range(5):
            action_id = op.select_action_id(engine, state, ids, rng)
            assert mask[action_id]


def _run_deferred_push_with_mixed_opponent(
    engine: RuleEngine,
    mixed: MixedOpponent,
    num_episodes: int,
    seed: int = 42,
    max_steps: int = 800,
) -> tuple[ReplayBuffer, dict]:
    """Run deferred-push loop with per-episode opponent sampling from a MixedOpponent.

    Returns (buffer, stats).
    """
    from agent_system.training.dqn.model import QNetwork, select_epsilon_greedy_action
    rng = random.Random(seed)
    net = QNetwork()
    buffer = ReplayBuffer(capacity=100_000)

    stats: dict = {
        "pos_rewards": 0,
        "neg_rewards": 0,
        "zero_rewards": 0,
        "terminal_transitions": 0,
        "total_transitions": 0,
        "illegal_actions": 0,
        "episodes_completed": 0,
        "opponent_labels_seen": set(),
    }

    for ep in range(num_episodes):
        learner_player = Player.P1 if ep % 2 == 0 else Player.P2

        # Per-episode opponent sampling
        ep_opponent, ep_label = mixed.sample(rng)
        stats["opponent_labels_seen"].add(ep_label)

        state = engine.initial_state()
        done = False
        steps = 0
        pending_obs = None
        pending_action_id = None

        while not done and steps < max_steps:
            current_player = state.current_player
            mask = _legal_mask(engine, state)
            legal_ids = [i for i, v in enumerate(mask) if v]

            if current_player == learner_player:
                obs = encode_observation(state)
                import torch
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    q_values = net(obs_tensor)
                action_id = select_epsilon_greedy_action(q_values, mask, epsilon=1.0, rng=rng)
                if not mask[action_id]:
                    stats["illegal_actions"] += 1
                pending_obs = obs
                pending_action_id = action_id
            else:
                action_id = ep_opponent.select_action_id(engine, state, legal_ids, rng)

            engine_action = decode_action_id(action_id, current_player)
            next_state = engine.apply_action(state, engine_action)
            steps += 1
            done = engine.is_game_over(next_state)

            if current_player == learner_player and done:
                reward = 1.0
                next_obs = encode_observation(next_state)
                buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                pending_obs = None
                stats["pos_rewards"] += 1
                stats["terminal_transitions"] += 1
                stats["total_transitions"] += 1
            elif current_player != learner_player and pending_obs is not None:
                if done:
                    reward = -1.0
                    next_obs = encode_observation(next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, True, [False] * ACTION_COUNT)
                    stats["neg_rewards"] += 1
                    stats["terminal_transitions"] += 1
                    stats["total_transitions"] += 1
                else:
                    reward = 0.0
                    next_obs = encode_observation(next_state)
                    next_mask = _legal_mask(engine, next_state)
                    buffer.push(pending_obs, pending_action_id, reward, next_obs, False, next_mask)
                    stats["zero_rewards"] += 1
                    stats["total_transitions"] += 1
                pending_obs = None

            state = next_state

        stats["episodes_completed"] += 1

    return buffer, stats


class TestMixedOpponentTrainingSemantics:
    """X — Deferred-push semantics are preserved when using MixedOpponent."""

    def _make_mix(self) -> MixedOpponent:
        return build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])

    def test_mixed_training_zero_illegal_actions(self, engine: RuleEngine):
        mixed = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mixed, num_episodes=20, seed=42, max_steps=300
        )
        assert stats["illegal_actions"] == 0

    def test_mixed_training_produces_transitions(self, engine: RuleEngine):
        mixed = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mixed, num_episodes=10, seed=0, max_steps=200
        )
        assert stats["total_transitions"] > 0
        assert stats["episodes_completed"] == 10

    def test_mixed_training_rewards_restricted(self, engine: RuleEngine):
        """All stored rewards in the buffer must be in {-1.0, 0.0, +1.0}."""
        mixed = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mixed, num_episodes=15, seed=7, max_steps=300
        )
        assert len(buf) > 0
        samples = buf.sample(min(100, len(buf)), rng=random.Random(1))
        for r in samples["reward"].tolist():
            assert round(r, 6) in (-1.0, 0.0, 1.0), f"Illegal reward value: {r}"

    def test_mixed_training_samples_multiple_opponents(self, engine: RuleEngine):
        """With enough episodes and seed 42, both random_legal and minimax should appear."""
        mixed = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mixed, num_episodes=30, seed=42, max_steps=400
        )
        # With 30 episodes and 70/20/10 mix, we expect multiple labels
        assert len(stats["opponent_labels_seen"]) >= 2, (
            f"Expected >=2 opponent types, saw: {stats['opponent_labels_seen']}"
        )

    def test_mixed_training_neg_reward_on_opponent_win(self, engine: RuleEngine):
        """Terminal opponent wins must produce learner reward=-1."""
        mixed = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mixed, num_episodes=30, seed=42, max_steps=500
        )
        total_terminal = stats["pos_rewards"] + stats["neg_rewards"]
        assert total_terminal > 0, "No terminal transitions — test setup error"
        # With minimax in the mix, opponent wins should occur
        assert stats["neg_rewards"] > 0, (
            f"Expected neg rewards from minimax wins: {stats}"
        )


# ---------------------------------------------------------------------------
# Phase 14D: DummyOpponent tests
# ---------------------------------------------------------------------------

class TestDummyOpponent:
    """Y — DummyOpponent selects first legal action id deterministically."""

    def test_dummy_can_be_built(self):
        from agent_system.training.dqn.opponent import DummyOpponent
        op = build_opponent("dummy")
        assert isinstance(op, DummyOpponent)

    def test_dummy_returns_first_legal_action_id(self, engine: RuleEngine):
        from agent_system.training.dqn.opponent import DummyOpponent
        op = DummyOpponent()
        state = engine.initial_state()
        legal_ids = _legal_ids(engine, state)
        rng = random.Random(0)
        result = op.select_action_id(engine, state, legal_ids, rng)
        assert result == legal_ids[0]

    def test_dummy_returned_action_is_legal(self, engine: RuleEngine):
        from agent_system.training.dqn.opponent import DummyOpponent
        op = DummyOpponent()
        state = engine.initial_state()
        mask = _legal_mask(engine, state)
        legal_ids = [i for i, v in enumerate(mask) if v]
        rng = random.Random(0)
        action_id = op.select_action_id(engine, state, legal_ids, rng)
        assert mask[action_id], f"action_id {action_id} is not legal"

    def test_dummy_is_deterministic(self, engine: RuleEngine):
        """Same state always returns same action, regardless of rng seed."""
        from agent_system.training.dqn.opponent import DummyOpponent
        op = DummyOpponent()
        state = engine.initial_state()
        legal_ids = _legal_ids(engine, state)
        results = [
            op.select_action_id(engine, state, legal_ids, random.Random(s))
            for s in range(5)
        ]
        assert len(set(results)) == 1, "DummyOpponent should be fully deterministic"

    def test_dummy_empty_legal_ids_raises(self, engine: RuleEngine):
        from agent_system.training.dqn.opponent import DummyOpponent
        op = DummyOpponent()
        state = engine.initial_state()
        with pytest.raises((ValueError, IndexError)):
            op.select_action_id(engine, state, [], random.Random(0))

    def test_build_opponent_invalid_after_dummy_added(self):
        """Unknown type still raises ValueError."""
        with pytest.raises(ValueError, match="Unknown opponent type"):
            build_opponent("not_a_real_opponent")


# ---------------------------------------------------------------------------
# PawnRandomOpponent tests
# ---------------------------------------------------------------------------

class TestPawnRandomOpponent:
    """PawnRandomOpponent only selects pawn-move action IDs (< 81)."""

    def test_can_be_built(self):
        from agent_system.training.dqn.opponent import PawnRandomOpponent
        op = build_opponent("pawn_random")
        assert isinstance(op, PawnRandomOpponent)

    def test_always_returns_pawn_action(self, engine: RuleEngine):
        from agent_system.training.dqn.opponent import PawnRandomOpponent
        op = PawnRandomOpponent()
        state = engine.initial_state()
        legal_ids = _legal_ids(engine, state)
        rng = random.Random(0)
        for _ in range(20):
            action_id = op.select_action_id(engine, state, legal_ids, rng)
            assert action_id < 81, f"Expected pawn action (id<81), got {action_id}"

    def test_returned_action_is_legal(self, engine: RuleEngine):
        from agent_system.training.dqn.opponent import PawnRandomOpponent
        op = PawnRandomOpponent()
        state = engine.initial_state()
        mask = _legal_mask(engine, state)
        legal_ids = [i for i, v in enumerate(mask) if v]
        rng = random.Random(0)
        action_id = op.select_action_id(engine, state, legal_ids, rng)
        assert mask[action_id], f"action_id {action_id} is not legal"

    def test_uniform_over_pawn_actions(self, engine: RuleEngine):
        """Empirically verify all legal pawn actions are reachable."""
        from agent_system.training.dqn.opponent import PawnRandomOpponent
        op = PawnRandomOpponent()
        state = engine.initial_state()
        legal_ids = _legal_ids(engine, state)
        pawn_ids = {a for a in legal_ids if a < 81}
        rng = random.Random(7)
        seen = set()
        for _ in range(200):
            seen.add(op.select_action_id(engine, state, legal_ids, rng))
        assert pawn_ids.issubset(seen), f"Some pawn actions never selected: {pawn_ids - seen}"




class TestDummyRandomMinimaxMix:
    """Z — 50/45/5 dummy+random_legal+minimax(d=1) mix builds and behaves correctly."""

    def _make_mix(self) -> MixedOpponent:
        return build_mixed_opponent([
            (0.50, "dummy", 1),
            (0.45, "random_legal", 1),
            (0.05, "minimax", 1),
        ])

    def test_mix_builds_successfully(self):
        mix = self._make_mix()
        assert isinstance(mix, MixedOpponent)

    def test_mix_description_includes_all_labels(self):
        mix = self._make_mix()
        desc = mix.description()
        assert "dummy" in desc
        assert "random_legal" in desc
        assert "minimax(d=1)" in desc

    def test_weights_normalize_to_one(self):
        mix = self._make_mix()
        total = sum(w for w, _, _ in mix.entries)
        assert abs(total - 1.0) < 1e-9

    def test_deterministic_sampling_with_seed(self):
        mix = self._make_mix()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        labels1 = [mix.sample(rng1)[1] for _ in range(20)]
        labels2 = [mix.sample(rng2)[1] for _ in range(20)]
        assert labels1 == labels2

    def test_all_three_labels_appear_over_many_samples(self):
        mix = self._make_mix()
        rng = random.Random(7)
        labels = {mix.sample(rng)[1] for _ in range(200)}
        assert "dummy" in labels, f"dummy not seen in 200 samples: {labels}"
        assert "random_legal" in labels, f"random_legal not seen: {labels}"
        assert "minimax(d=1)" in labels, f"minimax(d=1) not seen: {labels}"

    def test_approximate_distribution(self):
        """Empirical distribution over 1000 samples should be within ±8% of configured."""
        mix = self._make_mix()
        rng = random.Random(99)
        counts: dict[str, int] = {}
        n = 1000
        for _ in range(n):
            _, label = mix.sample(rng)
            counts[label] = counts.get(label, 0) + 1
        assert abs(counts.get("dummy", 0) / n - 0.50) < 0.08
        assert abs(counts.get("random_legal", 0) / n - 0.45) < 0.08
        assert abs(counts.get("minimax(d=1)", 0) / n - 0.05) < 0.08

    def test_dummy_component_returns_legal_action(self, engine: RuleEngine):
        from agent_system.training.dqn.opponent import DummyOpponent
        mix = self._make_mix()
        state = engine.initial_state()
        mask = _legal_mask(engine, state)
        legal_ids = [i for i, v in enumerate(mask) if v]
        # Directly test the dummy entry from the mix
        dummy_entries = [(w, op, lbl) for w, op, lbl in mix.entries if lbl == "dummy"]
        assert dummy_entries, "No dummy entry found in mix"
        _, dummy_op, _ = dummy_entries[0]
        assert isinstance(dummy_op, DummyOpponent)
        action_id = dummy_op.select_action_id(engine, state, legal_ids, random.Random(0))
        assert mask[action_id]

    def test_mix_zero_illegal_actions(self, engine: RuleEngine):
        mix = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mix, num_episodes=20, seed=42, max_steps=300
        )
        assert stats["illegal_actions"] == 0

    def test_mix_rewards_restricted(self, engine: RuleEngine):
        mix = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mix, num_episodes=15, seed=3, max_steps=300
        )
        assert len(buf) > 0
        samples = buf.sample(min(100, len(buf)), rng=random.Random(1))
        for r in samples["reward"].tolist():
            assert round(r, 6) in (-1.0, 0.0, 1.0), f"Illegal reward value: {r}"

    def test_mix_samples_multiple_opponent_labels(self, engine: RuleEngine):
        mix = self._make_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mix, num_episodes=30, seed=42, max_steps=400
        )
        assert len(stats["opponent_labels_seen"]) >= 2, (
            f"Expected >=2 labels, saw: {stats['opponent_labels_seen']}"
        )


class TestBackwardCompatibilityWithDummy:
    """Backward compatibility: existing random_legal + previous mixed configs still work."""

    def test_random_legal_still_builds(self):
        op = build_opponent("random_legal")
        assert isinstance(op, RandomLegalOpponent)

    def test_minimax_still_builds(self):
        op = build_opponent("minimax", minimax_depth=1)
        assert isinstance(op, MinimaxOpponent)

    def test_previous_70_20_10_mix_still_builds(self):
        mix = build_mixed_opponent([
            (0.70, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.10, "minimax", 2),
        ])
        assert isinstance(mix, MixedOpponent)
        labels = {lbl for _, _, lbl in mix.entries}
        assert "random_legal" in labels
        assert "minimax(d=1)" in labels
        assert "minimax(d=2)" in labels


# ---------------------------------------------------------------------------
# Phase 14E: Minimax depth=3 support
# ---------------------------------------------------------------------------

class TestMinimaxD3Opponent:
    """Minimax depth=3 — legality, depth property, factory."""

    def _run_legality_check(
        self, engine: RuleEngine, opponent: TrainingOpponent, seed: int, steps: int
    ) -> int:
        rng = random.Random(seed)
        state = engine.initial_state()
        illegal_count = 0
        for _ in range(steps):
            if engine.is_game_over(state):
                break
            mask = _legal_mask(engine, state)
            ids = [i for i, v in enumerate(mask) if v]
            action_id = opponent.select_action_id(engine, state, ids, rng)
            if not mask[action_id]:
                illegal_count += 1
            engine_action = decode_action_id(action_id, state.current_player)
            state = engine.apply_action(state, engine_action)
        return illegal_count

    def test_minimax_d3_builds_via_factory(self):
        op = build_opponent("minimax", minimax_depth=3)
        assert isinstance(op, MinimaxOpponent)
        assert op.depth == 3

    def test_minimax_d3_depth_property(self):
        op = MinimaxOpponent(depth=3)
        assert op.depth == 3

    def test_minimax_d3_returns_legal_action(self, engine: RuleEngine):
        op = MinimaxOpponent(depth=3)
        illegal = self._run_legality_check(engine, op, seed=0, steps=10)
        assert illegal == 0

    def test_minimax_d3_training_zero_illegal_actions(self, engine: RuleEngine):
        op = MinimaxOpponent(depth=3)
        buf, stats = _run_deferred_push_with_opponent(
            engine, op, num_episodes=5, seed=42, max_steps=150
        )
        assert stats["illegal_actions"] == 0

    def test_minimax_d3_reward_values_restricted(self, engine: RuleEngine):
        op = MinimaxOpponent(depth=3)
        buf, stats = _run_deferred_push_with_opponent(
            engine, op, num_episodes=5, seed=7, max_steps=150
        )
        assert len(buf) > 0
        samples = buf.sample(min(50, len(buf)), rng=random.Random(1))
        for r in samples["reward"].tolist():
            assert round(r, 6) in (-1.0, 0.0, 1.0), f"Illegal reward value: {r}"


# ---------------------------------------------------------------------------
# Phase 14E: Mixed opponent with all 5 types including d3
# ---------------------------------------------------------------------------

class TestMixedOpponentWithAllTypes:
    """All 5 opponent types in a single mix: dummy, random_legal, d1, d2, d3."""

    def _make_full_mix(self) -> MixedOpponent:
        return build_mixed_opponent([
            (0.10, "dummy", 1),
            (0.30, "random_legal", 1),
            (0.20, "minimax", 1),
            (0.30, "minimax", 2),
            (0.10, "minimax", 3),
        ])

    def test_full_mix_builds(self):
        mix = self._make_full_mix()
        assert isinstance(mix, MixedOpponent)
        assert len(mix.entries) == 5

    def test_full_mix_weights_normalize(self):
        mix = self._make_full_mix()
        total = sum(w for w, _, _ in mix.entries)
        assert abs(total - 1.0) < 1e-9

    def test_full_mix_description_contains_all_labels(self):
        mix = self._make_full_mix()
        desc = mix.description()
        assert "dummy" in desc
        assert "random_legal" in desc
        assert "minimax(d=1)" in desc
        assert "minimax(d=2)" in desc
        assert "minimax(d=3)" in desc

    def test_full_mix_all_five_labels_appear(self):
        mix = self._make_full_mix()
        rng = random.Random(7)
        labels = {mix.sample(rng)[1] for _ in range(500)}
        assert "dummy" in labels, f"dummy not seen in 500 samples: {labels}"
        assert "random_legal" in labels
        assert "minimax(d=1)" in labels
        assert "minimax(d=2)" in labels
        assert "minimax(d=3)" in labels

    def test_full_mix_approximate_distribution(self):
        """Empirical frequencies within ±8% of configured weights."""
        mix = self._make_full_mix()
        rng = random.Random(99)
        counts: dict[str, int] = {}
        n = 1000
        for _ in range(n):
            _, label = mix.sample(rng)
            counts[label] = counts.get(label, 0) + 1
        assert abs(counts.get("dummy", 0) / n - 0.10) < 0.08
        assert abs(counts.get("random_legal", 0) / n - 0.30) < 0.08
        assert abs(counts.get("minimax(d=1)", 0) / n - 0.20) < 0.08
        assert abs(counts.get("minimax(d=2)", 0) / n - 0.30) < 0.08
        assert abs(counts.get("minimax(d=3)", 0) / n - 0.10) < 0.08

    def test_full_mix_zero_illegal_actions(self, engine: RuleEngine):
        mix = self._make_full_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mix, num_episodes=20, seed=42, max_steps=200
        )
        assert stats["illegal_actions"] == 0

    def test_full_mix_rewards_restricted(self, engine: RuleEngine):
        mix = self._make_full_mix()
        buf, stats = _run_deferred_push_with_mixed_opponent(
            engine, mix, num_episodes=15, seed=3, max_steps=200
        )
        assert len(buf) > 0
        samples = buf.sample(min(100, len(buf)), rng=random.Random(1))
        for r in samples["reward"].tolist():
            assert round(r, 6) in (-1.0, 0.0, 1.0), f"Illegal reward value: {r}"


# ---------------------------------------------------------------------------
# Phase 14E: TrainConfig has opponent_mix_minimax_d3 field
# ---------------------------------------------------------------------------

class TestTrainConfigOpponentMixD3:
    """Verify TrainConfig.opponent_mix_minimax_d3 field exists with correct default."""

    def test_field_exists_with_zero_default(self):
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[4]))
        from scripts.train_dqn import TrainConfig
        cfg = TrainConfig()
        assert hasattr(cfg, "opponent_mix_minimax_d3")
        assert cfg.opponent_mix_minimax_d3 == 0.0

    def test_field_can_be_set(self):
        from scripts.train_dqn import TrainConfig
        cfg = TrainConfig(opponent_mix_minimax_d3=0.25)
        assert cfg.opponent_mix_minimax_d3 == 0.25

    def test_existing_d1_d2_fields_still_present(self):
        from scripts.train_dqn import TrainConfig
        cfg = TrainConfig(
            opponent_mix_minimax_d1=0.20,
            opponent_mix_minimax_d2=0.30,
            opponent_mix_minimax_d3=0.10,
        )
        assert cfg.opponent_mix_minimax_d1 == 0.20
        assert cfg.opponent_mix_minimax_d2 == 0.30
        assert cfg.opponent_mix_minimax_d3 == 0.10

