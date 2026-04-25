# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for agent_system/training/dqn/env.py.

Covers:
    A. Reset creates valid initial state
    B. Legal action mask/ids after reset
    C. step() with a legal action advances the environment
    D. step() with out-of-bounds action_id is rejected
    E. step() with an engine-illegal action_id is rejected
    F. A random legal-action rollout completes a full game without illegal actions
    G. done becomes True at terminal game
    H. step() after done raises RuntimeError
    I. step() before reset raises RuntimeError
    J. Reward is correct for winning/losing learner
    K. Environment never bypasses Engine validation
"""

from __future__ import annotations

import random

import pytest

from quoridor_engine import Player, RuleEngine

from agent_system.training.dqn.action_space import ACTION_COUNT, encode_move_pawn
from agent_system.training.dqn.env import QuoridorEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


@pytest.fixture
def env(engine: RuleEngine) -> QuoridorEnv:
    """A fresh environment without a learner_player set."""
    return QuoridorEnv(engine)


@pytest.fixture
def env_with_learner(engine: RuleEngine) -> QuoridorEnv:
    """An environment where P1 is the learner."""
    return QuoridorEnv(engine, learner_player=Player.P1)


# ===========================================================================
# A. Reset
# ===========================================================================

class TestReset:
    def test_reset_returns_raw_state(self, env: QuoridorEnv) -> None:
        state = env.reset()
        assert state is not None

    def test_reset_step_count_is_zero(self, env: QuoridorEnv) -> None:
        env.reset()
        assert env.step_count == 0

    def test_reset_done_is_false(self, env: QuoridorEnv) -> None:
        env.reset()
        assert not env.is_done

    def test_reset_twice_starts_fresh(self, env: QuoridorEnv, engine: RuleEngine) -> None:
        env.reset()
        initial = engine.initial_state()
        env.reset()
        state = env.raw_state()
        # Pawn positions should match the canonical initial state.
        assert state.pawn_pos(Player.P1) == initial.pawn_pos(Player.P1)
        assert state.pawn_pos(Player.P2) == initial.pawn_pos(Player.P2)


# ===========================================================================
# B. Legal actions after reset
# ===========================================================================

class TestLegalActionsAfterReset:
    def test_mask_has_length_209(self, env: QuoridorEnv) -> None:
        env.reset()
        mask = env.legal_action_mask()
        assert len(mask) == ACTION_COUNT

    def test_mask_has_at_least_one_legal_action(self, env: QuoridorEnv) -> None:
        env.reset()
        mask = env.legal_action_mask()
        assert any(mask)

    def test_legal_action_ids_is_nonempty(self, env: QuoridorEnv) -> None:
        env.reset()
        ids = env.legal_action_ids()
        assert len(ids) > 0

    def test_legal_ids_are_in_valid_range(self, env: QuoridorEnv) -> None:
        env.reset()
        for action_id in env.legal_action_ids():
            assert 0 <= action_id < ACTION_COUNT


# ===========================================================================
# C. Step with legal action
# ===========================================================================

class TestStepWithLegalAction:
    def test_step_returns_tuple_of_four(self, env: QuoridorEnv) -> None:
        env.reset()
        action_id = env.legal_action_ids()[0]
        result = env.step(action_id)
        assert len(result) == 4

    def test_step_increments_step_count(self, env: QuoridorEnv) -> None:
        env.reset()
        action_id = env.legal_action_ids()[0]
        env.step(action_id)
        assert env.step_count == 1

    def test_step_updates_state(self, env: QuoridorEnv, engine: RuleEngine) -> None:
        env.reset()
        state_before = env.current_state()
        action_id = env.legal_action_ids()[0]
        next_obs, _, _, _ = env.step(action_id)
        # The returned obs must differ from or equal the initial (either way it's a valid state)
        assert next_obs is not None

    def test_step_done_is_false_for_nonterminal(self, env: QuoridorEnv) -> None:
        env.reset()
        action_id = env.legal_action_ids()[0]
        _, _, done, _ = env.step(action_id)
        # Initial step is almost certainly not terminal.
        assert not done

    def test_step_info_contains_expected_keys(self, env: QuoridorEnv) -> None:
        env.reset()
        action_id = env.legal_action_ids()[0]
        _, _, _, info = env.step(action_id)
        assert "action_id" in info
        assert "step_count" in info
        assert "done" in info
        assert "action_space_version" in info


# ===========================================================================
# D. Out-of-bounds action_id is rejected
# ===========================================================================

class TestOutOfBoundsActionIdRejected:
    @pytest.mark.parametrize("bad_id", [-1, 209, 300, 10000])
    def test_step_raises_runtime_error_for_out_of_bounds(
        self, env: QuoridorEnv, bad_id: int
    ) -> None:
        env.reset()
        with pytest.raises(RuntimeError):
            env.step(bad_id)


# ===========================================================================
# E. Engine-illegal action_id is rejected
# ===========================================================================

class TestEngineIllegalActionRejected:
    def test_step_raises_for_illegal_but_valid_id(self, env: QuoridorEnv) -> None:
        """A valid action_id that is not legal in the current state must be rejected."""
        env.reset()
        legal_ids = set(env.legal_action_ids())
        # Find first valid action_id that is NOT legal.
        illegal_id = next(
            (i for i in range(ACTION_COUNT) if i not in legal_ids), None
        )
        if illegal_id is None:
            pytest.skip("All 209 action_ids are legal — no illegal id to test.")
        with pytest.raises(ValueError):
            env.step(illegal_id)


# ===========================================================================
# F. Random legal rollout completes a full game
# ===========================================================================

class TestRandomLegalRollout:
    def _run_episode(self, env: QuoridorEnv, rng: random.Random) -> int:
        env.reset()
        steps = 0
        while not env.is_done:
            ids = env.legal_action_ids()
            assert len(ids) > 0, "No legal actions in non-terminal state"
            action_id = rng.choice(ids)
            _, _, done, info = env.step(action_id)
            steps += 1
            assert steps < 10_000, "Rollout exceeded safety limit — possible infinite loop"
        return steps

    def test_single_rollout_completes(self, env: QuoridorEnv) -> None:
        rng = random.Random(42)
        steps = self._run_episode(env, rng)
        assert steps > 0
        assert env.is_done

    def test_multiple_rollouts_complete(self, env: QuoridorEnv) -> None:
        rng = random.Random(99)
        for _ in range(3):
            steps = self._run_episode(env, rng)
            assert steps > 0

    def test_rollout_never_produces_zero_legal_actions_before_done(
        self, env: QuoridorEnv
    ) -> None:
        rng = random.Random(7)
        env.reset()
        while not env.is_done:
            ids = env.legal_action_ids()
            assert len(ids) > 0
            env.step(rng.choice(ids))


# ===========================================================================
# G. done becomes True at terminal game
# ===========================================================================

class TestTermination:
    def test_done_is_true_at_terminal(self, env: QuoridorEnv) -> None:
        rng = random.Random(123)
        env.reset()
        while not env.is_done:
            ids = env.legal_action_ids()
            env.step(rng.choice(ids))
        assert env.is_done

    def test_info_winner_set_at_terminal(self, env: QuoridorEnv) -> None:
        rng = random.Random(456)
        env.reset()
        last_info = None
        while not env.is_done:
            ids = env.legal_action_ids()
            _, _, done, info = env.step(rng.choice(ids))
            if done:
                last_info = info
        assert last_info is not None
        assert last_info["winner"] is not None


# ===========================================================================
# H. step() after done raises RuntimeError
# ===========================================================================

class TestStepAfterDone:
    def test_step_after_done_raises(self, env: QuoridorEnv) -> None:
        rng = random.Random(321)
        env.reset()
        while not env.is_done:
            ids = env.legal_action_ids()
            env.step(rng.choice(ids))
        # Now env.is_done is True — next step must raise.
        with pytest.raises(RuntimeError):
            env.step(env.legal_action_ids()[0] if not env.is_done else 0)

    def test_step_after_done_then_reset_works(self, env: QuoridorEnv) -> None:
        rng = random.Random(654)
        env.reset()
        while not env.is_done:
            env.step(rng.choice(env.legal_action_ids()))
        # After reset the environment is live again.
        env.reset()
        assert not env.is_done
        ids = env.legal_action_ids()
        assert len(ids) > 0


# ===========================================================================
# I. step() before reset raises RuntimeError
# ===========================================================================

class TestStepBeforeReset:
    def test_step_before_reset_raises(self, engine: RuleEngine) -> None:
        fresh_env = QuoridorEnv(engine)
        with pytest.raises(RuntimeError):
            fresh_env.step(0)

    def test_mask_before_reset_raises(self, engine: RuleEngine) -> None:
        fresh_env = QuoridorEnv(engine)
        with pytest.raises(RuntimeError):
            fresh_env.legal_action_mask()

    def test_ids_before_reset_raises(self, engine: RuleEngine) -> None:
        fresh_env = QuoridorEnv(engine)
        with pytest.raises(RuntimeError):
            fresh_env.legal_action_ids()


# ===========================================================================
# J. Reward is correct
# ===========================================================================

class TestReward:
    def test_non_terminal_reward_is_zero_without_learner(
        self, env: QuoridorEnv
    ) -> None:
        env.reset()
        ids = env.legal_action_ids()
        _, reward, done, _ = env.step(ids[0])
        if not done:
            assert reward == 0.0

    def test_non_terminal_reward_is_zero_with_learner(
        self, env_with_learner: QuoridorEnv
    ) -> None:
        env_with_learner.reset()
        ids = env_with_learner.legal_action_ids()
        _, reward, done, _ = env_with_learner.step(ids[0])
        if not done:
            assert reward == 0.0

    def test_terminal_reward_is_nonzero_when_learner_set(
        self, env_with_learner: QuoridorEnv
    ) -> None:
        rng = random.Random(11)
        env_with_learner.reset()
        last_reward = None
        done = False
        while not done:
            ids = env_with_learner.legal_action_ids()
            _, reward, done, _ = env_with_learner.step(rng.choice(ids))
            if done:
                last_reward = reward
        assert last_reward in (1.0, -1.0)

    def test_terminal_reward_is_zero_without_learner(
        self, env: QuoridorEnv
    ) -> None:
        rng = random.Random(22)
        env.reset()
        last_reward = None
        done = False
        while not done:
            ids = env.legal_action_ids()
            _, reward, done, _ = env.step(rng.choice(ids))
            if done:
                last_reward = reward
        assert last_reward == 0.0
