# TEST_CLASSIFICATION: SPECIFIED
"""Integration tests for the DQN training foundation (Phase 1 + Phase 2).

Verifies the full Engine -> ActionSpace -> LegalMask -> RLEnv -> Terminal
path using only random legal-action rollouts. No neural network or trainer
is involved here.

Covers:
    A. Random rollout completes a game via QuoridorEnv
    B. Legal action IDs across a full episode are all engine-valid
    C. Multiple seeds produce games that terminate
    D. Reward signal reaches +1.0 or -1.0 at terminal when learner is set
"""

from __future__ import annotations

import random

import pytest

from quoridor_engine import Player, RuleEngine

from agent_system.training.dqn.action_space import (
    ACTION_COUNT,
    decode_action_id,
    encode_engine_action,
)
from agent_system.training.dqn.env import QuoridorEnv


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


class TestRandomRolloutIntegration:
    """Full random-legal-action rollout to terminal via the RL environment."""

    def _rollout(self, engine: RuleEngine, seed: int) -> tuple[int, float | None]:
        """Return (steps, terminal_reward) for a complete episode."""
        rng = random.Random(seed)
        env = QuoridorEnv(engine, learner_player=Player.P1)
        env.reset()
        terminal_reward = None
        while not env.is_done:
            ids = env.legal_action_ids()
            assert len(ids) > 0, "No legal actions in non-terminal state"
            _, reward, done, _ = env.step(rng.choice(ids))
            if done:
                terminal_reward = reward
        return env.step_count, terminal_reward

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 2026])
    def test_random_rollout_terminates(self, engine: RuleEngine, seed: int) -> None:
        steps, _ = self._rollout(engine, seed)
        assert steps > 0

    @pytest.mark.parametrize("seed", [0, 42, 2026])
    def test_terminal_reward_is_plus_or_minus_one(
        self, engine: RuleEngine, seed: int
    ) -> None:
        _, reward = self._rollout(engine, seed)
        assert reward in (1.0, -1.0)

    def test_all_action_ids_used_during_rollout_are_valid(
        self, engine: RuleEngine
    ) -> None:
        """Every action_id used during a rollout is within [0, ACTION_COUNT)."""
        rng = random.Random(777)
        env = QuoridorEnv(engine)
        env.reset()
        used_ids: list[int] = []
        while not env.is_done:
            ids = env.legal_action_ids()
            chosen = rng.choice(ids)
            used_ids.append(chosen)
            env.step(chosen)
        assert all(0 <= aid < ACTION_COUNT for aid in used_ids)

    def test_encode_decode_roundtrip_for_all_actions_in_rollout(
        self, engine: RuleEngine
    ) -> None:
        """For every action taken in a rollout, encode -> decode -> re-encode is stable."""
        rng = random.Random(555)
        env = QuoridorEnv(engine)
        env.reset()
        while not env.is_done:
            ids = env.legal_action_ids()
            chosen_id = rng.choice(ids)
            # Use raw_state() to get the current player before stepping.
            player = env.raw_state().current_player
            decoded = decode_action_id(chosen_id, player)
            re_encoded = encode_engine_action(decoded)
            assert re_encoded == chosen_id, (
                f"Round-trip failed: original={chosen_id}, re-encoded={re_encoded}"
            )
            env.step(chosen_id)

    def test_legal_mask_count_matches_engine_legal_actions(
        self, engine: RuleEngine
    ) -> None:
        """At each step, the legal mask True-count must equal len(engine.legal_actions())."""
        rng = random.Random(333)
        env = QuoridorEnv(engine)
        env.reset()
        steps = 0
        while not env.is_done and steps < 20:
            mask = env.legal_action_mask()
            ids = env.legal_action_ids()
            engine_count = len(engine.legal_actions(env.current_state()))
            assert sum(mask) == engine_count
            assert len(ids) == engine_count
            env.step(rng.choice(ids))
            steps += 1
