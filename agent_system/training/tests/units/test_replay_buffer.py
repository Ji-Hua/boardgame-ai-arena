"""Unit tests for agent_system/training/dqn/replay_buffer.py — Phase 5.

Test groups
-----------
TestReplayBufferConstruction    — capacity, initial state
TestReplayBufferPush            — push transitions, length, ring wrap
TestReplayBufferSample          — sample batch, shapes, types, reproducibility
TestReplayBufferEdgeCases       — error conditions, clear, is_ready
"""

from __future__ import annotations

import random

import pytest
import torch

from agent_system.training.dqn.action_space import ACTION_COUNT
from agent_system.training.dqn.observation import OBSERVATION_SIZE
from agent_system.training.dqn.replay_buffer import (
    BATCH_KEYS,
    REPLAY_BUFFER_VERSION,
    ReplayBuffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_obs(val: float = 0.0) -> list[float]:
    return [val] * OBSERVATION_SIZE


def _all_legal_mask() -> list[bool]:
    return [True] * ACTION_COUNT


def _push_n(buf: ReplayBuffer, n: int, *, done_at_last: bool = False) -> None:
    """Push n distinct transitions into buf."""
    for i in range(n):
        done = done_at_last and (i == n - 1)
        buf.push(
            obs=_dummy_obs(float(i)),
            action_id=i % ACTION_COUNT,
            reward=float(i) * 0.1,
            next_obs=_dummy_obs(float(i) + 0.5),
            done=done,
            next_legal_mask=_all_legal_mask(),
        )


# ---------------------------------------------------------------------------
# TestReplayBufferConstruction
# ---------------------------------------------------------------------------

class TestReplayBufferConstruction:
    def test_version_constant(self):
        assert REPLAY_BUFFER_VERSION == "replay_buffer_v1"

    def test_capacity_attribute(self):
        buf = ReplayBuffer(capacity=100)
        assert buf.capacity == 100

    def test_initial_length_is_zero(self):
        buf = ReplayBuffer(capacity=50)
        assert len(buf) == 0

    def test_is_ready_false_when_empty(self):
        buf = ReplayBuffer(capacity=50)
        assert not buf.is_ready(1)

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0)

    def test_capacity_one_is_valid(self):
        buf = ReplayBuffer(capacity=1)
        assert buf.capacity == 1


# ---------------------------------------------------------------------------
# TestReplayBufferPush
# ---------------------------------------------------------------------------

class TestReplayBufferPush:
    def test_push_increments_length(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(_dummy_obs(), 0, 0.0, _dummy_obs(), False, _all_legal_mask())
        assert len(buf) == 1

    def test_push_multiple_increments(self):
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 5)
        assert len(buf) == 5

    def test_length_caps_at_capacity(self):
        buf = ReplayBuffer(capacity=4)
        _push_n(buf, 10)
        assert len(buf) == 4

    def test_capacity_not_exceeded(self):
        buf = ReplayBuffer(capacity=3)
        _push_n(buf, 100)
        assert len(buf) == 3

    def test_is_ready_after_enough_pushes(self):
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 5)
        assert buf.is_ready(5)

    def test_is_ready_false_before_min(self):
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 3)
        assert not buf.is_ready(4)

    def test_ring_wraps_oldest_entry(self):
        """After overwriting, the buffer should still have exactly capacity entries."""
        cap = 4
        buf = ReplayBuffer(capacity=cap)
        _push_n(buf, cap + 2)
        assert len(buf) == cap

    def test_push_done_true_stored(self):
        buf = ReplayBuffer(capacity=5)
        _push_n(buf, 3, done_at_last=True)
        batch = buf.sample(3, rng=random.Random(0))
        # At least one done=1.0 exists in the buffer
        assert batch["done"].sum().item() >= 1.0


# ---------------------------------------------------------------------------
# TestReplayBufferSample
# ---------------------------------------------------------------------------

class TestReplayBufferSample:
    def _buf(self, n: int = 20) -> ReplayBuffer:
        buf = ReplayBuffer(capacity=50)
        _push_n(buf, n)
        return buf

    def test_sample_returns_dict(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert isinstance(batch, dict)

    def test_sample_has_all_keys(self):
        buf = self._buf()
        batch = buf.sample(4)
        for key in BATCH_KEYS:
            assert key in batch, f"Missing key: {key}"

    def test_obs_shape(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, OBSERVATION_SIZE)

    def test_next_obs_shape(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert batch["next_obs"].shape == (4, OBSERVATION_SIZE)

    def test_action_shape(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert batch["action"].shape == (4,)

    def test_reward_shape(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert batch["reward"].shape == (4,)

    def test_done_shape(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert batch["done"].shape == (4,)

    def test_next_mask_shape(self):
        buf = self._buf()
        batch = buf.sample(4)
        assert batch["next_mask"].shape == (4, ACTION_COUNT)

    def test_obs_dtype_float32(self):
        buf = self._buf()
        assert buf.sample(4)["obs"].dtype == torch.float32

    def test_action_dtype_int64(self):
        buf = self._buf()
        assert buf.sample(4)["action"].dtype == torch.int64

    def test_reward_dtype_float32(self):
        buf = self._buf()
        assert buf.sample(4)["reward"].dtype == torch.float32

    def test_done_dtype_float32(self):
        buf = self._buf()
        assert buf.sample(4)["done"].dtype == torch.float32

    def test_next_mask_dtype_bool(self):
        buf = self._buf()
        assert buf.sample(4)["next_mask"].dtype == torch.bool

    def test_done_values_zero_or_one(self):
        buf = self._buf(20)
        batch = buf.sample(10)
        assert torch.all((batch["done"] == 0.0) | (batch["done"] == 1.0))

    def test_sample_batch_size_1(self):
        buf = self._buf()
        batch = buf.sample(1)
        assert batch["obs"].shape[0] == 1

    def test_sample_full_buffer(self):
        buf = self._buf(20)
        batch = buf.sample(20)
        assert batch["obs"].shape[0] == 20

    def test_reproducible_with_rng(self):
        buf = self._buf(30)
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        b1 = buf.sample(8, rng=rng1)
        b2 = buf.sample(8, rng=rng2)
        assert torch.equal(b1["action"], b2["action"])

    def test_different_rng_seeds_may_differ(self):
        buf = self._buf(30)
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        b1 = buf.sample(8, rng=rng1)
        b2 = buf.sample(8, rng=rng2)
        # Very unlikely to be equal for seed 1 vs 2 with 30 items
        # (not guaranteed but near-certain)
        assert not torch.equal(b1["action"], b2["action"])


# ---------------------------------------------------------------------------
# TestReplayBufferEdgeCases
# ---------------------------------------------------------------------------

class TestReplayBufferEdgeCases:
    def test_sample_more_than_stored_raises(self):
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 3)
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(4)

    def test_sample_from_empty_raises(self):
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(ValueError):
            buf.sample(1)

    def test_clear_resets_length(self):
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 5)
        buf.clear()
        assert len(buf) == 0

    def test_clear_then_push_works(self):
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 5)
        buf.clear()
        _push_n(buf, 3)
        assert len(buf) == 3

    def test_sample_exactly_capacity(self):
        buf = ReplayBuffer(capacity=5)
        _push_n(buf, 5)
        batch = buf.sample(5)
        assert batch["obs"].shape[0] == 5

    def test_capacity_one_push_and_sample(self):
        buf = ReplayBuffer(capacity=1)
        buf.push(_dummy_obs(1.0), 0, 1.0, _dummy_obs(2.0), True, _all_legal_mask())
        batch = buf.sample(1)
        assert batch["done"][0].item() == 1.0
