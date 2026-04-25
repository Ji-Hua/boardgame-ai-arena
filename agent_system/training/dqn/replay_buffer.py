"""Fixed-capacity uniform-sampling replay buffer for DQN training.

Design:
    - Ring buffer with configurable capacity.
    - Stores raw Python / list types; converts to PyTorch tensors at sample time.
    - Each slot stores one transition:
        (obs, action_id, reward, next_obs, done, next_legal_mask)
    - Uniform random sampling — no prioritization.
    - Thread-safety is NOT guaranteed; single-trainer use only.

Public API:
    ReplayBuffer(capacity)
    buffer.push(obs, action_id, reward, next_obs, done, next_legal_mask)
    batch = buffer.sample(batch_size)        -> dict[str, torch.Tensor]
    len(buffer)                              -> int
    buffer.capacity                          -> int
    buffer.is_ready(min_size)               -> bool
    buffer.clear()

Sample batch keys (all torch.Tensor, float32 unless noted):
    "obs"           (batch_size, OBSERVATION_SIZE) float32
    "action"        (batch_size,)                  int64
    "reward"        (batch_size,)                  float32
    "next_obs"      (batch_size, OBSERVATION_SIZE) float32
    "done"          (batch_size,)                  float32  (0.0 or 1.0)
    "next_mask"     (batch_size, ACTION_COUNT)     bool
"""

from __future__ import annotations

import random as _random
from typing import Sequence

import torch

from agent_system.training.dqn.action_space import ACTION_COUNT
from agent_system.training.dqn.observation import OBSERVATION_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPLAY_BUFFER_VERSION: str = "replay_buffer_v1"

# Required keys in every sampled batch.
BATCH_KEYS: tuple[str, ...] = (
    "obs",
    "action",
    "reward",
    "next_obs",
    "done",
    "next_mask",
)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity uniform-sampling replay buffer for DQN transitions.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.  When full, the oldest
        transition is overwritten (ring buffer semantics).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._obs: list[list[float]] = [None] * capacity  # type: ignore[list-item]
        self._action: list[int] = [0] * capacity
        self._reward: list[float] = [0.0] * capacity
        self._next_obs: list[list[float]] = [None] * capacity  # type: ignore[list-item]
        self._done: list[bool] = [False] * capacity
        self._next_mask: list[list[bool]] = [None] * capacity  # type: ignore[list-item]
        self._pos: int = 0     # next write position
        self._size: int = 0    # current number of valid entries

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of transitions the buffer can hold."""
        return self._capacity

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        """Return True if the buffer contains at least min_size transitions."""
        return self._size >= min_size

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(
        self,
        obs: list[float],
        action_id: int,
        reward: float,
        next_obs: list[float],
        done: bool,
        next_legal_mask: list[bool],
    ) -> None:
        """Store a single transition.

        Parameters
        ----------
        obs:
            Encoded observation, length OBSERVATION_SIZE.
        action_id:
            Integer action id in [0, ACTION_COUNT).
        reward:
            Scalar reward for this transition.
        next_obs:
            Encoded next observation, length OBSERVATION_SIZE.
        done:
            True if the episode ended after this transition.
        next_legal_mask:
            Legal action mask for next_obs, length ACTION_COUNT.
            Ignored (but still stored) when done=True.
        """
        i = self._pos
        self._obs[i] = obs
        self._action[i] = action_id
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = done
        self._next_mask[i] = next_legal_mask
        self._pos = (i + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        rng: _random.Random | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a random minibatch of transitions.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.
        rng:
            Optional ``random.Random`` for reproducible sampling.

        Returns
        -------
        dict with keys defined in BATCH_KEYS.  All tensors are on CPU.

        Raises
        ------
        ValueError:
            If batch_size > len(buffer).
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with "
                f"only {self._size} stored."
            )
        sampler = rng if rng is not None else _random
        indices = sampler.sample(range(self._size), batch_size)

        obs = torch.tensor(
            [self._obs[i] for i in indices], dtype=torch.float32
        )  # (B, OBSERVATION_SIZE)
        action = torch.tensor(
            [self._action[i] for i in indices], dtype=torch.int64
        )  # (B,)
        reward = torch.tensor(
            [self._reward[i] for i in indices], dtype=torch.float32
        )  # (B,)
        next_obs = torch.tensor(
            [self._next_obs[i] for i in indices], dtype=torch.float32
        )  # (B, OBSERVATION_SIZE)
        done = torch.tensor(
            [float(self._done[i]) for i in indices], dtype=torch.float32
        )  # (B,)
        next_mask = torch.tensor(
            [self._next_mask[i] for i in indices], dtype=torch.bool
        )  # (B, ACTION_COUNT)

        return {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "next_mask": next_mask,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._pos = 0
        self._size = 0
