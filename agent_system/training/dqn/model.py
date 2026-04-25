"""DQN Q-network model and masked inference for Quoridor.

Network architecture (dqn_model_v1):
    Input  → Linear(OBSERVATION_SIZE, hidden_size)
    ReLU
    Linear(hidden_size, hidden_size)
    ReLU
    Linear(hidden_size, ACTION_COUNT)  →  Q-values for all 209 actions

All inference runs on CPU by default. The device can be configured at
construction time for later GPU support.

Masked action selection:
    Illegal action Q-values are set to -infinity before argmax so they
    can never be selected. Random exploration samples only from legal ids.

Public API:
    QNetwork(hidden_size, device) — the raw PyTorch module
    DQNPolicy(network)           — greedy / epsilon-greedy action selector

    select_greedy_action(q_values, mask)         -> action_id
    select_epsilon_greedy_action(q_values, mask, epsilon, rng) -> action_id
"""

from __future__ import annotations

import math
import random as _random
from typing import Any

import torch
import torch.nn as nn

from agent_system.training.dqn.action_space import ACTION_COUNT
from agent_system.training.dqn.observation import OBSERVATION_SIZE

# ---------------------------------------------------------------------------
# Model version constant
# ---------------------------------------------------------------------------

MODEL_VERSION: str = "dqn_model_v1"

# Default hidden layer width.
DEFAULT_HIDDEN_SIZE: int = 256

# Sentinel value used to mask illegal actions before argmax.
_ILLEGAL_Q: float = -math.inf


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Two-hidden-layer MLP Q-network for Quoridor DQN.

    Parameters
    ----------
    hidden_size:
        Width of both hidden layers. Defaults to DEFAULT_HIDDEN_SIZE (256).
    obs_size:
        Input observation dimension. Defaults to OBSERVATION_SIZE (292).
    action_count:
        Output dimension (number of discrete actions). Defaults to ACTION_COUNT (209).
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        obs_size: int = OBSERVATION_SIZE,
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_count = action_count
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a single observation or a batch.

        Parameters
        ----------
        obs:
            Float tensor of shape (obs_size,) or (batch, obs_size).

        Returns
        -------
        torch.Tensor of shape (action_count,) or (batch, action_count).
        """
        return self.net(obs)


# ---------------------------------------------------------------------------
# Masked action selection helpers (pure functions, no model dependency)
# ---------------------------------------------------------------------------

def select_greedy_action(
    q_values: torch.Tensor,
    mask: list[bool],
) -> int:
    """Return the legal action id with the highest Q-value.

    Parameters
    ----------
    q_values:
        1-D float tensor of shape (ACTION_COUNT,).
    mask:
        Boolean list of length ACTION_COUNT. True = legal.

    Returns
    -------
    int action_id with the highest Q-value among legal actions.

    Raises
    ------
    ValueError:
        If mask length does not match ACTION_COUNT, or if no legal action exists.
    """
    _validate_mask(mask)
    masked_q = _apply_mask(q_values, mask)
    return int(masked_q.argmax().item())


def select_epsilon_greedy_action(
    q_values: torch.Tensor,
    mask: list[bool],
    epsilon: float = 0.0,
    rng: _random.Random | None = None,
) -> int:
    """Return a legal action id using epsilon-greedy selection.

    With probability epsilon, sample uniformly from legal action ids.
    Otherwise, return the greedy (highest-Q legal) action.

    Parameters
    ----------
    q_values:
        1-D float tensor of shape (ACTION_COUNT,).
    mask:
        Boolean list of length ACTION_COUNT. True = legal.
    epsilon:
        Exploration probability in [0.0, 1.0].
    rng:
        Optional ``random.Random`` instance for reproducible exploration.
        Uses the module-level random if None.

    Raises
    ------
    ValueError:
        If mask length does not match ACTION_COUNT, or if no legal action exists.
    """
    _validate_mask(mask)
    legal_ids = [i for i, v in enumerate(mask) if v]

    if not legal_ids:
        raise ValueError("No legal actions available (mask is all False).")

    if epsilon > 0.0:
        rand_val = rng.random() if rng is not None else _random.random()
        if rand_val < epsilon:
            return rng.choice(legal_ids) if rng is not None else _random.choice(legal_ids)

    return select_greedy_action(q_values, mask)


# ---------------------------------------------------------------------------
# DQNPolicy — combines network + masked selection
# ---------------------------------------------------------------------------

class DQNPolicy:
    """Combines QNetwork inference with masked epsilon-greedy action selection.

    This is the minimal inference helper needed by the training loop
    and rollout tests. It is not a registered runtime agent.

    Parameters
    ----------
    network:
        A QNetwork (or compatible module with the same forward signature).
    device:
        torch.device or string. Defaults to CPU.
    """

    def __init__(
        self,
        network: QNetwork,
        device: torch.device | str = "cpu",
    ) -> None:
        self._network = network
        self._device = torch.device(device)
        self._network.to(self._device)
        self._network.eval()

    def select_action(
        self,
        observation: list[float],
        mask: list[bool],
        epsilon: float = 0.0,
        rng: _random.Random | None = None,
    ) -> int:
        """Select an action for the current observation and legal mask.

        Parameters
        ----------
        observation:
            Encoded observation as list[float] of length OBSERVATION_SIZE.
        mask:
            Legal action mask as list[bool] of length ACTION_COUNT.
        epsilon:
            Exploration probability. 0.0 = fully greedy.
        rng:
            Optional random.Random for reproducible exploration.

        Returns
        -------
        int action_id in [0, ACTION_COUNT).
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            q_values = self._network(obs_tensor)

        return select_epsilon_greedy_action(q_values, mask, epsilon=epsilon, rng=rng)

    @property
    def network(self) -> QNetwork:
        """The underlying QNetwork module."""
        return self._network

    @property
    def device(self) -> torch.device:
        """The device the network runs on."""
        return self._device


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_mask(mask: list[bool]) -> None:
    if len(mask) != ACTION_COUNT:
        raise ValueError(
            f"Mask length {len(mask)} does not match ACTION_COUNT {ACTION_COUNT}."
        )
    legal_ids = [i for i, v in enumerate(mask) if v]
    if not legal_ids:
        raise ValueError("No legal actions available (mask is all False).")


def _apply_mask(q_values: torch.Tensor, mask: list[bool]) -> torch.Tensor:
    """Return a copy of q_values with illegal positions set to -inf."""
    illegal_mask = torch.tensor(
        [not v for v in mask], dtype=torch.bool, device=q_values.device
    )
    masked = q_values.clone()
    masked[illegal_mask] = _ILLEGAL_Q
    return masked
