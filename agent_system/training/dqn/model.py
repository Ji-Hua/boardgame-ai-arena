"""DQN Q-network models and masked inference for Quoridor.

MLP architecture (dqn_model_v1):
    Input  → Linear(OBSERVATION_SIZE, hidden_layers[0])
    ReLU
    [Linear(hidden_layers[i], hidden_layers[i+1])  for each subsequent layer]
    ReLU
    Linear(hidden_layers[-1], ACTION_COUNT)  →  Q-values for all 209 actions

Default hidden_layers = [256, 256] (two layers of width 256).

CNN architecture (cnn_model_v1):
    Input  → [batch, C, 9, 9]  where C = CNN_CHANNELS (7 for dqn_obs_cnn_v1)
    Conv2d(C,  32, 3, padding=1) → ReLU
    Conv2d(32, 64, 3, padding=1) → ReLU
    Conv2d(64, 64, 3, padding=1) → ReLU
    Flatten → Linear(64*81, 256) → ReLU → Linear(256, 209)
    Output → Q-values for all 209 actions

Backward compatibility:
    The ``hidden_size`` constructor argument is preserved.  Passing
    ``hidden_size=N`` and no ``hidden_layers`` is equivalent to passing
    ``hidden_layers=[N, N]``.  Old checkpoints that store only ``hidden_size``
    in ``model_config`` are automatically reconstructed with ``[N, N]``.

All inference runs on CPU by default. The device can be configured at
construction time for later GPU support.

Masked action selection:
    Illegal action Q-values are set to -infinity before argmax so they
    can never be selected. Random exploration samples only from legal ids.

Public API:
    QNetwork(hidden_size, hidden_layers, obs_size, action_count)
    CNNQNetwork(in_channels, cnn_channels, action_count)
    build_q_network(model_arch, ...)   — factory for both architectures
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
# Model version constants
# ---------------------------------------------------------------------------

MODEL_VERSION: str = "dqn_model_v1"
CNN_MODEL_VERSION: str = "cnn_model_v1"

# Default hidden layer width (MLP).
DEFAULT_HIDDEN_SIZE: int = 256

# Default CNN channel sizes: [conv1_out, conv2_out, conv3_out].
CNN_DEFAULT_CHANNELS: list[int] = [32, 64, 64]

# CNN board dimensions (matches Quoridor 9×9 board).
_CNN_BOARD_SIZE: int = 9

# Sentinel value used to mask illegal actions before argmax.
_ILLEGAL_Q: float = -math.inf


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Variable-depth MLP Q-network for Quoridor DQN.

    Parameters
    ----------
    hidden_size:
        Width of hidden layers when ``hidden_layers`` is not provided.
        Kept for backward compatibility.  Default 256.
    hidden_layers:
        Explicit list of hidden layer widths, e.g. ``[256, 256]`` or
        ``[512, 512, 256]``.  If provided, ``hidden_size`` is ignored.
        Defaults to ``[hidden_size, hidden_size]``.
    obs_size:
        Input observation dimension.  Defaults to OBSERVATION_SIZE (292).
    action_count:
        Output dimension (number of discrete actions).  Defaults to ACTION_COUNT (209).

    Architecture
    ------------
    obs_size → hidden_layers[0] → ReLU → hidden_layers[1] → ReLU → … → action_count
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        obs_size: int = OBSERVATION_SIZE,
        action_count: int = ACTION_COUNT,
        hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_count = action_count

        # Resolve hidden layer sizes
        if hidden_layers is not None:
            _validate_hidden_layers(hidden_layers)
            self.hidden_layers: list[int] = list(hidden_layers)
        else:
            self.hidden_layers = [hidden_size, hidden_size]

        # Keep hidden_size as the first layer width for backward compat
        self.hidden_size: int = self.hidden_layers[0]

        # Build MLP dynamically
        layers: list[nn.Module] = []
        in_dim = obs_size
        for width in self.hidden_layers:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, action_count))
        self.net = nn.Sequential(*layers)

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

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _validate_hidden_layers(layers: list[int]) -> None:
    """Raise ValueError if hidden_layers is empty or contains non-positive values."""
    if not layers:
        raise ValueError("hidden_layers must not be empty.")
    for i, w in enumerate(layers):
        if not isinstance(w, int) or w <= 0:
            raise ValueError(
                f"hidden_layers[{i}]={w!r} is invalid: all widths must be positive integers."
            )


# ---------------------------------------------------------------------------
# CNN Q-network
# ---------------------------------------------------------------------------

class CNNQNetwork(nn.Module):
    """Small CNN Q-network for Quoridor DQN.

    Designed for board-tensor observations of shape [C, 9, 9] (e.g. the
    ``dqn_obs_cnn_v1`` encoder which produces [7, 9, 9] tensors).

    Architecture
    ------------
    [batch, C, 9, 9]
      → Conv2d(C,  cnn_channels[0], 3, padding=1) → ReLU
      → Conv2d(cnn_channels[0], cnn_channels[1], 3, padding=1) → ReLU
      → Conv2d(cnn_channels[1], cnn_channels[2], 3, padding=1) → ReLU
      → Flatten
      → Linear(cnn_channels[-1] * 81, 256) → ReLU
      → Linear(256, action_count)

    The spatial dimensions are preserved (same padding, stride=1) so the
    final feature map is [cnn_channels[-1], 9, 9] = cnn_channels[-1] * 81
    elements after flattening.

    Parameters
    ----------
    in_channels:
        Number of input channels (planes).  Defaults to 7 for dqn_obs_cnn_v1.
    cnn_channels:
        Output channels for each Conv2d layer.  Length determines the number
        of convolutional layers.  Defaults to [32, 64, 64].
    action_count:
        Number of output Q-values (discrete actions).  Defaults to 209.
    dense_width:
        Width of the single dense hidden layer after flatten.  Default 256.
    """

    def __init__(
        self,
        in_channels: int = 7,
        cnn_channels: list[int] | None = None,
        action_count: int = ACTION_COUNT,
        dense_width: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cnn_channels: list[int] = list(cnn_channels or CNN_DEFAULT_CHANNELS)
        self.action_count = action_count
        self.dense_width = dense_width

        # Convolutional layers
        conv_layers: list[nn.Module] = []
        ch = in_channels
        for out_ch in self.cnn_channels:
            conv_layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        # Dense head
        flat_size = ch * _CNN_BOARD_SIZE * _CNN_BOARD_SIZE
        self.head = nn.Sequential(
            nn.Linear(flat_size, dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, action_count),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a single observation or a batch.

        Parameters
        ----------
        obs:
            Float tensor of shape ``(C, 9, 9)`` (single) or
            ``(batch, C, 9, 9)`` (batch).

        Returns
        -------
        torch.Tensor of shape ``(action_count,)`` or ``(batch, action_count)``.
        """
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)          # → (1, C, 9, 9)
        x = self.conv(obs)                  # → (B, cnn_channels[-1], 9, 9)
        x = x.flatten(1)                    # → (B, cnn_channels[-1]*81)
        q = self.head(x)                    # → (B, action_count)
        if single:
            q = q.squeeze(0)                # → (action_count,)
        return q

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_q_network(
    model_arch: str,
    action_count: int = ACTION_COUNT,
    *,
    # MLP-specific
    obs_size: int = OBSERVATION_SIZE,
    hidden_layers: list[int] | None = None,
    # CNN-specific
    in_channels: int = 7,
    cnn_channels: list[int] | None = None,
    dense_width: int = 256,
) -> "QNetwork | CNNQNetwork":
    """Construct a Q-network for the requested architecture.

    Parameters
    ----------
    model_arch:
        ``"mlp"`` — builds :class:`QNetwork`.
        ``"cnn"`` — builds :class:`CNNQNetwork`.
    action_count:
        Number of discrete output actions.  Defaults to 209.
    obs_size:
        MLP input dimension (ignored for CNN).  Defaults to OBSERVATION_SIZE.
    hidden_layers:
        MLP hidden layer widths (ignored for CNN).  Defaults to [256, 256].
    in_channels:
        CNN input channel count (ignored for MLP).  Defaults to 7.
    cnn_channels:
        CNN conv-layer channel counts (ignored for MLP).  Defaults to
        CNN_DEFAULT_CHANNELS = [32, 64, 64].
    dense_width:
        CNN dense hidden layer width (ignored for MLP).  Defaults to 256.

    Returns
    -------
    QNetwork or CNNQNetwork depending on *model_arch*.

    Raises
    ------
    ValueError:
        If *model_arch* is not ``"mlp"`` or ``"cnn"``.
    """
    if model_arch == "mlp":
        _hl = list(hidden_layers) if hidden_layers else [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE]
        return QNetwork(hidden_layers=_hl, obs_size=obs_size, action_count=action_count)
    if model_arch == "cnn":
        return CNNQNetwork(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            action_count=action_count,
            dense_width=dense_width,
        )
    raise ValueError(
        f"Unknown model_arch '{model_arch}'. Expected 'mlp' or 'cnn'."
    )


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
