"""DQN checkpoint save/load utilities.  Supports MLP and CNN networks.

Checkpoint format: a single PyTorch .pt file produced by ``torch.save()``.

Saved dict keys:
    checkpoint_id        str     — unique identifier (caller-supplied or auto)
    agent_id             str     — logical agent name
    training_step        int     — global optimizer step count
    episode_count        int     — episodes completed at save time (0 if unknown)
    model_state_dict     dict    — network.state_dict()
    optimizer_state_dict dict    — optimizer.state_dict() (or {} if not provided)
    model_config         dict    — architecture config, varies by model_arch:
                                   MLP: {"obs_size", "hidden_size", "hidden_layers",
                                         "action_count", "model_arch", "algorithm",
                                         "param_count", "reward_mode", ...}
                                   CNN: {"in_channels", "cnn_channels", "action_count",
                                         "model_arch", "observation_shape",
                                         "algorithm", "param_count", "reward_mode", ...}
    observation_version  str     — e.g. "dqn_obs_v1" or "dqn_obs_cnn_v1"
    observation_size     int     — flat element count (292 for v1, 567 for cnn_v1)
    action_space_version str     — e.g. "dqn_action_v1"
    action_count         int     — must match ACTION_COUNT (209)
    created_at           str     — ISO-8601 UTC timestamp
    eval_summary         dict    — optional evaluation metrics (may be empty)

Backward compatibility:
    Old MLP checkpoints that only have ``model_config["hidden_size"] = N`` are
    automatically loaded as ``hidden_layers = [N, N]``.
    Old checkpoints without ``model_arch`` default to ``"mlp"``.

Compatibility validation (checked on load):
    - observation_version must match expected version
    - observation_size must match the expected size for that obs version
    - action_count must equal current ACTION_COUNT
    - model_config["action_count"] must equal current ACTION_COUNT (if present)
    - MLP: model_config["obs_size"] must equal OBSERVATION_SIZE (if present)

Public API:
    save_checkpoint(path, network, ...) -> None
    load_checkpoint(path) -> DQNCheckpoint
    DQNCheckpoint                       — dataclass with all metadata + network
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from agent_system.training.dqn.action_space import ACTION_COUNT, ACTION_SPACE_VERSION
from agent_system.training.dqn.model import (
    CNN_DEFAULT_CHANNELS,
    DEFAULT_HIDDEN_SIZE,
    CNNQNetwork,
    QNetwork,
    build_q_network,
)
from agent_system.training.dqn.observation import OBSERVATION_SIZE, OBSERVATION_VERSION
from agent_system.training.dqn.observation_cnn import (
    CNN_CHANNELS,
    CNN_OBSERVATION_SHAPE,
    CNN_OBSERVATION_SIZE,
    CNN_OBSERVATION_VERSION,
)

# ---------------------------------------------------------------------------
# Checkpoint version (format version, not model version)
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT_VERSION: str = "dqn_checkpoint_v1"

# Required metadata keys that must be present in a .pt file.
_REQUIRED_KEYS: tuple[str, ...] = (
    "checkpoint_id",
    "agent_id",
    "training_step",
    "episode_count",
    "model_state_dict",
    "model_config",
    "observation_version",
    "observation_size",
    "action_space_version",
    "action_count",
    "created_at",
)


# ---------------------------------------------------------------------------
# Dataclass representing a loaded checkpoint
# ---------------------------------------------------------------------------

@dataclass
class DQNCheckpoint:
    """All metadata and weights from a loaded DQN checkpoint file.

    Consumers should use ``network`` directly rather than reconstructing
    from scratch; it is already loaded and ready for inference.

    ``network`` may be a :class:`QNetwork` (MLP) or :class:`CNNQNetwork`
    depending on the value of ``model_config['model_arch']``.
    """

    checkpoint_id: str
    agent_id: str
    training_step: int
    episode_count: int
    model_config: dict
    observation_version: str
    observation_size: int
    action_space_version: str
    action_count: int
    created_at: str
    eval_summary: dict
    network: nn.Module   # QNetwork (MLP) or CNNQNetwork
    optimizer_state_dict: dict = field(default_factory=dict)

    @property
    def hidden_size(self) -> int:
        """Width of the first hidden layer (kept for backward compatibility)."""
        return self.model_config.get("hidden_size", DEFAULT_HIDDEN_SIZE)

    @property
    def hidden_layers(self) -> list[int]:
        """List of hidden layer widths.  Falls back to [hidden_size, hidden_size]
        when loading an old checkpoint that only stores hidden_size."""
        hl = self.model_config.get("hidden_layers")
        if hl is not None:
            return list(hl)
        hs = self.model_config.get("hidden_size", DEFAULT_HIDDEN_SIZE)
        return [hs, hs]

    @property
    def algorithm(self) -> str:
        """Training algorithm ('dqn' or 'double_dqn'). Defaults to 'dqn' for old checkpoints."""
        return self.model_config.get("algorithm", "dqn")

    @property
    def is_double_dqn(self) -> bool:
        """True when algorithm == 'double_dqn'."""
        return self.algorithm == "double_dqn"

    @property
    def model_arch(self) -> str:
        """Model architecture identifier. Defaults to 'mlp'."""
        return self.model_config.get("model_arch", "mlp")

    @property
    def observation_shape(self) -> list[int] | None:
        """Observation tensor shape, or None if not stored.

        For MLP: ``[292]``; for CNN: ``[7, 9, 9]``.
        """
        return self.model_config.get("observation_shape")

    @property
    def cnn_channels(self) -> list[int] | None:
        """CNN convolutional channel widths, or None for MLP checkpoints."""
        return self.model_config.get("cnn_channels")

    @property
    def param_count(self) -> int | None:
        """Total trainable parameter count, or None if not recorded."""
        return self.model_config.get("param_count")

    @property
    def reward_mode(self) -> str:
        """Reward mode used during training. Defaults to 'terminal'."""
        return self.model_config.get("reward_mode", "terminal")

    @property
    def distance_reward_weight(self) -> float:
        """Distance reward weight used during training."""
        return float(self.model_config.get("distance_reward_weight", 0.01))

    @property
    def distance_delta_clip(self) -> float:
        """Distance delta clip used during training."""
        return float(self.model_config.get("distance_delta_clip", 2.0))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    network: nn.Module,
    *,
    agent_id: str = "dqn_agent",
    training_step: int = 0,
    episode_count: int = 0,
    checkpoint_id: str | None = None,
    optimizer: optim.Optimizer | None = None,
    eval_summary: dict | None = None,
    obs_version: str | None = None,
    device: str | None = None,
    algorithm: str = "dqn",
    model_arch: str = "mlp",
    param_count: int | None = None,
    reward_mode: str = "terminal",
    distance_reward_weight: float = 0.01,
    distance_delta_clip: float = 2.0,
    cnn_channels: list[int] | None = None,
    opponent: str | None = None,
) -> Path:
    """Save a DQN checkpoint to *path*.  Supports both MLP and CNN networks.

    Parameters
    ----------
    path:
        Destination file path.  Parent directories are created if missing.
        Should end in ``.pt`` by convention.
    network:
        The QNetwork or CNNQNetwork whose weights to save.
    agent_id:
        Logical name for the agent (used as a human-readable identifier).
    training_step:
        Optimizer step count at save time.
    episode_count:
        Episode count at save time (0 if not tracked).
    checkpoint_id:
        Unique string identifier.  Auto-generated UUID4 if None.
    optimizer:
        If provided, saves ``optimizer.state_dict()`` for later resume.
    eval_summary:
        Optional dict of evaluation metrics to embed in the checkpoint.
    obs_version:
        Observation encoding version string (e.g. ``"dqn_obs_v1"`` or
        ``"dqn_obs_cnn_v1"``).  Defaults to the current ``OBSERVATION_VERSION``
        (MLP v1) for backward compatibility.  Pass explicitly for CNN.
    algorithm:
        Training algorithm: ``"dqn"`` or ``"double_dqn"``.
    model_arch:
        Architecture identifier: ``"mlp"`` or ``"cnn"``.
    param_count:
        Total trainable parameter count.  If None, computed from *network*.
    reward_mode:
        Reward function mode used during training (e.g. ``"distance_delta"``)
    distance_reward_weight:
        Distance reward scale factor used during training.
    distance_delta_clip:
        Distance delta clip value used during training.
    cnn_channels:
        CNN conv-layer channel counts (CNN only, ignored for MLP).
        Stored in model_config for checkpoint reload.
    opponent:
        Training opponent identifier (stored for informational purposes).

    Returns
    -------
    Resolved Path where the checkpoint was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _obs_version = obs_version if obs_version is not None else (
        CNN_OBSERVATION_VERSION if model_arch == "cnn" else OBSERVATION_VERSION
    )
    _param_count = param_count if param_count is not None else network.parameter_count()

    # Build architecture-specific model_config and observation_size
    if model_arch == "cnn":
        _in_channels = getattr(network, "in_channels", CNN_CHANNELS)
        _cnn_channels = list(
            cnn_channels
            or getattr(network, "cnn_channels", None)
            or CNN_DEFAULT_CHANNELS
        )
        _action_count = getattr(network, "action_count", ACTION_COUNT)
        _obs_shape = list(CNN_OBSERVATION_SHAPE)
        _obs_size = CNN_OBSERVATION_SIZE
        _model_config: dict = {
            "model_arch": model_arch,
            "in_channels": _in_channels,
            "cnn_channels": _cnn_channels,
            "action_count": _action_count,
            "observation_shape": _obs_shape,
            "algorithm": algorithm,
            "param_count": _param_count,
            "reward_mode": reward_mode,
            "distance_reward_weight": distance_reward_weight,
            "distance_delta_clip": distance_delta_clip,
        }
    else:
        # MLP — access QNetwork-specific attributes
        _model_config = {
            "obs_size": getattr(network, "obs_size", OBSERVATION_SIZE),
            "hidden_size": getattr(network, "hidden_size", DEFAULT_HIDDEN_SIZE),
            "hidden_layers": list(getattr(network, "hidden_layers", [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE])),
            "action_count": getattr(network, "action_count", ACTION_COUNT),
            "algorithm": algorithm,
            "model_arch": model_arch,
            "param_count": _param_count,
            "reward_mode": reward_mode,
            "distance_reward_weight": distance_reward_weight,
            "distance_delta_clip": distance_delta_clip,
            "observation_shape": [OBSERVATION_SIZE],
        }
        _obs_size = OBSERVATION_SIZE

    # Include opponent if provided
    if opponent is not None:
        _model_config["opponent"] = opponent

    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_id": checkpoint_id or str(uuid.uuid4()),
        "agent_id": agent_id,
        "training_step": training_step,
        "episode_count": episode_count,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
        "model_config": _model_config,
        "observation_version": _obs_version,
        "observation_size": _obs_size,
        "action_space_version": ACTION_SPACE_VERSION,
        "action_count": ACTION_COUNT,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "eval_summary": eval_summary or {},
        "device": device,
    }

    torch.save(payload, path)
    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    expected_obs_version: str | None = None,
) -> DQNCheckpoint:
    """Load a DQN checkpoint from *path* and return a :class:`DQNCheckpoint`.

    Validates metadata compatibility before constructing the network.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.
    expected_obs_version:
        Expected observation version string (e.g. ``"dqn_obs_v1"`` or
        ``"dqn_obs_cnn_v1"``).  When ``None`` (default), the version stored
        in the checkpoint is used — this allows transparent loading of both
        MLP and CNN checkpoints without requiring the caller to know the
        architecture in advance.  Pass explicitly to enforce a strict match.

    Returns
    -------
    DQNCheckpoint with a fully constructed and weight-loaded network.

    Raises
    ------
    FileNotFoundError:
        If *path* does not exist.
    ValueError:
        If any required metadata key is missing or compatibility fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw: dict = torch.load(path, map_location="cpu", weights_only=False)

    # ---- Check required keys ----
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing} in {path}"
        )

    # ---- Compatibility validation ----
    # Auto-detect expected obs_version from model_arch rather than from the
    # saved obs_version itself (detecting from the saved value would disable
    # version validation entirely).  CNN checkpoints use CNN_OBSERVATION_VERSION;
    # everything else validates against OBSERVATION_VERSION.
    if expected_obs_version is None:
        _arch = raw.get("model_config", {}).get("model_arch", "mlp")
        _expected = CNN_OBSERVATION_VERSION if _arch == "cnn" else OBSERVATION_VERSION
    else:
        _expected = expected_obs_version
    _validate_compatibility(raw, path, expected_obs_version=_expected)

    # ---- Reconstruct network ----
    cfg = raw["model_config"]
    arch = cfg.get("model_arch", "mlp")

    if arch == "cnn":
        network: nn.Module = CNNQNetwork(
            in_channels=cfg.get("in_channels", CNN_CHANNELS),
            cnn_channels=cfg.get("cnn_channels", CNN_DEFAULT_CHANNELS),
            action_count=cfg.get("action_count", ACTION_COUNT),
        )
    else:
        # MLP — fall back for old checkpoints that only store hidden_size
        if "hidden_layers" in cfg:
            _hidden_layers = cfg["hidden_layers"]
        else:
            _hs = cfg.get("hidden_size", DEFAULT_HIDDEN_SIZE)
            _hidden_layers = [_hs, _hs]
        network = QNetwork(
            hidden_layers=_hidden_layers,
            obs_size=cfg["obs_size"],
            action_count=cfg["action_count"],
        )

    network.load_state_dict(raw["model_state_dict"])
    network.eval()

    return DQNCheckpoint(
        checkpoint_id=raw["checkpoint_id"],
        agent_id=raw["agent_id"],
        training_step=raw["training_step"],
        episode_count=raw["episode_count"],
        model_config=cfg,
        observation_version=raw["observation_version"],
        observation_size=raw["observation_size"],
        action_space_version=raw["action_space_version"],
        action_count=raw["action_count"],
        created_at=raw["created_at"],
        eval_summary=raw.get("eval_summary", {}),
        network=network,
        optimizer_state_dict=raw.get("optimizer_state_dict", {}),
    )


# ---------------------------------------------------------------------------
# Compatibility validation (internal)
# ---------------------------------------------------------------------------

def _validate_compatibility(
    raw: dict,
    path: Path,
    expected_obs_version: str | None = None,
) -> None:
    """Raise ValueError if the checkpoint is incompatible with current constants.

    Parameters
    ----------
    expected_obs_version:
        The observation version to validate against.  Defaults to the
        module-level ``OBSERVATION_VERSION`` constant if None.
        Pass ``CNN_OBSERVATION_VERSION`` for CNN checkpoints.
    """
    errors: list[str] = []

    _expected_obs = expected_obs_version if expected_obs_version is not None else OBSERVATION_VERSION

    saved_obs_version = raw["observation_version"]
    if saved_obs_version != _expected_obs:
        errors.append(
            f"observation_version mismatch: checkpoint has '{saved_obs_version}', "
            f"expected '{_expected_obs}'"
        )

    # Validate observation_size against the correct expected size for the version.
    saved_obs_size = raw["observation_size"]
    if saved_obs_version == CNN_OBSERVATION_VERSION:
        _expected_size = CNN_OBSERVATION_SIZE
    else:
        _expected_size = OBSERVATION_SIZE
    if saved_obs_size != _expected_size:
        errors.append(
            f"observation_size mismatch: checkpoint has {saved_obs_size}, "
            f"expected {_expected_size} for obs_version='{saved_obs_version}'"
        )

    saved_action_count = raw["action_count"]
    if saved_action_count != ACTION_COUNT:
        errors.append(
            f"action_count mismatch: checkpoint has {saved_action_count}, "
            f"current ACTION_COUNT is {ACTION_COUNT}"
        )

    cfg = raw.get("model_config", {})
    arch = cfg.get("model_arch", "mlp")

    # Only validate obs_size in model_config for MLP (CNN uses in_channels instead).
    if arch != "cnn":
        cfg_obs_size = cfg.get("obs_size")
        if cfg_obs_size is not None and cfg_obs_size != OBSERVATION_SIZE:
            errors.append(
                f"model_config.obs_size mismatch: checkpoint has {cfg_obs_size}, "
                f"current OBSERVATION_SIZE is {OBSERVATION_SIZE}"
            )

    cfg_action_count = cfg.get("action_count")
    if cfg_action_count is not None and cfg_action_count != ACTION_COUNT:
        errors.append(
            f"model_config.action_count mismatch: checkpoint has {cfg_action_count}, "
            f"current ACTION_COUNT is {ACTION_COUNT}"
        )

    if errors:
        bullet = "\n  - ".join(errors)
        raise ValueError(
            f"Checkpoint at '{path}' is incompatible with current environment:\n"
            f"  - {bullet}"
        )
