"""DQN checkpoint save/load utilities.

Checkpoint format: a single PyTorch .pt file produced by ``torch.save()``.

Saved dict keys:
    checkpoint_id        str     — unique identifier (caller-supplied or auto)
    agent_id             str     — logical agent name
    training_step        int     — global optimizer step count
    episode_count        int     — episodes completed at save time (0 if unknown)
    model_state_dict     dict    — QNetwork.state_dict()
    optimizer_state_dict dict    — optimizer.state_dict() (or {} if not provided)
    model_config         dict    — {"obs_size", "hidden_size", "action_count"}
    observation_version  str     — e.g. "dqn_obs_v1"
    observation_size     int     — must match OBSERVATION_SIZE
    action_space_version str     — e.g. "dqn_action_v1"
    action_count         int     — must match ACTION_COUNT (209)
    created_at           str     — ISO-8601 UTC timestamp
    eval_summary         dict    — optional evaluation metrics (may be empty)

Compatibility validation (checked on load):
    - observation_version must equal current OBSERVATION_VERSION
    - observation_size must equal current OBSERVATION_SIZE
    - action_count must equal current ACTION_COUNT
    - model_config["obs_size"] must equal current OBSERVATION_SIZE
    - model_config["action_count"] must equal current ACTION_COUNT

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
import torch.optim as optim

from agent_system.training.dqn.action_space import ACTION_COUNT, ACTION_SPACE_VERSION
from agent_system.training.dqn.model import DEFAULT_HIDDEN_SIZE, QNetwork
from agent_system.training.dqn.observation import OBSERVATION_SIZE, OBSERVATION_VERSION

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
    network: QNetwork
    optimizer_state_dict: dict = field(default_factory=dict)

    @property
    def hidden_size(self) -> int:
        return self.model_config.get("hidden_size", DEFAULT_HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    network: QNetwork,
    *,
    agent_id: str = "dqn_agent",
    training_step: int = 0,
    episode_count: int = 0,
    checkpoint_id: str | None = None,
    optimizer: optim.Optimizer | None = None,
    eval_summary: dict | None = None,
    obs_version: str | None = None,
) -> Path:
    """Save a DQN checkpoint to *path*.

    Parameters
    ----------
    path:
        Destination file path.  Parent directories are created if missing.
        Should end in ``.pt`` by convention.
    network:
        The QNetwork whose weights to save.
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
        ``"dqn_obs_v2"``).  Defaults to the current ``OBSERVATION_VERSION``
        imported from ``observation.py``.  Pass explicitly when training with
        a non-default encoder (e.g. board-flip v2).

    Returns
    -------
    Resolved Path where the checkpoint was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _obs_version = obs_version if obs_version is not None else OBSERVATION_VERSION

    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_id": checkpoint_id or str(uuid.uuid4()),
        "agent_id": agent_id,
        "training_step": training_step,
        "episode_count": episode_count,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
        "model_config": {
            "obs_size": network.obs_size,
            "hidden_size": network.hidden_size,
            "action_count": network.action_count,
        },
        "observation_version": _obs_version,
        "observation_size": OBSERVATION_SIZE,
        "action_space_version": ACTION_SPACE_VERSION,
        "action_count": ACTION_COUNT,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "eval_summary": eval_summary or {},
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
        ``"dqn_obs_v2"``).  Defaults to the current ``OBSERVATION_VERSION``
        (v1).  Pass explicitly when loading a checkpoint saved with a
        non-default encoder so that the version check enforces the right
        version.

    Returns
    -------
    DQNCheckpoint with a fully constructed and weight-loaded QNetwork.

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
    _expected = expected_obs_version if expected_obs_version is not None else OBSERVATION_VERSION
    _validate_compatibility(raw, path, expected_obs_version=_expected)

    # ---- Reconstruct network ----
    cfg = raw["model_config"]
    network = QNetwork(
        hidden_size=cfg.get("hidden_size", DEFAULT_HIDDEN_SIZE),
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
    """
    errors: list[str] = []

    _expected_obs = expected_obs_version if expected_obs_version is not None else OBSERVATION_VERSION

    saved_obs_version = raw["observation_version"]
    if saved_obs_version != _expected_obs:
        errors.append(
            f"observation_version mismatch: checkpoint has '{saved_obs_version}', "
            f"expected '{_expected_obs}'"
        )

    saved_obs_size = raw["observation_size"]
    if saved_obs_size != OBSERVATION_SIZE:
        errors.append(
            f"observation_size mismatch: checkpoint has {saved_obs_size}, "
            f"current OBSERVATION_SIZE is {OBSERVATION_SIZE}"
        )

    saved_action_count = raw["action_count"]
    if saved_action_count != ACTION_COUNT:
        errors.append(
            f"action_count mismatch: checkpoint has {saved_action_count}, "
            f"current ACTION_COUNT is {ACTION_COUNT}"
        )

    cfg = raw.get("model_config", {})
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
