"""Configuration for the Training Dashboard backend."""

from __future__ import annotations

import os
from pathlib import Path

# Default artifact roots, relative to project root
_DEFAULT_ROOTS = [
    "agent_system/training/artifacts/dqn",
    "agent_system/training/artifacts",
    "artifacts/training_runs",
    "training_runs",
    "runs",
    "outputs",
    "agents/checkpoints",
    "arena/results",
]


def get_artifact_roots() -> list[str]:
    """
    Return the list of artifact root directories to scan.

    Reads TRAINING_DASHBOARD_ARTIFACT_ROOTS (colon-separated) if set.
    Falls back to default candidates relative to the project root.
    """
    env_val = os.environ.get("TRAINING_DASHBOARD_ARTIFACT_ROOTS", "").strip()
    if env_val:
        return [r.strip() for r in env_val.split(":") if r.strip()]

    # Resolve defaults relative to repository root (two levels up from this file)
    repo_root = Path(__file__).parent.parent.parent
    resolved: list[str] = []
    for rel in _DEFAULT_ROOTS:
        candidate = repo_root / rel
        if candidate.exists():
            resolved.append(str(candidate))
    return resolved


PORT = int(os.environ.get("TRAINING_DASHBOARD_BACKEND_PORT", "8740"))
