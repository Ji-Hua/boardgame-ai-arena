"""Shared Agent Spec model — canonical semantic definition of an agent.

This module provides the single source of truth (SSOT) in-code model for
agent definitions.  Both Arena and Agent Service will consume this model
(in future phases); this phase only establishes the model and loader.

Design basis:
    - documents/system/design/agent-system-design.md (v1)
    - .copilot/plans/agent-system-migration-plan.md — Section 5

Agent Spec is pure data.  It contains no runtime objects, no
instance-creation methods, and no runtime context (seed, room_id, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentSpec:
    """Canonical semantic definition of an agent.

    Pure data.  Serializable.  Context-free.
    No runtime objects, no instance-creation methods.

    Attributes:
        id: Unique semantic identity of the agent.
        algo_type: Algorithm family (e.g. ``minimax``, ``greedy``, ``random``).
        algo_params: Algorithm configuration (e.g. ``{"depth": 3}``).
        policy_type: Action-selection policy (e.g. ``top_k``).
        policy_params: Policy configuration (e.g. ``{"k": 3}``).
        display_name: Human-readable label.  Defaults to *id*.
        category: Agent classification (``ai``, ``scripted``, ``replay``).
        deterministic: Whether the agent's behavior is deterministic.
    """

    id: str
    algo_type: str
    algo_params: dict[str, Any] = field(default_factory=dict)
    policy_type: str = "top_k"
    policy_params: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    category: str = "ai"
    deterministic: bool = True

    def __post_init__(self) -> None:
        # Defensive copies so callers cannot mutate frozen internals.
        object.__setattr__(self, "algo_params", dict(self.algo_params))
        object.__setattr__(self, "policy_params", dict(self.policy_params))
        if not self.display_name:
            object.__setattr__(self, "display_name", self.id)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def parse_agent_spec(data: dict[str, Any]) -> AgentSpec:
    """Parse a raw YAML dict into an :class:`AgentSpec`.

    Supports both the 6-field Agent Service schema and the 3-field
    legacy Arena schema.  Missing optional fields are populated with
    defaults.

    YAML shapes handled::

        # Agent Service style (full)
        id: greedy
        display_name: Greedy Agent
        category: ai
        deterministic: true
        algo:
          type: greedy
        policy:
          type: top_k
          k: 1

        # Arena style (minimal)
        id: greedy
        algo:
          type: greedy
        policy:
          type: top_k
          k: 1

    Normalization rules:
        - ``policy_type`` comes from ``policy["type"]``.
        - ``policy_params`` contains the remaining policy fields only
          (i.e. the ``type`` key is stripped).
        - Missing ``policy`` block defaults to ``top_k`` with ``k=1``.
    """
    algo = data["algo"]
    policy = data.get("policy") or {}

    # Extract policy_type and remaining policy_params.
    policy_type = policy.get("type", "top_k")
    policy_params = {k: v for k, v in policy.items() if k != "type"}

    # If no policy block at all, default to top_k k=1.
    if not policy:
        policy_params = {"k": 1}

    return AgentSpec(
        id=data["id"],
        algo_type=algo["type"],
        algo_params=dict(algo.get("params") or {}),
        policy_type=policy_type,
        policy_params=policy_params,
        display_name=data.get("display_name", ""),
        category=data.get("category", "ai"),
        deterministic=data.get("deterministic", True),
    )


def load_agent_spec(path: str | Path) -> AgentSpec:
    """Load an :class:`AgentSpec` from a single YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_agent_spec(data)


def load_agent_specs_from_dir(directory: str | Path) -> list[AgentSpec]:
    """Load all ``.yaml`` / ``.yml`` agent specs from a directory.

    Returns specs sorted by ``id`` for deterministic ordering.
    """
    directory = Path(directory)
    specs: list[AgentSpec] = []
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(directory.glob(pattern)):
            specs.append(load_agent_spec(path))
    return specs
