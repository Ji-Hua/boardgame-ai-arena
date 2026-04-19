"""YAML → BaseAgent loader for the Agent Service.

Loads agent definitions from YAML files via the shared ``AgentSpec`` model
(``agents.agent_spec``), then maps to BaseAgent implementations for
runtime materialization.

YAML schema::

    id: minimax
    display_name: Minimax Agent
    category: ai
    deterministic: true
    algo:
      type: minimax
      params:
        depth: 2
    policy:
      type: top_k
      k: 3

The ``algo.type`` field maps to a BaseAgent implementation class.
The ``policy`` block (if k > 1) and seed are injected at instance-creation
time via the agent constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.agent_spec import (
    AgentSpec as SharedAgentSpec,
    load_agent_spec as _load_shared_spec,
    load_agent_specs_from_dir as _load_shared_specs_from_dir,
    parse_agent_spec as _parse_shared_spec,
)
from agents.agent_service.base_agent import BaseAgent
from agents.agent_service.policy import Policy, TopKPolicy, build_policy


# ---------------------------------------------------------------------------
# Agent definition — thin compatibility wrapper over shared AgentSpec
# ---------------------------------------------------------------------------

class AgentDefinition:
    """Parsed YAML agent definition — compatibility wrapper over shared AgentSpec.

    Preserves the existing Agent-Service-internal interface (including
    ``policy_config`` dict and ``policy_k`` property) while delegating
    canonical definition data to the shared ``AgentSpec`` model.
    """

    __slots__ = ("_spec", "policy_config")

    def __init__(self, spec: SharedAgentSpec) -> None:
        self._spec = spec
        # Reconstruct policy_config dict for backward compatibility with
        # create_agent_from_definition() and YamlAgentMaterializer.
        self.policy_config: dict[str, Any] = {"type": spec.policy_type, **spec.policy_params}

    # --- Delegated properties from shared AgentSpec ---

    @property
    def id(self) -> str:
        return self._spec.id

    @property
    def display_name(self) -> str:
        return self._spec.display_name

    @property
    def category(self) -> str:
        return self._spec.category

    @property
    def deterministic(self) -> bool:
        return self._spec.deterministic

    @property
    def algo_type(self) -> str:
        return self._spec.algo_type

    @property
    def algo_params(self) -> dict[str, Any]:
        return dict(self._spec.algo_params)

    @property
    def policy_k(self) -> int:
        return self.policy_config.get("k", 1)

    @property
    def spec(self) -> SharedAgentSpec:
        """Access the underlying shared AgentSpec."""
        return self._spec


# ---------------------------------------------------------------------------
# YAML parsing — delegates to shared AgentSpec parser
# ---------------------------------------------------------------------------

def parse_agent_definition(data: dict[str, Any]) -> AgentDefinition:
    """Parse a raw YAML dict into an AgentDefinition."""
    return AgentDefinition(_parse_shared_spec(data))


def load_definition(path: str | Path) -> AgentDefinition:
    """Load an AgentDefinition from a YAML file."""
    return AgentDefinition(_load_shared_spec(path))


def load_definitions_from_dir(directory: str | Path) -> list[AgentDefinition]:
    """Load all .yaml/.yml agent definitions from a directory."""
    return [AgentDefinition(spec) for spec in _load_shared_specs_from_dir(directory)]


# ---------------------------------------------------------------------------
# Agent construction from definition
# ---------------------------------------------------------------------------

def _build_policy_from_config(policy_config: dict[str, Any]) -> Policy | None:
    """Build a Policy from config, returning None if k=1 (use default path)."""
    k = policy_config.get("k", 1)
    if k <= 1:
        return None  # Let the agent use its default deterministic path
    return build_policy(policy_config)


def create_agent_from_definition(
    defn: AgentDefinition,
    config_overrides: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> BaseAgent:
    """Instantiate a BaseAgent from an AgentDefinition.

    Args:
        defn: Parsed YAML definition.
        config_overrides: Runtime overrides (e.g. from API request).
            May contain ``policy`` and ``depth`` overrides.
        context: Runtime context (e.g. ``{"seed": 42}``).

    Returns:
        A configured BaseAgent instance.
    """
    overrides = dict(config_overrides or {})
    ctx = dict(context or {})
    seed = ctx.get("seed")

    # Merge: overrides take precedence over YAML definition
    algo_params = dict(defn.algo_params)
    algo_params.update({k: v for k, v in overrides.items()
                        if k not in ("policy",)})

    # Policy: override > YAML definition
    policy_config = overrides.get("policy", defn.policy_config)
    policy = _build_policy_from_config(policy_config)

    return _AGENT_BUILDERS[defn.algo_type](algo_params, policy, seed)


# ---------------------------------------------------------------------------
# Agent builder registry (algo_type → constructor function)
# ---------------------------------------------------------------------------

def _build_random_v2(params: dict, policy: Policy | None, seed: int | None) -> BaseAgent:
    from agents.agent_service.agents.random_agent import RandomAgentV2
    return RandomAgentV2(threshold=params.get("threshold", 0.8))


def _build_greedy(params: dict, policy: Policy | None, seed: int | None) -> BaseAgent:
    from agents.agent_service.agents.greedy_agent import GreedyAgent
    return GreedyAgent(policy=policy, seed=seed)


def _build_minimax(params: dict, policy: Policy | None, seed: int | None) -> BaseAgent:
    from agents.agent_service.agents.minimax_agent import MinimaxAgent
    return MinimaxAgent(
        depth=int(params.get("depth", 2)),
        policy=policy,
        seed=seed,
    )


_AGENT_BUILDERS: dict[str, Any] = {
    "random_v2": _build_random_v2,
    "greedy": _build_greedy,
    "minimax": _build_minimax,
}
