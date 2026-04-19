"""YAML → Agent pipeline: load agent definitions via shared AgentSpec.

Parses YAML files through the shared ``AgentSpec`` model
(``agents.agent_spec``), then materializes Arena-specific runtime
objects (Scorer + Policy → Agent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_system.definition.agent_spec import (
    AgentSpec as SharedAgentSpec,
    load_agent_spec as _load_shared_spec,
    load_agent_specs_from_dir as _load_shared_specs_from_dir,
)
from agent_system.evaluation.arena.agents.core import Agent, Policy, Scorer, TopKPolicy
from agent_system.evaluation.arena.agents.scorers import GreedyScorer, MinimaxScorer, RandomScorer

# ---------------------------------------------------------------------------
# Scorer factory
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: dict[str, type[Scorer]] = {
    "random": RandomScorer,
    "greedy": GreedyScorer,
    "minimax": MinimaxScorer,
}


def build_scorer(algo_type: str, algo_params: dict[str, Any]) -> Scorer:
    """Construct a Scorer from algo type and params.

    Args:
        algo_type: Algorithm family (e.g. ``"minimax"``).
        algo_params: Algorithm parameters (e.g. ``{"depth": 3}``).

    Params are passed opaquely to the Scorer constructor.
    """
    cls = _SCORER_REGISTRY.get(algo_type)
    if cls is None:
        raise ValueError(
            f"Unknown algo type '{algo_type}'. "
            f"Available: {sorted(_SCORER_REGISTRY)}"
        )
    return cls(**algo_params)


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def build_policy(policy_type: str, policy_params: dict[str, Any]) -> Policy:
    """Construct a Policy from policy type and params.

    Args:
        policy_type: Policy family (e.g. ``"top_k"``).
        policy_params: Policy parameters (e.g. ``{"k": 3}``).
    """
    if policy_type == "top_k":
        k = policy_params.get("k", 1)
        return TopKPolicy(k=k)
    raise ValueError(
        f"Unknown policy type '{policy_type}'. Available: ['top_k']"
    )


# ---------------------------------------------------------------------------
# Materialization: AgentSpec → Arena Agent
# ---------------------------------------------------------------------------

def materialize_agent(spec: SharedAgentSpec) -> Agent:
    """Materialize a shared AgentSpec into an Arena-specific Agent.

    This is the Arena's evaluation-side materialization step:
    AgentSpec (pure data) → Scorer + Policy → Arena Agent (runtime object).
    """
    scorer = build_scorer(spec.algo_type, spec.algo_params)
    policy = build_policy(spec.policy_type, spec.policy_params)

    return Agent(
        id=spec.id,
        scorer=scorer,
        policy=policy,
        algo_type=spec.algo_type,
        algo_params=dict(spec.algo_params),
        policy_type=spec.policy_type,
        policy_params=dict(spec.policy_params),
    )


# ---------------------------------------------------------------------------
# Loader (public API — preserves existing interface)
# ---------------------------------------------------------------------------

def load_agent(path: str | Path) -> Agent:
    """Load an Agent definition from a YAML file.

    Parses through the shared AgentSpec model, then materializes an
    Arena-specific Agent (scorer + policy, no RNG yet).
    """
    spec = _load_shared_spec(path)
    return materialize_agent(spec)


def load_agents_from_dir(directory: str | Path) -> list[Agent]:
    """Load all .yaml / .yml agent definitions from a directory.

    Specs whose ``algo_type`` is not in ``_SCORER_REGISTRY`` are silently
    skipped (they belong to other consumers, e.g. the Agent Service).

    Returns agents sorted by id for deterministic ordering.
    """
    agents: list[Agent] = []
    for spec in _load_shared_specs_from_dir(directory):
        if spec.algo_type in _SCORER_REGISTRY:
            agents.append(materialize_agent(spec))
    return agents
