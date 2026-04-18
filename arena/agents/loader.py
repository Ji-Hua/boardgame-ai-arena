"""YAML → Agent pipeline: load agent definitions from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from arena.agents.core import Agent, Policy, Scorer, TopKPolicy
from arena.agents.scorers import GreedyScorer, MinimaxScorer, RandomScorer

# ---------------------------------------------------------------------------
# Scorer factory
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: dict[str, type[Scorer]] = {
    "random": RandomScorer,
    "greedy": GreedyScorer,
    "minimax": MinimaxScorer,
}


def build_scorer(algo_config: dict[str, Any]) -> Scorer:
    """Construct a Scorer from an algo config block.

    Example config::

        {"type": "minimax", "params": {"depth": 3}}

    Params are passed opaquely to the Scorer constructor.
    """
    algo_type = algo_config["type"]
    cls = _SCORER_REGISTRY.get(algo_type)
    if cls is None:
        raise ValueError(
            f"Unknown algo type '{algo_type}'. "
            f"Available: {sorted(_SCORER_REGISTRY)}"
        )
    params = algo_config.get("params") or {}
    return cls(**params)


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def build_policy(policy_config: dict[str, Any]) -> Policy:
    """Construct a Policy from a policy config block.

    Example config::

        {"type": "top_k", "k": 3}
    """
    policy_type = policy_config["type"]
    if policy_type == "top_k":
        k = policy_config.get("k", 1)
        return TopKPolicy(k=k)
    raise ValueError(
        f"Unknown policy type '{policy_type}'. Available: ['top_k']"
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_agent(path: str | Path) -> Agent:
    """Load an Agent definition from a YAML file.

    Expected YAML structure::

        id: minimax_d3_topk3
        algo:
          type: minimax
          params:
            depth: 3
        policy:
          type: top_k
          k: 3

    Returns:
        An Agent instance (scorer + policy, no RNG yet).
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    agent_id: str = data["id"]
    algo_config = data["algo"]
    policy_config = data["policy"]
    scorer = build_scorer(algo_config)
    policy = build_policy(policy_config)

    # Extract config metadata for canonical identity.
    algo_type = algo_config["type"]
    algo_params = dict(algo_config.get("params") or {})
    policy_type = policy_config["type"]
    policy_params = {k: v for k, v in policy_config.items() if k != "type"}

    return Agent(
        id=agent_id,
        scorer=scorer,
        policy=policy,
        algo_type=algo_type,
        algo_params=algo_params,
        policy_type=policy_type,
        policy_params=policy_params,
    )


def load_agents_from_dir(directory: str | Path) -> list[Agent]:
    """Load all .yaml / .yml agent definitions from a directory.

    Returns agents sorted by id for deterministic ordering.
    """
    directory = Path(directory)
    agents: list[Agent] = []
    for path in sorted(directory.glob("*.yaml")):
        agents.append(load_agent(path))
    for path in sorted(directory.glob("*.yml")):
        agents.append(load_agent(path))
    return agents
