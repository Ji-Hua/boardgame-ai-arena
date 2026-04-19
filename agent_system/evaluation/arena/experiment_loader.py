"""Experiment YAML loader — parse experiment definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Match:
    """A single matchup between two agents."""

    agent_1_id: str
    agent_2_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """A complete experiment definition."""

    id: str
    matches: list[Match]


def load_experiment(path: str | Path) -> Experiment:
    """Load an Experiment from a YAML file.

    Expected YAML structure::

        id: basic_matchups
        matches:
          - agent_1: random
            agent_2: minimax_d2
            params:
              num_games: 50

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the schema is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid experiment YAML: expected a mapping, got {type(data).__name__}")

    exp_id = data.get("id")
    if not exp_id or not isinstance(exp_id, str):
        raise ValueError("Experiment YAML must have a non-empty string 'id'")

    raw_matches = data.get("matches")
    if not raw_matches or not isinstance(raw_matches, list):
        raise ValueError("Experiment YAML must have a non-empty 'matches' list")

    matches: list[Match] = []
    for i, entry in enumerate(raw_matches):
        if not isinstance(entry, dict):
            raise ValueError(f"Match #{i}: expected a mapping, got {type(entry).__name__}")

        agent_1 = entry.get("agent_1")
        agent_2 = entry.get("agent_2")

        if not agent_1 or not isinstance(agent_1, str):
            raise ValueError(f"Match #{i}: 'agent_1' must be a non-empty string")
        if not agent_2 or not isinstance(agent_2, str):
            raise ValueError(f"Match #{i}: 'agent_2' must be a non-empty string")

        params = entry.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"Match #{i}: 'params' must be a mapping")

        matches.append(Match(agent_1_id=agent_1, agent_2_id=agent_2, params=params))

    return Experiment(id=exp_id, matches=matches)
