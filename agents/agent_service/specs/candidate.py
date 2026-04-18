"""Candidate — immutable, serializable description of an evaluatable agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """The smallest evaluatable unit.

    Data-only object describing a specific agent configuration.
    NOT a runtime object — use AgentSpec.create_instance() to get a live agent.

    Properties:
        - Immutable (frozen dataclass)
        - Serializable (to_dict / from_dict)
        - Reproducible (same candidate → same agent behavior)
    """

    id: str
    agent_type: str
    params: dict[str, Any] = field(default_factory=dict)
    version_tag: str | None = None

    def __post_init__(self) -> None:
        # Defensive copy so the caller's dict can't mutate our state.
        object.__setattr__(self, "params", dict(self.params))

    @staticmethod
    def create(
        agent_type: str,
        params: dict[str, Any] | None = None,
        *,
        version_tag: str | None = None,
        candidate_id: str | None = None,
    ) -> Candidate:
        """Convenience factory with auto-generated id."""
        return Candidate(
            id=candidate_id or str(uuid.uuid4()),
            agent_type=agent_type,
            params=dict(params or {}),
            version_tag=version_tag,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "params": dict(self.params),
            "version_tag": self.version_tag,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Candidate:
        return Candidate(
            id=data["id"],
            agent_type=data["agent_type"],
            params=data.get("params", {}),
            version_tag=data.get("version_tag"),
        )
