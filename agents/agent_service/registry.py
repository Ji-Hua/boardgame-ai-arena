"""Agent Registry — spec-based agent type registry.

Maps agent type names to AgentSpec instances. Creates agents exclusively
through the spec path: register(spec) → create_candidate() → create_instance().
"""

from __future__ import annotations

from typing import Any

from agents.agent_service.base_agent import BaseAgent
from agents.agent_service.specs.agent_spec import AgentSpec
from agents.agent_service.specs.candidate import Candidate


class AgentRegistry:
    """Registry of available agent types via AgentSpec.

    All agent creation goes through specs — the registry never
    instantiates agents directly.
    """

    def __init__(self) -> None:
        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register an AgentSpec. Keyed by spec.name."""
        if not spec.name:
            raise ValueError(f"AgentSpec {spec!r} has no name")
        self._specs[spec.name] = spec

    def get_spec(self, agent_type: str) -> AgentSpec:
        """Return the spec for the given type, or raise KeyError."""
        spec = self._specs.get(agent_type)
        if spec is None:
            raise KeyError(f"Unknown agent type: {agent_type}")
        return spec

    def create_candidate(
        self,
        agent_type: str,
        params: dict[str, Any] | None = None,
        *,
        version_tag: str | None = None,
        candidate_id: str | None = None,
    ) -> Candidate:
        """Create a Candidate after validating params against the spec's schema."""
        spec = self.get_spec(agent_type)
        validated = spec.param_schema.validate(params)
        return Candidate.create(
            agent_type=agent_type,
            params=validated,
            version_tag=version_tag or spec.version,
            candidate_id=candidate_id,
        )

    def create_instance(
        self,
        candidate: Candidate,
        context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        """Create a live agent from a Candidate via its spec."""
        spec = self.get_spec(candidate.agent_type)
        return spec.create_instance(candidate.params, context)

    def list_types(self) -> list[dict]:
        return [
            {
                "type_id": spec.name,
                "display_name": spec.display_name,
                "category": spec.category,
            }
            for spec in self._specs.values()
        ]

    def has_type(self, type_id: str) -> bool:
        return type_id in self._specs
