"""YamlAgentMaterializer — materializer backed by a YAML agent definition.

Bridges YAML agent definitions into the existing materializer-based registry so
that the Agent Service can instantiate agents from YAML without changing
the registry/instance-manager plumbing.
"""

from __future__ import annotations

from typing import Any

from agent_system.runtime.service.base_agent import BaseAgent
from agent_system.runtime.service.specs.agent_spec import AgentMaterializer
from agent_system.runtime.service.specs.param_schema import ParamDef, ParamSchema
from agent_system.runtime.service.yaml_loader import AgentDefinition, create_agent_from_definition


# Param schemas per algo type (used for API-level validation).
_PARAM_SCHEMAS: dict[str, ParamSchema] = {
    "random_v2": ParamSchema({
        "threshold": ParamDef(type=float, default=0.8, min_val=0.0, max_val=1.0),
    }),
    "greedy": ParamSchema(),
    "minimax": ParamSchema({
        "depth": ParamDef(type=int, default=2, min_val=1, max_val=10),
    }),
}


class YamlAgentMaterializer(AgentMaterializer):
    """Materializer backed by an AgentDefinition loaded from YAML.

    This is the primary materializer type for builtin heuristic agents.
    It delegates construction to ``create_agent_from_definition()``.
    """

    def __init__(self, defn: AgentDefinition) -> None:
        self._defn = defn

    @property
    def name(self) -> str:
        return self._defn.id

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def param_schema(self) -> ParamSchema:
        return _PARAM_SCHEMAS.get(self._defn.algo_type, ParamSchema())

    @property
    def deterministic(self) -> bool:
        return self._defn.deterministic

    @property
    def supports_explain(self) -> bool:
        return False

    @property
    def display_name(self) -> str:
        return self._defn.display_name

    @property
    def category(self) -> str:
        return self._defn.category

    def create_instance(
        self, config: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        """Create an agent from the YAML definition + runtime overrides.

        ``config`` may contain runtime overrides (policy, depth, etc.).
        ``context`` may contain seed and other runtime metadata.
        """
        # Extract policy from config overrides before param_schema validation
        config = dict(config or {})
        policy_override = config.pop("policy", None)
        validated = self.param_schema.validate(config)

        # Re-inject policy if provided as override
        if policy_override is not None:
            validated["policy"] = policy_override

        return create_agent_from_definition(self._defn, validated, context)
