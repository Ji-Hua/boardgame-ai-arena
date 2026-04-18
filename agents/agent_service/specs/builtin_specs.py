"""Builtin agent specs — wraps existing agents into the spec system.

Each spec defines param_schema and create_instance() without rewriting
any agent logic. The original agent classes are used as-is.
"""

from __future__ import annotations

from typing import Any

from agents.agent_service.base_agent import BaseAgent
from agents.agent_service.specs.agent_spec import AgentSpec
from agents.agent_service.specs.param_schema import ParamDef, ParamSchema


# ---------------------------------------------------------------------------
# RandomV2Spec
# ---------------------------------------------------------------------------

class RandomV2Spec(AgentSpec):
    """Spec for RandomAgentV2 — weighted random with pawn bias threshold."""

    @property
    def name(self) -> str:
        return "random_v2"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def param_schema(self) -> ParamSchema:
        return ParamSchema({
            "threshold": ParamDef(type=float, default=0.8, min_val=0.0, max_val=1.0),
        })

    @property
    def deterministic(self) -> bool:
        return False

    @property
    def supports_explain(self) -> bool:
        return False

    @property
    def display_name(self) -> str:
        return "Random Agent V2"

    @property
    def category(self) -> str:
        return "scripted"

    def create_instance(
        self, config: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        from agents.agent_service.agents.random_agent import RandomAgentV2
        params = self.param_schema.validate(config)
        return RandomAgentV2(threshold=params["threshold"])


# ---------------------------------------------------------------------------
# GreedySpec
# ---------------------------------------------------------------------------

class GreedySpec(AgentSpec):
    """Spec for GreedyAgent — one-step lookahead via shortest-path heuristic."""

    @property
    def name(self) -> str:
        return "greedy"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def param_schema(self) -> ParamSchema:
        return ParamSchema()

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def supports_explain(self) -> bool:
        return False

    @property
    def display_name(self) -> str:
        return "Greedy Agent"

    @property
    def category(self) -> str:
        return "ai"

    def create_instance(
        self, config: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        from agents.agent_service.agents.greedy_agent import GreedyAgent
        return GreedyAgent()


# ---------------------------------------------------------------------------
# MinimaxSpec
# ---------------------------------------------------------------------------

class MinimaxSpec(AgentSpec):
    """Spec for MinimaxAgent — depth-configurable minimax with alpha-beta."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def param_schema(self) -> ParamSchema:
        return ParamSchema({
            "depth": ParamDef(type=int, default=2, min_val=1, max_val=10),
        })

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def supports_explain(self) -> bool:
        return False

    @property
    def display_name(self) -> str:
        return "Minimax Agent"

    @property
    def category(self) -> str:
        return "ai"

    def create_instance(
        self, config: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        from agents.agent_service.agents.minimax_agent import MinimaxAgent
        params = self.param_schema.validate(config)
        return MinimaxAgent(depth=params["depth"])
