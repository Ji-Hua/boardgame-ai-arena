"""AgentMaterializer — agent type factory for the Agent Service.

Defines how a registered agent type creates runtime instances.
This is NOT the canonical Agent Spec (which lives in agents.agent_spec).
The AgentMaterializer is a consumer-side factory that materializes
Agent Spec definitions into live BaseAgent instances."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from agent_system.runtime.service.base_agent import BaseAgent
from agent_system.runtime.service.specs.param_schema import ParamSchema


class AgentMaterializer(ABC):
    """Agent type materializer for the Agent Service.

    Declares identity, parameter schema, capabilities, and how to
    create a concrete agent instance from validated configuration.

    This is the deployment/runtime-side materialization interface,
    not the canonical definition. The canonical Agent Spec lives
    in ``agents.agent_spec.AgentSpec``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique type identifier (matches BaseAgent.type_id)."""
        ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def param_schema(self) -> ParamSchema: ...

    @property
    @abstractmethod
    def deterministic(self) -> bool: ...

    @property
    @abstractmethod
    def supports_explain(self) -> bool: ...

    @property
    def display_name(self) -> str:
        """Human-readable name. Defaults to self.name."""
        return self.name

    @property
    def category(self) -> str:
        """Agent category (ai, scripted, replay). Defaults to 'ai'."""
        return "ai"

    @abstractmethod
    def create_instance(
        self, config: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        """Create a live agent instance from validated config.

        Args:
            config: Validated parameter dict (already passed through param_schema).
            context: Optional runtime context (seed, game metadata, etc.).

        Returns:
            A BaseAgent ready for gameplay.
        """
        ...


class ClassAgentMaterializer(AgentMaterializer):
    """Materializer that wraps a BaseAgent class with no configurable params.

    Used to bring legacy / simple agents into the materializer system
    without rewriting their logic.
    """

    def __init__(
        self,
        cls: Type[BaseAgent],
        *,
        version: str = "0.1.0",
        deterministic: bool = False,
        supports_explain: bool = False,
    ) -> None:
        self._cls = cls
        self._version = version
        self._deterministic = deterministic
        self._supports_explain = supports_explain
        self._param_schema = ParamSchema()

    @property
    def name(self) -> str:
        return self._cls.type_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def param_schema(self) -> ParamSchema:
        return self._param_schema

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def supports_explain(self) -> bool:
        return self._supports_explain

    @property
    def display_name(self) -> str:
        return self._cls.display_name or self._cls.__name__

    @property
    def category(self) -> str:
        return self._cls.category or "ai"

    def create_instance(
        self, config: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> BaseAgent:
        agent = self._cls()
        if config:
            agent.configure(config)
        return agent
