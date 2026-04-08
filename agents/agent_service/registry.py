"""Agent Registry — maps type_id to agent factories."""

from __future__ import annotations

from typing import Any, Type

from agents.agent_service.base_agent import BaseAgent


class AgentTypeInfo:
    """Metadata for a registered agent type."""

    def __init__(self, type_id: str, display_name: str, category: str, cls: Type[BaseAgent]) -> None:
        self.type_id = type_id
        self.display_name = display_name
        self.category = category
        self.cls = cls


class AgentRegistry:
    """Registry of available agent types. Populated at service startup."""

    def __init__(self) -> None:
        self._types: dict[str, AgentTypeInfo] = {}

    def register(self, cls: Type[BaseAgent]) -> None:
        if not cls.type_id:
            raise ValueError(f"Agent class {cls.__name__} has no type_id")
        self._types[cls.type_id] = AgentTypeInfo(
            type_id=cls.type_id,
            display_name=cls.display_name or cls.__name__,
            category=cls.category,
            cls=cls,
        )

    def create(self, type_id: str, config: dict[str, Any] | None = None) -> BaseAgent:
        info = self._types.get(type_id)
        if info is None:
            raise KeyError(f"Unknown agent type: {type_id}")
        agent = info.cls()
        if config:
            agent.configure(config)
        return agent

    def list_types(self) -> list[dict]:
        return [
            {
                "type_id": info.type_id,
                "display_name": info.display_name,
                "category": info.category,
            }
            for info in self._types.values()
        ]

    def has_type(self, type_id: str) -> bool:
        return type_id in self._types
