"""Agent Instance Manager — tracks active agent instances."""

from __future__ import annotations

import uuid
from typing import Any

from agents.agent_service.base_agent import BaseAgent
from agents.agent_service.registry import AgentRegistry


class AgentInstance:
    """A live agent instance bound to a room/seat."""

    def __init__(self, instance_id: str, agent: BaseAgent, room_id: str, seat: int) -> None:
        self.instance_id = instance_id
        self.agent = agent
        self.room_id = room_id
        self.seat = seat
        self.active = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False


class AgentInstanceManager:
    """Creates, tracks, and tears down agent instances."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._instances: dict[str, AgentInstance] = {}
        # room_id+seat → instance_id for fast lookup
        self._binding: dict[str, str] = {}

    def _binding_key(self, room_id: str, seat: int) -> str:
        return f"{room_id}:{seat}"

    def create(self, agent_type: str, room_id: str, seat: int,
               config: dict[str, Any] | None = None) -> AgentInstance:
        instance_id = str(uuid.uuid4())
        agent = self._registry.create(agent_type, config)
        instance = AgentInstance(instance_id, agent, room_id, seat)
        self._instances[instance_id] = instance
        self._binding[self._binding_key(room_id, seat)] = instance_id
        return instance

    def get(self, instance_id: str) -> AgentInstance | None:
        return self._instances.get(instance_id)

    def get_by_seat(self, room_id: str, seat: int) -> AgentInstance | None:
        iid = self._binding.get(self._binding_key(room_id, seat))
        if iid is None:
            return None
        return self._instances.get(iid)

    def start(self, instance_id: str) -> None:
        inst = self._instances.get(instance_id)
        if inst:
            inst.start()

    def stop(self, instance_id: str) -> None:
        inst = self._instances.get(instance_id)
        if inst:
            inst.stop()

    def destroy(self, instance_id: str) -> None:
        inst = self._instances.pop(instance_id, None)
        if inst:
            key = self._binding_key(inst.room_id, inst.seat)
            self._binding.pop(key, None)

    def destroy_room(self, room_id: str) -> None:
        to_remove = [iid for iid, inst in self._instances.items()
                     if inst.room_id == room_id]
        for iid in to_remove:
            self.destroy(iid)

    def get_room_instances(self, room_id: str) -> list[AgentInstance]:
        """Return all instances bound to the given room."""
        return [inst for inst in self._instances.values() if inst.room_id == room_id]
