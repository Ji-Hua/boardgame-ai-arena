"""Agent Service — main entry point for agent management.

Provides the unified interface for the Backend to interact with agents.
All agent operations (create, start, stop, get_action) go through this service.
"""

from __future__ import annotations

from typing import Any

from agents.agent_service.registry import AgentRegistry
from agents.agent_service.instance_manager import AgentInstanceManager
from agents.agent_service.specs.agent_spec import AgentSpec, ClassAgentSpec
from agents.agent_service.specs.builtin_specs import (
    RandomV2Spec,
    GreedySpec,
    MinimaxSpec,
)
from agents.agent_service.agents import (
    DummyAgent, RandomAgent,
    MinimaxAgentSimple, MinimaxAgentMedium, MinimaxAgentHard,
    ReplayAgent,
)


class AgentService:
    """Unified agent lifecycle and decision service."""

    def __init__(self) -> None:
        self._registry = AgentRegistry()
        self._instance_mgr = AgentInstanceManager(self._registry)
        self._register_builtin_agents()

    def _register_builtin_agents(self) -> None:
        # Primary agents — dedicated specs with param_schema
        self._registry.register(RandomV2Spec())
        self._registry.register(GreedySpec())
        self._registry.register(MinimaxSpec())

        # Legacy / simple agents — wrapped via ClassAgentSpec
        self._registry.register(ClassAgentSpec(DummyAgent, deterministic=True))
        self._registry.register(ClassAgentSpec(RandomAgent))
        self._registry.register(ClassAgentSpec(MinimaxAgentSimple, deterministic=True))
        self._registry.register(ClassAgentSpec(MinimaxAgentMedium, deterministic=True))
        self._registry.register(ClassAgentSpec(MinimaxAgentHard, deterministic=True))
        self._registry.register(ClassAgentSpec(ReplayAgent, deterministic=True))

    def register_agent_type(self, spec: AgentSpec) -> None:
        """Register an external agent spec."""
        self._registry.register(spec)

    # --- Control Plane ---

    def list_types(self) -> list[dict]:
        return self._registry.list_types()

    def create_agent(self, agent_type: str, room_id: str, seat: int,
                     config: dict[str, Any] | None = None) -> str:
        """Create an agent instance. Returns instance_id."""
        instance = self._instance_mgr.create(agent_type, room_id, seat, config)
        return instance.instance_id

    def configure_agent(self, instance_id: str, config: dict[str, Any]) -> None:
        """Configure an existing agent instance."""
        instance = self._instance_mgr.get(instance_id)
        if instance is None:
            raise KeyError(f"Unknown agent instance: {instance_id}")
        if instance.active:
            raise RuntimeError("Cannot configure an active agent")
        instance.agent.configure(config)

    def start_agent(self, instance_id: str) -> None:
        self._instance_mgr.start(instance_id)

    def stop_agent(self, instance_id: str) -> None:
        self._instance_mgr.stop(instance_id)

    def destroy_agent(self, instance_id: str) -> None:
        self._instance_mgr.destroy(instance_id)

    def destroy_room_agents(self, room_id: str) -> None:
        self._instance_mgr.destroy_room(room_id)

    def start_room_agents(self, room_id: str) -> list[dict]:
        """Start all inactive agents in a room. Returns list of started seats."""
        started = []
        for inst in self._instance_mgr.get_room_instances(room_id):
            if not inst.active:
                inst.start()
                started.append({"seat": inst.seat, "instance_id": inst.instance_id})
        return started

    # --- Gameplay Plane ---

    def get_action(self, room_id: str, seat: int,
                   game_state: dict, legal_actions: list[dict]) -> dict:
        """Request an action from the agent bound to room_id/seat.

        Returns an Action dict in wire format.
        Raises RuntimeError if no active agent is found.
        """
        instance = self._instance_mgr.get_by_seat(room_id, seat)
        if instance is None:
            raise RuntimeError(f"No agent instance for room={room_id} seat={seat}")
        if not instance.active:
            raise RuntimeError(f"Agent instance {instance.instance_id} is not active")
        return instance.agent.make_action(game_state, legal_actions)

    def advance_agent(self, room_id: str, seat: int) -> None:
        """Advance the agent's internal cursor (for replay agents).

        Call after an action is ACCEPTED by the engine.
        No-op for non-replay agents.
        """
        instance = self._instance_mgr.get_by_seat(room_id, seat)
        if instance is None:
            return
        if hasattr(instance.agent, "advance"):
            instance.agent.advance()

    def get_agent_category(self, room_id: str, seat: int) -> str | None:
        """Return the category of the agent at room_id/seat, or None."""
        instance = self._instance_mgr.get_by_seat(room_id, seat)
        if instance is None:
            return None
        return instance.agent.category

    def has_agent(self, room_id: str, seat: int) -> bool:
        instance = self._instance_mgr.get_by_seat(room_id, seat)
        return instance is not None and instance.active
