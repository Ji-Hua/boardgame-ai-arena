"""Agent Service Adapter — backend interface to the Agent Service.

This adapter isolates the backend from the Agent Service implementation.
All agent interactions from the backend go through this module.

Communicates with the Agent Service via HTTP (standalone service).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import os

import httpx


logger = logging.getLogger(__name__)

# Default timeout for agent action requests (seconds)
AGENT_ACTION_TIMEOUT = 5.0

# Agent Service base URL — overridable via environment variable
AGENT_SERVICE_URL = os.environ.get("AGENT_SERVICE_URL", "http://localhost:8090")


class AgentServiceAdapter:
    """Backend adapter for the Agent Service.

    All agent operations go through HTTP to the standalone Agent Service.
    """

    def __init__(self, base_url: str = AGENT_SERVICE_URL) -> None:
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=10.0)
        self._async_client = httpx.AsyncClient(base_url=base_url, timeout=AGENT_ACTION_TIMEOUT)

    # --- Control Plane ---

    def list_types(self) -> list[dict]:
        resp = self._client.get("/agent/types")
        resp.raise_for_status()
        return resp.json()["agent_types"]

    def create_agent(self, agent_type: str, room_id: str, seat: int,
                     config: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "agent_type": agent_type,
            "room_id": room_id,
            "seat": seat,
        }
        if config is not None:
            payload["config"] = config
        resp = self._client.post("/agent/create", json=payload)
        resp.raise_for_status()
        return resp.json()["instance_id"]

    def start_agent(self, instance_id: str) -> None:
        resp = self._client.post("/agent/start", json={"instance_id": instance_id})
        resp.raise_for_status()

    def stop_agent(self, instance_id: str) -> None:
        resp = self._client.post("/agent/stop", json={"instance_id": instance_id})
        resp.raise_for_status()

    def destroy_agent(self, instance_id: str) -> None:
        resp = self._client.post("/agent/destroy", json={"instance_id": instance_id})
        resp.raise_for_status()

    def destroy_room_agents(self, room_id: str) -> None:
        resp = self._client.post("/agent/destroy_room", json={"room_id": room_id})
        resp.raise_for_status()

    def start_room_agents(self, room_id: str) -> list[dict]:
        """Start all agents in a room. Returns list of started seats."""
        resp = self._client.post("/agent/start_room", json={"room_id": room_id})
        resp.raise_for_status()
        return resp.json()["started"]

    # --- Gameplay Plane ---

    async def request_action(self, room_id: str, seat: int,
                             game_state: dict, legal_actions: list[dict],
                             timeout: float = AGENT_ACTION_TIMEOUT) -> dict:
        """Request an action from an agent. Returns action dict.

        Raises:
            TimeoutError: If the agent does not respond in time.
            RuntimeError: If no active agent is found or agent errors.
        """
        try:
            resp = await self._async_client.post(
                "/agent/action",
                json={
                    "room_id": room_id,
                    "seat": seat,
                    "game_state": game_state,
                    "legal_actions": legal_actions,
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Agent action failed: {resp.status_code} {resp.text}"
                )
            return resp.json()["action"]
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Agent for room={room_id} seat={seat} did not respond within {timeout}s"
            )

    def has_agent(self, room_id: str, seat: int) -> bool:
        resp = self._client.post(
            "/agent/has_agent",
            json={"room_id": room_id, "seat": seat},
        )
        resp.raise_for_status()
        return resp.json()["has_agent"]

    def advance_agent(self, room_id: str, seat: int) -> None:
        """Advance agent cursor after ACCEPTED action (replay agents)."""
        resp = self._client.post(
            "/agent/advance",
            json={"room_id": room_id, "seat": seat},
        )
        resp.raise_for_status()

    def get_agent_category(self, room_id: str, seat: int) -> str | None:
        """Get the category of the agent at the given room/seat."""
        resp = self._client.post(
            "/agent/category",
            json={"room_id": room_id, "seat": seat},
        )
        resp.raise_for_status()
        return resp.json()["category"]
