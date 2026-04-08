"""Agent Control REST routes — agent lifecycle management.

These endpoints allow the Frontend (or admin clients) to manage agents
via the Backend. All agent operations route through the Agent Service Adapter.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Literal, Any

agent_router = APIRouter(prefix="/api")


# --- Request schemas ---

class CreateAgentRequest(BaseModel):
    seat: Literal[1, 2]
    agent_type: str
    config: dict[str, Any] | None = None


# --- Helpers ---

def _agent_adapter(request: Request):
    return request.app.state.agent_adapter


def _room_mgr(request: Request):
    return request.app.state.room_manager


def _get_room_or_404(request: Request, room_id: str):
    room = _room_mgr(request).get_room(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
    return room


# --- Agent Control Endpoints ---

@agent_router.post("/rooms/{room_id}/agent/create")
def create_agent(room_id: str, body: CreateAgentRequest, request: Request):
    room = _get_room_or_404(request, room_id)
    adapter = _agent_adapter(request)

    if room.seats[body.seat].actor_type != "agent":
        raise HTTPException(
            status_code=400,
            detail=f"Seat {body.seat} is not configured as agent"
        )

    try:
        instance_id = adapter.create_agent(
            agent_type=body.agent_type,
            room_id=room_id,
            seat=body.seat,
            config=body.config,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"instance_id": instance_id, "room_id": room_id, "seat": body.seat}


@agent_router.post("/rooms/{room_id}/agent/start")
def start_agent(room_id: str, request: Request):
    _get_room_or_404(request, room_id)
    adapter = _agent_adapter(request)
    started = adapter.start_room_agents(room_id)
    return {"room_id": room_id, "started": started}


@agent_router.post("/rooms/{room_id}/agent/stop")
def stop_agent(room_id: str, request: Request):
    _get_room_or_404(request, room_id)
    adapter = _agent_adapter(request)
    adapter.destroy_room_agents(room_id)
    return {"room_id": room_id, "stopped": True}


@agent_router.get("/agent/types")
def list_agent_types(request: Request):
    adapter = _agent_adapter(request)
    return {"agent_types": adapter.list_types()}
