"""Agent Service — standalone FastAPI server.

Exposes Control Plane and Gameplay Plane endpoints.
The Backend communicates with this service via HTTP.
"""

from __future__ import annotations

from typing import Any, Literal
import os
import secrets
import threading

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_system.runtime.service.service import AgentService

# ── App ────────────────────────────────────────────────────────

app = FastAPI(title="Quoridor Agent Service", version="0.2.0")
service = AgentService()
research_init_lock = threading.Lock()


@app.middleware("http")
async def research_only(request: Request, call_next):
    if (os.environ.get("QUORIDOR_RESEARCH_ONLY") == "1" and request.url.path != "/health"
            and not request.url.path.startswith("/agent/research/")):
        return JSONResponse({"detail": "Legacy controls disabled in isolated research Runtime"}, status_code=404)
    return await call_next(request)


def research_service(authorization: str = Header(default="")):
    key = os.environ.get("QUORIDOR_RESEARCH_SERVICE_KEY", "")
    if not key:
        raise HTTPException(404, "Research sessions are disabled")
    if not secrets.compare_digest(authorization, "Bearer " + key):
        raise HTTPException(403, "Research service authorization required")
    with research_init_lock:
        if not hasattr(app.state, "research"):
            from agent_system.runtime.service.research import ResearchSessions
            import torch
            torch.set_num_threads(1)
            app.state.research = ResearchSessions(
                os.environ["QUORIDOR_RESEARCH_CATALOG"],
                os.environ["QUORIDOR_RESEARCH_ROOTS"].split(os.pathsep))
    return app.state.research


class ResearchCreateRequest(BaseModel):
    sha256: str
    room_id: str
    seat: Literal[1, 2]


class ResearchActionRequest(BaseModel):
    session_id: str
    token: str
    room_id: str
    seat: Literal[1, 2]
    game_state: dict[str, Any]
    legal_actions: list[dict[str, Any]]


class ResearchDestroyRequest(BaseModel):
    session_id: str
    token: str


@app.post("/agent/research/create")
def create_research(body: ResearchCreateRequest, manager=Depends(research_service)):
    try:
        return manager.create(body.sha256, body.room_id, body.seat)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/agent/research/action")
def research_action(body: ResearchActionRequest, manager=Depends(research_service)):
    try:
        return manager.action(body.session_id, body.token, body.room_id, body.seat,
                              body.game_state, body.legal_actions)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/agent/research/destroy")
def destroy_research(body: ResearchDestroyRequest, manager=Depends(research_service)):
    try:
        manager.destroy(body.session_id, body.token)
        return {"ok": True}
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


# ── Request/Response Schemas ───────────────────────────────────

class CreateAgentRequest(BaseModel):
    agent_type: str
    room_id: str
    seat: Literal[1, 2]
    config: dict[str, Any] | None = None


class ConfigureAgentRequest(BaseModel):
    instance_id: str
    config: dict[str, Any]


class DestroyAgentRequest(BaseModel):
    instance_id: str


class DestroyRoomAgentsRequest(BaseModel):
    room_id: str


class StartAgentRequest(BaseModel):
    instance_id: str


class StopAgentRequest(BaseModel):
    instance_id: str


class ActionRequest(BaseModel):
    room_id: str
    seat: Literal[1, 2]
    game_state: dict[str, Any]
    legal_actions: list[dict[str, Any]]


class HasAgentRequest(BaseModel):
    room_id: str
    seat: Literal[1, 2]


# ── Health ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Control Plane ──────────────────────────────────────────────

@app.get("/agent/types")
def list_types():
    return {"agent_types": service.list_types()}


@app.post("/agent/create")
def create_agent(body: CreateAgentRequest):
    try:
        instance_id = service.create_agent(
            agent_type=body.agent_type,
            room_id=body.room_id,
            seat=body.seat,
            config=body.config,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"instance_id": instance_id}


@app.post("/agent/configure")
def configure_agent(body: ConfigureAgentRequest):
    try:
        service.configure_agent(body.instance_id, body.config)
    except (KeyError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/agent/start")
def start_agent(body: StartAgentRequest):
    try:
        service.start_agent(body.instance_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/agent/stop")
def stop_agent(body: StopAgentRequest):
    try:
        service.stop_agent(body.instance_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/agent/destroy")
def destroy_agent(body: DestroyAgentRequest):
    service.destroy_agent(body.instance_id)
    return {"ok": True}


@app.post("/agent/destroy_room")
def destroy_room_agents(body: DestroyRoomAgentsRequest):
    service.destroy_room_agents(body.room_id)
    return {"ok": True}


class StartRoomAgentsRequest(BaseModel):
    room_id: str


@app.post("/agent/start_room")
def start_room_agents(body: StartRoomAgentsRequest):
    started = service.start_room_agents(body.room_id)
    return {"started": started}


# ── Gameplay Plane ─────────────────────────────────────────────

@app.post("/agent/action")
def get_action(body: ActionRequest):
    try:
        action = service.get_action(
            room_id=body.room_id,
            seat=body.seat,
            game_state=body.game_state,
            legal_actions=body.legal_actions,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"action": action}


class AdvanceAgentRequest(BaseModel):
    room_id: str
    seat: Literal[1, 2]


@app.post("/agent/advance")
def advance_agent(body: AdvanceAgentRequest):
    """Advance the agent's cursor after an ACCEPTED action (replay agents)."""
    service.advance_agent(body.room_id, body.seat)
    return {"ok": True}


class AgentCategoryRequest(BaseModel):
    room_id: str
    seat: Literal[1, 2]


@app.post("/agent/category")
def get_agent_category(body: AgentCategoryRequest):
    category = service.get_agent_category(body.room_id, body.seat)
    return {"category": category}


@app.post("/agent/has_agent")
def has_agent(body: HasAgentRequest):
    """Check if an active agent exists for the given room/seat."""
    return {"has_agent": service.has_agent(body.room_id, body.seat)}


# ── Entrypoint ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agents.agent_service.server:app", host="0.0.0.0", port=8090, reload=True)
