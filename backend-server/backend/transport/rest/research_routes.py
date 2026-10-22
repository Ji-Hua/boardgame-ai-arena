"""Capability-protected human games using existing game and inference adapters."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import secrets
import time
import os
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request, Depends
from pydantic import BaseModel, Field


def gateway_authorization(x_research_gateway: str = Header(default="")):
    expected = os.environ.get("QUORIDOR_RESEARCH_SERVICE_KEY", "")
    if not expected or not secrets.compare_digest(x_research_gateway, expected):
        raise HTTPException(403, "Research gateway authorization required")


router = APIRouter(prefix="/api/research/games", dependencies=[Depends(gateway_authorization)])


class CreateGame(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_seat: Literal[1, 2] = 1


class SubmitAction(BaseModel):
    expected_step: int = Field(ge=0)
    type: Literal["pawn", "horizontal", "vertical"]
    target: tuple[int, int]


@dataclass
class ResearchGame:
    room_id: str
    game: object
    token: str
    human_seat: int
    binding: dict
    initial_state: dict
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: bool = False
    error: str | None = None
    released: bool = False
    task: asyncio.Task | None = None
    created: float = field(default_factory=time.monotonic)

    def snapshot(self):
        projected = self.game.engine.research_snapshot()
        return dict(room_id=self.room_id, game_id=self.game.game_id,
            phase=self.game.phase, step_count=self.game.step_count,
            human_seat=self.human_seat, identity=self.binding["identity"],
            initial_state=self.initial_state, actions=list(self.game.actions),
            result=self.game.result, pending=self.pending, error=self.error, **projected)


def get_game(request, identifier, authorization):
    session = request.app.state.research_games.get(identifier)
    if session is None or not secrets.compare_digest(authorization, "Bearer " + session.token):
        raise HTTPException(404, "Research game not found or unauthorized; no game was recreated")
    return session


async def release(request, session):
    if not session.released:
        try:
            await request.app.state.agent_adapter.destroy_research(session.binding)
            session.released = True
        except Exception:
            session.error = "Runtime cleanup unavailable; the instance expires automatically"


async def drive_agent(request, session):
    async with session.lock:
        try:
            if session.game.phase != "running":
                return
            before = session.game.step_count
            projected = session.game.engine.research_snapshot()
            if projected["state"]["current_player"] == session.human_seat:
                return
            action = await request.app.state.agent_adapter.research_action(
                session.binding, projected["state"], projected["legal_actions"])
            if session.game.step_count != before or session.game.phase != "running":
                raise RuntimeError("Stale Agent action rejected")
            result = request.app.state.game_manager.submit_action(session.game, action)
            if not result["success"]:
                raise RuntimeError("Engine rejected Agent action: " + result["error"])
            session.error = None
            if result.get("game_over"):
                await release(request, session)
        except Exception as exc:
            session.error = str(exc)
        finally:
            session.pending = False


def schedule(request, session):
    if not session.pending and session.game.phase == "running":
        session.pending = True
        session.error = None
        session.task = asyncio.create_task(drive_agent(request, session))


@router.post("")
async def create_game(body: CreateGame, request: Request):
    sessions = request.app.state.research_games
    if len(sessions) >= 16:
        raise HTTPException(429, "Research game capacity reached; close an old game")
    room_id = "research-" + secrets.token_urlsafe(18)
    try:
        binding = await request.app.state.agent_adapter.create_research(
            body.sha256, room_id, 3 - body.human_seat)
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(503, str(exc)) from exc
    identity = binding.get("identity", {})
    if (identity.get("actual_sha256") != body.sha256 or identity.get("sha256") != body.sha256
            or identity.get("room_id") != room_id or identity.get("agent_seat") != 3 - body.human_seat):
        await request.app.state.agent_adapter.destroy_research(binding)
        raise HTTPException(502, "Runtime returned an unexpected model identity")
    game = request.app.state.game_manager.create_game()
    session = ResearchGame(room_id, game, secrets.token_urlsafe(32), body.human_seat,
        binding, game.engine.research_snapshot()["state"])
    sessions[game.game_id] = session
    if body.human_seat == 2:
        schedule(request, session)
    return dict(session.snapshot(), token=session.token)


@router.get("/{identifier}")
async def snapshot(identifier: str, request: Request, authorization: str = Header(default="")):
    return get_game(request, identifier, authorization).snapshot()


@router.post("/{identifier}/actions")
async def submit(identifier: str, body: SubmitAction, request: Request,
                 authorization: str = Header(default="")):
    session = get_game(request, identifier, authorization)
    if session.pending:
        raise HTTPException(409, "Agent is deciding")
    async with session.lock:
        if body.expected_step != session.game.step_count:
            raise HTTPException(409, "Stale action; reload the authoritative position")
        if session.game.get_state()["current_player"] != session.human_seat:
            raise HTTPException(409, "Not the human player's turn")
        action = dict(player=session.human_seat, type=body.type, target=list(body.target))
        result = request.app.state.game_manager.submit_action(session.game, action)
        if not result["success"]:
            raise HTTPException(422, result["error"])
        if result.get("game_over"):
            await release(request, session)
        else:
            schedule(request, session)
        return session.snapshot()


@router.post("/{identifier}/retry")
async def retry(identifier: str, request: Request, authorization: str = Header(default="")):
    session = get_game(request, identifier, authorization)
    if session.game.get_state()["current_player"] == session.human_seat:
        raise HTTPException(409, "Not an Agent turn")
    schedule(request, session)
    return session.snapshot()


@router.post("/{identifier}/surrender")
async def surrender(identifier: str, request: Request, authorization: str = Header(default="")):
    session = get_game(request, identifier, authorization)
    async with session.lock:
        if session.game.phase != "running":
            raise HTTPException(409, "Game is already finished")
        request.app.state.game_manager.surrender(session.game, session.human_seat)
        await release(request, session)
        return session.snapshot()


@router.delete("/{identifier}")
async def close(identifier: str, request: Request, authorization: str = Header(default="")):
    session = get_game(request, identifier, authorization)
    async with session.lock:
        if session.game.phase == "running":
            request.app.state.game_manager.force_end(session.game)
        await release(request, session)
        del request.app.state.research_games[identifier]
        request.app.state.game_manager._games.pop(identifier, None)
        return {"closed": True}