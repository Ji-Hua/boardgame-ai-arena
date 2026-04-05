"""REST routes — room and game domain endpoints.

All routes follow the interface defined in backend-interface.md.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Literal

router = APIRouter(prefix="/api")


# --- Request schemas ---

class JoinRequest(BaseModel):
    client_id: str
    seat: Literal[1, 2]


class SelectActorRequest(BaseModel):
    seat: Literal[1, 2]
    actor_type: Literal["human", "agent"]


# --- Helpers ---

def _room_mgr(request: Request):
    return request.app.state.room_manager


def _game_mgr(request: Request):
    return request.app.state.game_manager


def _get_room_or_404(request: Request, room_id: str):
    room = _room_mgr(request).get_room(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
    return room


# --- Room Domain ---

@router.post("/rooms")
def create_room(request: Request):
    room = _room_mgr(request).create_room()
    return room.snapshot()


@router.get("/rooms")
def list_rooms(request: Request):
    return {"rooms": _room_mgr(request).list_rooms()}


@router.post("/rooms/{room_id}/join")
def join_room(room_id: str, body: JoinRequest, request: Request):
    try:
        room = _room_mgr(request).join_room(room_id, body.seat, body.client_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return room.snapshot()


@router.post("/rooms/{room_id}/select_actor")
def select_actor(room_id: str, body: SelectActorRequest, request: Request):
    try:
        room = _room_mgr(request).select_actor(room_id, body.seat, body.actor_type)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return room.snapshot()


@router.post("/rooms/{room_id}/start_game")
def start_game(room_id: str, request: Request):
    rm = _room_mgr(request)
    room = _get_room_or_404(request, room_id)

    if not rm.can_start(room):
        raise HTTPException(status_code=400, detail="Preconditions not met for game start")

    gm = _game_mgr(request)
    game = gm.create_game()
    rm.set_using(room, game.game_id)

    snapshot = room.snapshot()
    snapshot["game"] = game.to_dict()
    return snapshot


@router.post("/rooms/{room_id}/close")
def close_room(room_id: str, request: Request):
    try:
        room = _room_mgr(request).close_room(room_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return room.snapshot()


@router.post("/rooms/{room_id}/swap_seats")
def swap_seats(room_id: str, request: Request):
    try:
        room = _room_mgr(request).swap_seats(room_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return room.snapshot()


# --- Game Domain ---

@router.get("/rooms/{room_id}/game/state")
def get_game_state(room_id: str, request: Request):
    room = _get_room_or_404(request, room_id)
    if room.current_game_id is None:
        raise HTTPException(status_code=400, detail="No active game")

    game = _game_mgr(request).get_game(room.current_game_id)
    if game is None:
        raise HTTPException(status_code=400, detail="No active game")
    return game.to_dict()


@router.post("/rooms/{room_id}/game/new")
def new_game(room_id: str, request: Request):
    rm = _room_mgr(request)
    room = _get_room_or_404(request, room_id)

    if room.status != "config":
        raise HTTPException(status_code=400, detail=f"Room is {room.status}, must be config")

    if not rm.can_start(room):
        raise HTTPException(status_code=400, detail="Preconditions not met for game start")

    gm = _game_mgr(request)
    game = gm.create_game()
    rm.set_using(room, game.game_id)

    snapshot = room.snapshot()
    snapshot["game"] = game.to_dict()
    return snapshot


@router.post("/rooms/{room_id}/game/force_end")
def force_end(room_id: str, request: Request):
    rm = _room_mgr(request)
    room = _get_room_or_404(request, room_id)

    if room.status != "using":
        raise HTTPException(status_code=400, detail=f"Room is {room.status}, not using")

    game = _game_mgr(request).get_game(room.current_game_id)
    if game is None:
        raise HTTPException(status_code=400, detail="No active game")

    gm = _game_mgr(request)
    result = gm.force_end(game)
    rm.set_config(room)

    return {"room_id": room.room_id, "status": room.status, "result": result}


@router.get("/rooms/{room_id}/game/replay")
def get_replay(room_id: str, request: Request):
    room = _get_room_or_404(request, room_id)

    # Look for last game (could be current or finished)
    if room.current_game_id is None:
        raise HTTPException(status_code=400, detail="No completed game")

    game = _game_mgr(request).get_game(room.current_game_id)
    if game is None or game.phase != "finished":
        raise HTTPException(status_code=400, detail="No completed game")

    return {
        "game_id": game.game_id,
        "actions": game.actions,
        "result": game.result,
    }
