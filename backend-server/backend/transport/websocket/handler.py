"""WebSocket handler — room-scoped real-time events.

Handles subscribe, take_action, validate_action, surrender.
Broadcasts state_update, game_started, game_ended.
Triggers agent turns via Turn Orchestrator when applicable.
"""

from __future__ import annotations

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.application.room_manager import RoomManager
from backend.application.game_manager import GameManager
from backend.adapters.agent_service_adapter import AgentServiceAdapter
from backend.runtime.broadcast import BroadcastHub
from backend.runtime.orchestrator import maybe_trigger_agent_turn

ws_router = APIRouter()


@ws_router.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: str,
):
    rm: RoomManager = websocket.app.state.room_manager
    gm: GameManager = websocket.app.state.game_manager
    hub: BroadcastHub = websocket.app.state.broadcast_hub
    agent_adapter: AgentServiceAdapter = websocket.app.state.agent_adapter

    room = rm.get_room(room_id)
    if room is None:
        await websocket.close(code=4004, reason="Room not found")
        return

    await websocket.accept()

    client_id: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "INVALID_PAYLOAD", "Invalid JSON")
                continue

            msg_type = msg.get("type")

            if msg_type == "subscribe":
                client_id = msg.get("client_id")
                if not client_id:
                    await _send_error(websocket, "INVALID_PAYLOAD", "client_id required")
                    continue
                hub.subscribe(room_id, client_id, websocket)
                await _send_room_snapshot(websocket, room, gm)

            elif msg_type == "take_action":
                if client_id is None:
                    await _send_error(websocket, "UNBOUND_CLIENT", "Must subscribe first")
                    continue
                await _handle_take_action(websocket, msg, room, gm, hub, room_id, rm, agent_adapter)

            elif msg_type == "validate_action":
                if client_id is None:
                    await _send_error(websocket, "UNBOUND_CLIENT", "Must subscribe first")
                    continue
                await _handle_validate_action(websocket, msg, room, gm)

            elif msg_type == "get_legal_actions":
                if client_id is None:
                    await _send_error(websocket, "UNBOUND_CLIENT", "Must subscribe first")
                    continue
                await _handle_get_legal_actions(websocket, room, gm)

            elif msg_type == "surrender":
                if client_id is None:
                    await _send_error(websocket, "UNBOUND_CLIENT", "Must subscribe first")
                    continue
                await _handle_surrender(websocket, msg, room, gm, hub, room_id, rm, agent_adapter)

            else:
                await _send_error(websocket, "INVALID_PAYLOAD", f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        pass
    finally:
        if client_id:
            hub.unsubscribe(room_id, client_id)


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json({"type": "error", "code": code, "message": message})


async def _send_room_snapshot(ws: WebSocket, room, gm: GameManager) -> None:
    game_data = None
    if room.current_game_id:
        game = gm.get_game(room.current_game_id)
        if game:
            game_data = game.to_dict()

    await ws.send_json({
        "type": "room_snapshot",
        "room_id": room.room_id,
        "status": room.status,
        "seats": {
            "1": room.seats[1].to_dict(),
            "2": room.seats[2].to_dict(),
        },
        "game": game_data,
    })


async def _handle_take_action(ws, msg, room, gm: GameManager, hub: BroadcastHub, room_id: str, rm: RoomManager, agent_adapter: AgentServiceAdapter):
    if room.status != "using" or room.current_game_id is None:
        await _send_error(ws, "NO_ACTIVE_GAME", "No active game in this room")
        return

    game = gm.get_game(room.current_game_id)
    if game is None:
        await _send_error(ws, "NO_ACTIVE_GAME", "No active game")
        return

    action = msg.get("action")
    if not action:
        await _send_error(ws, "INVALID_PAYLOAD", "action field required")
        return

    result = gm.submit_action(game, action)

    # Send action_result to submitter
    await ws.send_json({
        "type": "action_result",
        "success": result["success"],
        "error": result.get("error"),
    })

    if result["success"]:
        # Broadcast state_update to all subscribers
        await hub.broadcast(room_id, {
            "type": "state_update",
            "game_id": game.game_id,
            "state": result["state"],
            "last_action": action,
            "step_count": game.step_count,
        })

        if result.get("game_over"):
            await hub.broadcast(room_id, {
                "type": "game_ended",
                "game_id": game.game_id,
                "result": result["result"],
            })
            rm.set_config(room)
            agent_adapter.destroy_room_agents(room_id)
        else:
            # After a successful human action, check if the next turn belongs to an agent
            await maybe_trigger_agent_turn(room, room_id, gm, rm, hub, agent_adapter)


async def _handle_validate_action(ws, msg, room, gm: GameManager):
    if room.status != "using" or room.current_game_id is None:
        await _send_error(ws, "NO_ACTIVE_GAME", "No active game in this room")
        return

    game = gm.get_game(room.current_game_id)
    if game is None:
        await _send_error(ws, "NO_ACTIVE_GAME", "No active game")
        return

    action = msg.get("action")
    if not action:
        await _send_error(ws, "INVALID_PAYLOAD", "action field required")
        return

    result = game.engine.validate_action(action)
    await ws.send_json({
        "type": "validate_result",
        "valid": result["valid"],
        "reason": result.get("reason"),
    })


async def _handle_get_legal_actions(ws: WebSocket, room, gm: GameManager):
    """Return all legal pawn moves for the current player."""
    if room.status != "using" or room.current_game_id is None:
        await ws.send_json({"type": "legal_actions_result", "actions": []})
        return

    game = gm.get_game(room.current_game_id)
    if game is None:
        await ws.send_json({"type": "legal_actions_result", "actions": []})
        return

    actions = game.engine.legal_pawn_actions()
    await ws.send_json({"type": "legal_actions_result", "actions": actions})


async def _handle_surrender(ws, msg, room, gm: GameManager, hub: BroadcastHub, room_id: str, rm: RoomManager, agent_adapter: AgentServiceAdapter):
    if room.status != "using" or room.current_game_id is None:
        await _send_error(ws, "NO_ACTIVE_GAME", "No active game in this room")
        return

    game = gm.get_game(room.current_game_id)
    if game is None:
        await _send_error(ws, "NO_ACTIVE_GAME", "No active game")
        return

    seat = msg.get("seat")
    if seat not in (1, 2):
        await _send_error(ws, "INVALID_PAYLOAD", "seat must be 1 or 2")
        return

    result = gm.surrender(game, seat)

    await hub.broadcast(room_id, {
        "type": "game_ended",
        "game_id": game.game_id,
        "result": result,
    })
    rm.set_config(room)
    agent_adapter.destroy_room_agents(room_id)
