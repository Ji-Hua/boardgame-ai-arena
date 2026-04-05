"""Broadcast Hub — WebSocket state broadcasting.

Manages per-room subscriber lists and broadcasts events to all
connected clients in a room.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import WebSocket


class BroadcastHub:
    """Per-room WebSocket subscriber management and broadcasting."""

    def __init__(self) -> None:
        # room_id -> {client_id -> WebSocket}
        self._subscribers: dict[str, dict[str, WebSocket]] = {}

    def subscribe(self, room_id: str, client_id: str, ws: WebSocket) -> None:
        if room_id not in self._subscribers:
            self._subscribers[room_id] = {}
        self._subscribers[room_id][client_id] = ws

    def unsubscribe(self, room_id: str, client_id: str) -> None:
        if room_id in self._subscribers:
            self._subscribers[room_id].pop(client_id, None)
            if not self._subscribers[room_id]:
                del self._subscribers[room_id]

    async def send_to(self, ws: WebSocket, event: dict) -> None:
        try:
            await ws.send_json(event)
        except Exception:
            pass

    async def broadcast(self, room_id: str, event: dict) -> None:
        subs = self._subscribers.get(room_id, {})
        for ws in list(subs.values()):
            await self.send_to(ws, event)

    async def broadcast_except(self, room_id: str, event: dict, exclude_client: str) -> None:
        subs = self._subscribers.get(room_id, {})
        for cid, ws in list(subs.items()):
            if cid != exclude_client:
                await self.send_to(ws, event)
