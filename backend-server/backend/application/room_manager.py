"""Room Manager — room lifecycle and seat binding.

Owns the in-memory room registry. All room configuration operations
go through this module.
"""

from __future__ import annotations

import uuid
from typing import Optional, Literal


class Seat:
    """Binding for a single seat in a room."""

    def __init__(self) -> None:
        self.client_id: Optional[str] = None
        self.actor_type: Optional[Literal["human", "agent"]] = None

    def to_dict(self) -> dict:
        return {"client_id": self.client_id, "actor_type": self.actor_type}


class Room:
    """A room holds two seats and manages configuration lifecycle."""

    def __init__(self) -> None:
        self.room_id: str = str(uuid.uuid4())
        self.status: Literal["config", "using", "closed"] = "config"
        self.seats: dict[int, Seat] = {1: Seat(), 2: Seat()}
        self.current_game_id: Optional[str] = None

    def snapshot(self) -> dict:
        return {
            "room_id": self.room_id,
            "status": self.status,
            "seats": {
                "1": self.seats[1].to_dict(),
                "2": self.seats[2].to_dict(),
            },
        }


class RoomManager:
    """In-memory room registry and lifecycle manager."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}

    def create_room(self) -> Room:
        room = Room()
        self._rooms[room.room_id] = room
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def list_rooms(self) -> list[dict]:
        return [r.snapshot() for r in self._rooms.values()]

    def join_room(self, room_id: str, seat: int, client_id: str) -> Room:
        room = self._require_room(room_id)
        self._require_config(room)

        if seat not in (1, 2):
            raise ValueError(f"Invalid seat: {seat}. Must be 1 or 2.")

        existing = room.seats[seat].client_id
        if existing is not None and existing != client_id:
            raise ValueError(f"Seat {seat} already taken by {existing}")

        for s in (1, 2):
            if room.seats[s].client_id == client_id and s != seat:
                raise ValueError(f"Client {client_id} already on seat {s}")

        room.seats[seat].client_id = client_id
        return room

    def select_actor(self, room_id: str, seat: int, actor_type: str) -> Room:
        room = self._require_room(room_id)
        self._require_config(room)

        if seat not in (1, 2):
            raise ValueError(f"Invalid seat: {seat}. Must be 1 or 2.")
        if actor_type not in ("human", "agent"):
            raise ValueError(f"Invalid actor_type: {actor_type}")

        room.seats[seat].actor_type = actor_type
        return room

    def can_start(self, room: Room) -> bool:
        if room.status != "config":
            return False
        for s in (1, 2):
            seat = room.seats[s]
            if seat.actor_type is None:
                return False
            if seat.actor_type == "human" and seat.client_id is None:
                return False
        return True

    def set_using(self, room: Room, game_id: str) -> None:
        room.status = "using"
        room.current_game_id = game_id

    def set_config(self, room: Room) -> None:
        room.status = "config"
        room.current_game_id = None

    def close_room(self, room_id: str) -> Room:
        room = self._require_room(room_id)
        if room.status == "using":
            raise ValueError("Cannot close room with active game")
        room.status = "closed"
        return room

    def swap_seats(self, room_id: str) -> Room:
        room = self._require_room(room_id)
        self._require_config(room)
        s1, s2 = room.seats[1], room.seats[2]
        s1.client_id, s2.client_id = s2.client_id, s1.client_id
        s1.actor_type, s2.actor_type = s2.actor_type, s1.actor_type
        return room

    def _require_room(self, room_id: str) -> Room:
        room = self._rooms.get(room_id)
        if room is None:
            raise KeyError(f"Room {room_id} not found")
        return room

    def _require_config(self, room: Room) -> None:
        if room.status != "config":
            raise ValueError(f"Room is {room.status}, not config")
