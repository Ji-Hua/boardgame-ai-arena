# Quoridor Backend Interface

Author: Ji Hua  
Created Date: 2026-04-03  
Last Modified: 2026-04-03  
Current Version: 1  
Document Type: Interface  
Document Subtype: Backend API Contract  
Document Status: In Development  
Document Authority Scope: Backend module  
Document Purpose:  
This document defines the external API contract of the Quoridor backend system. It specifies REST endpoints, WebSocket protocol, core data structures, and communication rules. This document operates within the structural boundaries defined by the Backend System Design document.

---

# 1. Overview

The backend exposes two communication protocols:

- **REST** — Configuration and control (room management, game lifecycle)
- **WebSocket** — Runtime interaction (action submission, state broadcast)

All APIs use JSON encoding. Seat (1 or 2) is the only rule-level player identity.

---

# 2. REST API

Base path: `/api`

All REST endpoints return JSON. Error responses use the standard error envelope defined in Section 5.3.

---

## 2.1 Room Domain

### POST /api/rooms

Create a new room.

**Request body:** None required.

**Response:**

```json
{
  "room_id": "string (UUID)",
  "status": "config",
  "seats": {
    "1": { "client_id": null, "actor_type": null },
    "2": { "client_id": null, "actor_type": null }
  }
}
```

---

### POST /api/rooms/{room_id}/join

Bind a client to a seat.

**Request body:**

```json
{
  "client_id": "string",
  "seat": 1 | 2
}
```

**Response:** Room snapshot (same shape as create response).

**Errors:**
- 404 if room not found
- 400 if room not in `config` status
- 400 if seat already taken by a different client
- 400 if client already bound to another seat

---

### POST /api/rooms/{room_id}/select_actor

Set actor type for a seat.

**Request body:**

```json
{
  "seat": 1 | 2,
  "actor_type": "human" | "agent"
}
```

**Response:** Room snapshot.

**Errors:**
- 404 if room not found
- 400 if room not in `config` status
- 400 if invalid seat or actor_type

---

### POST /api/rooms/{room_id}/start_game

Start a game in the room. Transitions room status from `config` to `using`.

**Preconditions:**
- Room status must be `config`
- Both seats must have actor_type set
- Human seats must have client_id bound

**Request body:** None required.

**Response:**

```json
{
  "room_id": "string",
  "status": "using",
  "game": {
    "game_id": "string (UUID)",
    "phase": "running",
    "state": { GameState }
  }
}
```

**Errors:**
- 404 if room not found
- 400 if preconditions not met

---

### POST /api/rooms/{room_id}/close

Close the room. Terminal operation.

**Preconditions:**
- Room status must be `config`

**Request body:** None required.

**Response:**

```json
{
  "room_id": "string",
  "status": "closed"
}
```

**Errors:**
- 404 if room not found
- 400 if room status is `using` (active game must finish first)

---

### GET /api/rooms

List all rooms.

**Response:**

```json
{
  "rooms": [
    {
      "room_id": "string",
      "status": "config" | "using" | "closed",
      "seats": {
        "1": { "client_id": "string | null", "actor_type": "string | null" },
        "2": { "client_id": "string | null", "actor_type": "string | null" }
      }
    }
  ]
}
```

---

## 2.2 Game Domain

### GET /api/rooms/{room_id}/game/state

Get current game state.

**Response:**

```json
{
  "game_id": "string",
  "phase": "running" | "finished",
  "state": { GameState }
}
```

**Errors:**
- 404 if room not found
- 400 if no active game

---

### POST /api/rooms/{room_id}/game/new

Start a new game in the room (after a previous game finished). Transitions room back to `using`.

**Preconditions:**
- Room status must be `config` (previous game has finished, room reverted to config)

**Request body:** None required.

**Response:** Same as start_game.

---

### POST /api/rooms/{room_id}/swap_seats

Swap seat assignments.

**Preconditions:**
- Room status must be `config`

**Request body:** None required.

**Response:** Room snapshot with swapped seat bindings.

---

### POST /api/rooms/{room_id}/game/force_end

Force-end the current game.

**Preconditions:**
- Room status must be `using`

**Request body:** None required.

**Response:**

```json
{
  "room_id": "string",
  "status": "config",
  "result": {
    "winner_seat": null,
    "termination": "forced"
  }
}
```

---

### GET /api/rooms/{room_id}/game/replay

Retrieve replay data for the last completed game.

**Response:**

```json
{
  "game_id": "string",
  "actions": [
    { Action }
  ],
  "result": { GameResult }
}
```

**Errors:**
- 404 if room not found
- 400 if no completed game

---

# 3. WebSocket Protocol

## 3.1 Connection

**Endpoint:** `ws://{host}/ws/{room_id}`

Connection establishes a real-time channel for a specific room. All messages use the event envelope format defined in Section 5.4.

---

## 3.2 Client → Server Events

### subscribe

Register to receive room events.

```json
{
  "type": "subscribe",
  "client_id": "string"
}
```

Server responds with `room_snapshot`.

---

### take_action

Submit a game action.

```json
{
  "type": "take_action",
  "action": {
    "player": 1 | 2,
    "type": "pawn" | "horizontal" | "vertical",
    "target": [row, col]
  }
}
```

Server responds with `action_result` to submitter and broadcasts `state_update` to all subscribers on success.

---

### validate_action

Validate an action without executing it.

```json
{
  "type": "validate_action",
  "action": {
    "player": 1 | 2,
    "type": "pawn" | "horizontal" | "vertical",
    "target": [row, col]
  }
}
```

Server responds with `validate_result` to submitter only.

---

### surrender

Surrender the game.

```json
{
  "type": "surrender",
  "seat": 1 | 2
}
```

Server broadcasts `game_ended` to all subscribers.

---

## 3.3 Server → Client Events

### room_snapshot

Full room state, sent in response to `subscribe`.

```json
{
  "type": "room_snapshot",
  "room_id": "string",
  "status": "config" | "using",
  "seats": {
    "1": { "client_id": "string | null", "actor_type": "string | null" },
    "2": { "client_id": "string | null", "actor_type": "string | null" }
  },
  "game": {
    "game_id": "string | null",
    "phase": "string | null",
    "state": { GameState } | null
  }
}
```

---

### state_update

Authoritative game state progression. Broadcast to all subscribers after a successful action.

```json
{
  "type": "state_update",
  "game_id": "string",
  "state": { GameState },
  "last_action": { Action },
  "step_count": "integer"
}
```

This is the **only** authoritative signal for game progression. Clients must not derive state from any other event.

---

### action_result

Immediate feedback to the action submitter.

```json
{
  "type": "action_result",
  "success": true | false,
  "error": "string | null"
}
```

---

### validate_result

Response to a validate_action request.

```json
{
  "type": "validate_result",
  "valid": true | false,
  "reason": "string | null"
}
```

---

### game_started

Notification that a game has begun. Broadcast to all subscribers.

```json
{
  "type": "game_started",
  "game_id": "string",
  "state": { GameState }
}
```

---

### game_ended

Notification that a game has ended. Broadcast to all subscribers.

```json
{
  "type": "game_ended",
  "game_id": "string",
  "result": {
    "winner_seat": 1 | 2 | null,
    "termination": "goal" | "surrender" | "forced"
  }
}
```

---

### error

Error notification.

```json
{
  "type": "error",
  "code": "string",
  "message": "string"
}
```

Error codes:

| Code | Meaning |
|------|---------|
| INVALID_PAYLOAD | Malformed or missing fields |
| NOT_YOUR_TURN | Action submitted for wrong seat |
| INVALID_ACTION | Engine rejected the action |
| NO_ACTIVE_GAME | No game currently running |
| ROOM_NOT_FOUND | Room does not exist |
| UNBOUND_CLIENT | Client not subscribed to this room |

---

# 4. Core Data Structures

## 4.1 Action

The canonical action format used across all backend APIs.

```json
{
  "player": 1 | 2,
  "type": "pawn" | "horizontal" | "vertical",
  "target": [row, col]
}
```

- `player` — Seat number. The only rule-level identity.
- `type` — Action kind. `"pawn"` for movement, `"horizontal"` or `"vertical"` for wall placement.
- `target` — Logical coordinates. For pawn moves: `[row, col]` where row, col ∈ [0, 8]. For walls: `[row, col]` where row, col ∈ [0, 7].

---

## 4.2 GameState

The serialized game state returned by the backend. Derived from engine state.

```json
{
  "current_player": 1 | 2,
  "pawns": {
    "1": { "row": 0-8, "col": 0-8 },
    "2": { "row": 0-8, "col": 0-8 }
  },
  "walls_remaining": {
    "1": 0-10,
    "2": 0-10
  },
  "game_over": true | false,
  "winner": 1 | 2 | null
}
```

The backend serializes engine state into this format. The backend does not define or extend game state semantics; the engine is authoritative. Additional fields (e.g., wall positions) may be added when the engine exposes them.

---

## 4.3 Seat Model

```json
{
  "client_id": "string | null",
  "actor_type": "human" | "agent" | null
}
```

- `client_id` — External client identifier. Required for human seats.
- `actor_type` — Actor type bound to this seat. Must be set before game start.

---

## 4.4 GameResult

```json
{
  "winner_seat": 1 | 2 | null,
  "termination": "goal" | "surrender" | "forced"
}
```

---

## 4.5 Event Envelope

All WebSocket messages use a flat JSON object with a `type` field as discriminator.

```json
{
  "type": "string",
  ...event-specific fields
}
```

---

# 5. Rules and Constraints

## 5.1 Seat Identity

- Seat (1 or 2) is the only rule-level player identity.
- All actions reference seats, not client IDs.
- Client IDs exist only for transport-level binding.

## 5.2 Authority Model

- The engine is the sole rule authority.
- The backend does not enforce game rules.
- The backend routes actions to the engine and broadcasts results.
- `state_update` is the only authoritative event for game progression.

## 5.3 Error Envelope (REST)

All REST error responses use:

```json
{
  "detail": "string"
}
```

HTTP status codes:
- 400 — Client error (bad request, invalid state transition)
- 404 — Resource not found
- 409 — Conflict (operation not valid in current state)

## 5.4 Lifecycle Constraints

- Configuration operations (join, select_actor, swap_seats, close) are only valid when room status is `config`.
- Game operations (take_action, surrender, force_end) are only valid when room status is `using`.
- Room transitions to `config` automatically when a game finishes.

---

# Changelog

Version 1 (2026-04-03)
- Initial backend interface document aligned with backend architecture design
- REST API: Room domain (create, join, select_actor, start_game, close, list) and Game domain (state, new_game, swap_seats, force_end, replay)
- WebSocket protocol: subscribe, take_action, validate_action, surrender; state_update, action_result, validate_result, game_started, game_ended, error
- Core data structures: Action, GameState, Seat, GameResult, Event Envelope
- Rules: seat identity, authority model, error envelope, lifecycle constraints
