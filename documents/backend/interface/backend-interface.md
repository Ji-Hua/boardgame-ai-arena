# Quoridor Backend Interface

Author: Ji Hua  
Created Date: 2026-04-03  
Last Modified: 2026-04-07  
Current Version: 5  
Document Type: Interface  
Document Subtype: Backend API Contract  
Document Status: In Development  
Document Authority Scope: Backend module  
Document Purpose:  
This document defines the external API contract of the Quoridor backend system. It specifies REST endpoints, WebSocket protocol, core data structures, and communication rules. This document operates within the structural boundaries defined by the Backend System Design document.

The backend API is organized into three distinct planes:

- **Frontend Control API** — Room setup, seat management, game lifecycle control, and player-initiated control actions (e.g., surrender). Used by the Frontend only.
- **Agent Control API** — Agent lifecycle management: creation, configuration, seat binding, and start/stop. Used by the Frontend (or administrative clients) to manage agents via the Backend.
- **Gameplay API (Shared)** — Action submission, state broadcast, and validation. Shared by both Frontend (human players) and Agent Service. This plane must not be duplicated; a single contract serves both consumers.

---

# 1. Overview

The backend exposes two communication protocols:

- **REST** — Configuration, control, and agent management
- **WebSocket** — Runtime interaction (action submission, state broadcast)

All APIs use JSON encoding. Seat (1 or 2) is the only rule-level player identity.

## 1.1 API Plane Summary

| Plane | Consumer | Protocol | Purpose |
|-------|----------|----------|---------|
| Frontend Control | Frontend | REST + WebSocket | Room management, seat binding, game lifecycle, surrender |
| Agent Control | Frontend / Admin | REST | Agent creation, configuration, binding, start/stop |
| Gameplay (Shared) | Frontend + Agent Service | WebSocket (+ REST query) | Action submission, state broadcast, validation |

**Key constraint:** The Gameplay API is a single shared contract. Both human players (via Frontend) and automated players (via Agent Service) use the same action submission and state update mechanisms. The Backend does not distinguish between human and agent at the gameplay layer.

---

# 2. Frontend Control API (REST)

Base path: `/api`

All REST endpoints return JSON. Error responses use the standard error envelope defined in Section 6.3.

This section defines APIs used exclusively by the Frontend for room management, seat configuration, and game lifecycle control.

---

## 2.1 Room Management

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
- Replay constraint: if any seat has a replay agent, both seats must have replay agents

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

## 2.2 Game Lifecycle Control

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

### POST /api/rooms/{room_id}/game/speed

Update the game speed multiplier for the active game. Takes effect immediately for the next inter-step delay.

**Preconditions:**
- Room status must be `using` (active game)

**Request body:**

```json
{
  "speed_multiplier": 0.5 | 1 | 2 | 4 | 8
}
```

- `speed_multiplier` — Playback speed relative to the base interval. The base step interval is 0.5 s (at 1×). Allowed values: `0.5`, `1`, `2`, `4`, `8`.
- Actual inter-step delay = `0.5 / speed_multiplier` seconds.

**Response:**

```json
{
  "room_id": "string",
  "speed_multiplier": 0.5 | 1 | 2 | 4 | 8
}
```

**Behavior:**
- Speed control applies **only** when both seats are agent-controlled (agent vs agent, including replay mode).
- Has no effect on human-controlled turns.
- The backend is the **sole authority** for inter-step timing. The engine and agent modules are never delayed.

**Errors:**
- 404 if room not found
- 400 if no active game
- 400 if `speed_multiplier` is not one of the allowed values

---

## 2.3 Surrender (Frontend Control — WebSocket)

Surrender is a **Frontend Control** action, not a Gameplay action. It terminates the game without submitting a game action through the Engine.

### surrender

Surrender the game. Sent over WebSocket.

```json
{
  "type": "surrender",
  "seat": 1 | 2
}
```

Server broadcasts `game_ended` to all subscribers.

**Note:** Only human players (via Frontend) may surrender. Agent Service does not initiate surrender; the Backend controls agent lifecycle independently.

---

# 3. Agent Control API (REST)

Base path: `/api`

This section defines APIs for managing agent lifecycle. These endpoints are called by the Frontend (or administrative clients) to instruct the Backend to create, configure, bind, and control agents via the Agent Service.

**Note:** Concrete request/response schemas are not yet defined. This section establishes the logical operations and their purpose. Detailed schemas will be added in a future revision.

---

## 3.1 Operations

### POST /api/rooms/{room_id}/agent/create

Create an agent instance for a room.

**Purpose:** Instructs the Backend to request agent creation from the Agent Service.

**Parameters (conceptual):**
- `seat` — Target seat (1 or 2)
- `agent_type` — Agent type identifier (e.g., `"random"`, `"greedy"`, `"replay"`)
- `config` — Optional agent-specific configuration (e.g., replay data for replay agents)

---

### POST /api/rooms/{room_id}/agent/start

Start an agent for the current game.

**Purpose:** Instructs the Backend to activate the agent bound to a seat. The agent begins receiving GameState and producing actions.

**Preconditions:**
- Room status must be `using` (game active)
- Agent must be created and bound to a seat

---

### POST /api/rooms/{room_id}/agent/stop

Stop an active agent.

**Purpose:** Instructs the Backend to deactivate the agent. The agent stops receiving GameState and producing actions.

---

### GET /api/agent/types

List available agent types.

**Purpose:** Returns the set of agent types registered in the Agent Service.

**Response (conceptual):**

```json
{
  "agent_types": [
    {
      "type_id": "string",
      "display_name": "string",
      "category": "ai" | "replay" | "scripted"
    }
  ]
}
```

---

# 4. Gameplay API (Shared)

This section defines the API used by **both** the Frontend (human players) and the Agent Service (automated players) for gameplay interaction. This is a single shared contract — the Backend does not distinguish between human and agent at this layer.

---

## 4.1 Game State Query (REST)

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

## 4.2 WebSocket Connection

**Endpoint:** `ws://{host}/ws/{room_id}`

Connection establishes a real-time channel for a specific room. All messages use the event envelope format defined in Section 6.5.

---

## 4.3 Client → Server Events (Gameplay)

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

Submit a game action. Used by both Frontend (human) and Agent Service (automated).

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

### get_legal_actions

Request all legal pawn moves for the current player.

```json
{
  "type": "get_legal_actions"
}
```

Server responds with `legal_actions_result` to the requesting client only. The returned actions represent only pawn moves for the current player. Wall placement legality is not included.

**Errors:**
- If no active game exists, returns `legal_actions_result` with an empty `actions` array.

---

## 4.4 Server → Client Events (Gameplay)

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

### legal_actions_result

Response to a `get_legal_actions` request. Sent only to the requesting client.

```json
{
  "type": "legal_actions_result",
  "actions": [
    {
      "player": 1 | 2,
      "type": "pawn",
      "target": [row, col]
    }
  ]
}
```

- `actions` — Array of legal pawn moves for the current player. Empty array if no game is active or no legal moves are available.
- Each action follows the standard Action schema (Section 5.1).
- Only pawn moves are returned. Wall placement legality is determined only after a `take_action` attempt.

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

# 5. Core Data Structures

## 5.1 Action

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

## 5.2 GameState

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

## 5.3 Seat Model

```json
{
  "client_id": "string | null",
  "actor_type": "human" | "agent" | null
}
```

- `client_id` — External client identifier. Required for human seats.
- `actor_type` — Actor type bound to this seat. Must be set before game start.

---

## 5.4 GameResult

```json
{
  "winner_seat": 1 | 2 | null,
  "termination": "goal" | "surrender" | "forced"
}
```

---

## 5.5 Event Envelope

All WebSocket messages use a flat JSON object with a `type` field as discriminator.

```json
{
  "type": "string",
  ...event-specific fields
}
```

---

# 6. Rules and Constraints

## 6.1 Seat Identity

- Seat (1 or 2) is the only rule-level player identity.
- All actions reference seats, not client IDs.
- Client IDs exist only for transport-level binding.

## 6.2 Authority Model

- The engine is the sole rule authority.
- The backend does not enforce game rules.
- The backend routes actions to the engine and broadcasts results.
- `state_update` is the only authoritative event for game progression.

## 6.3 Error Envelope (REST)

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

## 6.5 Timing Authority

- The backend is the sole authority for inter-step timing in agent-driven gameplay.
- A configurable `speed_multiplier` determines the delay between consecutive agent actions.
- Base step interval: 0.5 seconds at 1× speed. Actual delay = `0.5 / speed_multiplier`.
- Speed control applies **only** in agent vs agent (or replay) mode — when both seats are agent-controlled.
- Human turns are always immediately responsive; no delay is introduced for human players.
- The engine is a pure rule evaluator and is never delayed.
- The Agent Service is a stateless decision provider and is never delayed.
- Speed may be updated mid-game via `POST /api/rooms/{room_id}/game/speed`; the new value applies to the next step.

---

## 6.4 Lifecycle Constraints

- Configuration operations (join, select_actor, swap_seats, close) are only valid when room status is `config`.
- Game operations (take_action, surrender, force_end) are only valid when room status is `using`.
- Room transitions to `config` automatically when a game finishes.

## 6.5 Local Mode (Single-Connection Two-Player)

Local Mode allows a single WebSocket connection to control both seats.

- One client subscribes to a room with a single `client_id`.
- The client submits `take_action` for either seat by specifying `player: 1` or `player: 2` in the action.
- The backend does not enforce that a connection may only act for one seat. Seat identity is determined by the `player` field in the action, not by the connection.
- Turn order is enforced by the engine: only the action for the current player's seat will be accepted.
- All events (`action_result`, `state_update`, `game_ended`) are delivered to the single connection.
- This mode requires no special API or configuration. It is the default behavior when one connection sends actions for both seats.

---

# Changelog

Version 5 (2026-04-07)
- Added `POST /api/rooms/{room_id}/game/speed` endpoint (Section 2.2)
- Added Section 6.5 Timing Authority: backend is sole authority for inter-step delay
- Speed control applies only in agent vs agent / replay mode; human turns unaffected
- Allowed speed_multiplier values: 0.5, 1, 2, 4, 8; base interval 0.5 s at 1×

Version 4 (2026-04-05)
- Reorganized API into three planes: Frontend Control API (Section 2), Agent Control API (Section 3), Gameplay API (Section 4)
- Moved surrender from Gameplay to Frontend Control (Section 2.3)
- Added Agent Control API with conceptual operations (create, start, stop, list types)
- Moved game state query into Gameplay API (Section 4.1)
- Renumbered Core Data Structures to Section 5 and Rules to Section 6
- Updated overview to reflect three-plane API model

Version 3 (2026-04-05)
- Added `get_legal_actions` client → server message
- Added `legal_actions_result` server → client event

Version 2 (2026-04-04)
- Added Local Mode (single-connection two-player)

Version 1 (2026-04-03)
- Initial backend interface document
