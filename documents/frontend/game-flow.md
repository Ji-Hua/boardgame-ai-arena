# Quoridor Frontend Game Flow (Local Mode)

Author: Ji Hua  
Created Date: 2026-04-04  
Last Modified: 2026-04-05  
Current Version: 2  
Document Type: Design  
Document Subtype: Game Flow  
Document Status: In Development  
Document Authority Scope: Frontend module  
Document Purpose:  
This document describes the full lifecycle of a game from the frontend perspective, including user flow, frontend ↔ backend interaction, and all communication protocols. It is the mandatory companion to frontend-architecture.md and governs implementation of the protocol layer.

---

# 1. Initialization Flow

## 1.1 Overview

The frontend operates in local mode. A single client controls both seats through one WebSocket connection. The initialization sequence is fully automated — no manual room creation or player selection UI is exposed to the user.

## 1.2 Bootstrap Sequence

LiveController executes the following REST calls in order on application start:

1. **Create room**
   - `POST /api/rooms`
   - Response: `{ room_id, status: "config", seats: { "1": { client_id: null, actor_type: null }, "2": { ... } } }`
   - LiveController stores `room_id`.

2. **Generate client ID**
   - LiveController generates a UUID `client_id` locally.
   - This ID is used for both seats and for the WebSocket subscription.

3. **Join seat 1**
   - `POST /api/rooms/{room_id}/join`
   - Body: `{ "client_id": "<client_id>", "seat": 1 }`
   - Response: Room snapshot.

4. **Join seat 2**
   - `POST /api/rooms/{room_id}/join`
   - Body: `{ "client_id": "<client_id>", "seat": 2 }`
   - Response: Room snapshot.

5. **Select actor for seat 1**
   - `POST /api/rooms/{room_id}/select_actor`
   - Body: `{ "seat": 1, "actor_type": "human" }`
   - Response: Room snapshot.

6. **Select actor for seat 2**
   - `POST /api/rooms/{room_id}/select_actor`
   - Body: `{ "seat": 2, "actor_type": "human" }`
   - Response: Room snapshot.

7. **Start game**
   - `POST /api/rooms/{room_id}/start_game`
   - Response: `{ room_id, status: "using", game: { game_id, phase: "running", state: { GameState } } }`
   - LiveController stores `game_id`.

## 1.3 Post-Bootstrap

After successful bootstrap:
- LiveController establishes WebSocket connection.
- The initial GameState from `start_game` response may be used to render the board immediately, but `state_update` remains the only authoritative runtime signal.

---

# 2. WebSocket Flow

## 2.1 Connection

- Endpoint: `ws://{host}/ws/{room_id}`
- LiveController opens a WebSocket connection to the room after bootstrap completes.

## 2.2 Subscribe

After connection is established, LiveController sends:

```json
{
  "type": "subscribe",
  "client_id": "<client_id>"
}
```

Server responds with `room_snapshot`:

```json
{
  "type": "room_snapshot",
  "room_id": "<room_id>",
  "status": "using",
  "seats": { ... },
  "game": {
    "game_id": "<game_id>",
    "phase": "running",
    "state": { GameState }
  }
}
```

## 2.3 Receiving state_update

On every successful action, the backend broadcasts:

```json
{
  "type": "state_update",
  "game_id": "<game_id>",
  "state": { GameState },
  "last_action": { Action },
  "step_count": <integer>
}
```

Frontend processing:
1. LiveController receives `state_update`.
2. StateMapper converts `GameState` → `RenderState`.
3. `renderStore.setState(newRenderState)` replaces current state entirely.
4. UI components re-render from the new RenderState.

This is the **only** authoritative signal for game progression. The frontend must not derive state from any other event.

## 2.4 GameState Schema

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

All coordinates are 0-based. No notation conversion is required.

---

# 3. Action Flow

## 3.1 User Interaction

The user clicks on the board to:
- Click the current player's pawn to reveal legal moves (highlight mode).
- Click a highlighted cell to move a pawn.
- Click elsewhere (non-highlighted cell) to attempt a pawn move directly (backend validates).
- Place a wall at a position.

The user controls both seats. The frontend determines which seat is acting based on `currentSeat` from RenderState.

## 3.2 Legal Move Highlight Flow

When the user clicks the current player's pawn:

1. InteractionLayer detects the click is on the current player's pawn position.
2. LiveController sends `get_legal_actions` via WebSocket:
   ```json
   { "type": "get_legal_actions" }
   ```
3. Backend calls `legal_pawn_actions()` on the engine and responds with `legal_actions_result`:
   ```json
   {
     "type": "legal_actions_result",
     "actions": [
       { "player": 1, "type": "pawn", "target": [row, col] }
     ]
   }
   ```
4. LiveController maps each `target: [engine_x, engine_y]` → `[engine_y, engine_x]` (RenderState convention).
5. `legalMoves` state in GamePage is updated.
6. HighlightLayer renders colored circles at each legal position.

Highlight clearing rules:
- Legal moves are cleared on any `state_update` from backend.
- Legal moves are cleared when a move action is submitted.
- Legal moves are cleared when entering wall placement mode.
- Clicking the pawn again while highlights are showing clears them (toggle).

## 3.3 Action Construction

1. **InteractionLayer** captures the user click (row, col, action type).
2. **ActionTranslator** converts the UI event into the backend Action format:

```json
{
  "player": 1 | 2,
  "type": "pawn" | "horizontal" | "vertical",
  "target": [row, col]
}
```

- `player` is set to `renderState.currentSeat`.
- `type` is `"pawn"` for movement, `"horizontal"` or `"vertical"` for wall placement.
- `target` is `[row, col]` in 0-based coordinates.

## 3.4 Sending take_action

LiveController sends via WebSocket:

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

## 3.4 Receiving action_result

The backend responds to the submitter with:

```json
{
  "type": "action_result",
  "success": true | false,
  "error": "string | null"
}
```

- If `success: true` — Frontend waits for `state_update` (the only authority).
- If `success: false` — Frontend may display an error toast. No state change occurs.

`action_result` is acknowledgment only. The frontend must NOT update game state based on it.

## 3.6 State Update After Action

On success, the backend broadcasts `state_update` to all subscribers (Section 2.3). The frontend replaces its entire RenderState.

---

# 4. Game End Flow

## 4.1 game_ended Event

When the game terminates (win by goal, surrender, or forced end), the backend broadcasts:

```json
{
  "type": "game_ended",
  "game_id": "<game_id>",
  "result": {
    "winner_seat": 1 | 2 | null,
    "termination": "goal" | "surrender" | "forced"
  }
}
```

## 4.2 Frontend Processing

1. LiveController receives `game_ended`.
2. RenderState is updated:
   - `isTerminal = true`
   - `result = { winnerSeat, termination }`
3. `renderStore.setState(updatedRenderState)` triggers UI update.
4. UI displays the victory modal or game-over state.

## 4.3 Post-Game

After game end:
- The board remains displayed in its final state.
- The user may close the page or refresh to start a new game.
- No new-game or rematch flow is required in the initial local mode implementation.

---

# 5. Error Handling

## 5.1 REST Errors

During bootstrap, if any REST call fails:
- LiveController logs the error.
- The UI may display an error state.
- The game does not start.

## 5.2 WebSocket Errors

The backend may send:

```json
{
  "type": "error",
  "code": "INVALID_ACTION" | "NOT_YOUR_TURN" | ...,
  "message": "string"
}
```

Frontend displays the error message as a toast notification. No state change occurs.

## 5.3 Connection Loss

If the WebSocket connection drops:
- LiveController may attempt reconnection.
- The UI may display a disconnection indicator.
- No state reconstruction is attempted — the frontend waits for the next `state_update` after reconnection.

---

# 6. Constraints

- This document follows the backend API defined in `documents/backend/interface/backend-interface.md`.
- This document follows the frontend architecture defined in `documents/frontend/frontend-architecture.md`.
- Local mode: single client controls both seats via one WebSocket connection.
- No rule logic in frontend.
- No state inference or reconstruction.
- `state_update` is the only authoritative game progression signal.
- All coordinates are 0-based (no Display Grid notation).

---

# Changelog

Version 2 (2026-04-05)
- Added Section 3.2: Legal Move Highlight Flow (get_legal_actions → legal_actions_result → HighlightLayer)
- Renumbered Sections 3.3–3.6 (previously 3.2–3.5)
- Clarified action_result: no UI change on rejected actions; wall optimistic rendering explicitly forbidden
- Aligned with backend-interface.md v3

Version 1 (2026-04-04)
- Initial game flow document for local mode frontend
- Defined initialization, WebSocket, action, and game end flows
- Aligned with backend-interface.md v2 and frontend-architecture.md v1
