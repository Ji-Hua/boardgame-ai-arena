# Quoridor Frontend Architecture (Local Mode)

Author: Ji Hua  
Created Date: 2026-04-04  
Last Modified: 2026-04-04  
Current Version: 1  
Document Type: Design  
Document Subtype: Frontend Architecture  
Document Status: In Development  
Document Authority Scope: Frontend module  
Document Purpose:  
This document defines the frontend architecture for the Quoridor system under the new backend interface. It preserves the render-driven architecture of the previous frontend while adapting the protocol layer to the updated backend API. The frontend operates in local mode, where a single client controls both seats through one WebSocket connection.

---

# 1. Overview

The frontend adopts a **Render-Driven, Backend-Authoritative Architecture**.

The system is designed to:

- Preserve the existing rendering architecture and UI components.
- Adapt to the new backend API and WebSocket protocol.
- Support local two-player interaction using a single connection.
- Maintain strict separation between rendering and backend communication.

The frontend does not implement any game rules, state transitions, or legality checks.

---

# 2. Core Principles

## 2.1 Single Authority Model

- WebSocket `state_update` is the only authoritative signal of game state progression.
- The frontend must not infer, merge, or reconcile state.
- Every state update replaces the current RenderState.

---

## 2.2 Render-Driven Architecture

- All UI rendering is derived from RenderState.
- RenderState is a projection of backend GameState and WebSocket events.
- UI components are stateless with respect to game logic.

---

## 2.3 Seat-Based Identity

- Seat (1 or 2) is the only rule-level identity.
- The frontend does not define or reinterpret player identity.
- UI labels such as Player 1 / Player 2 are purely presentational.

---

## 2.4 No Rule Logic in Frontend

The frontend must not:

- Validate actions
- Compute legal moves
- Infer turn order
- Determine game outcome

All rule logic belongs to the engine and backend.

---

## 2.5 Local Mode Operation

- A single WebSocket connection controls both seats.
- Actions include an explicit `player` field (1 or 2).
- The backend enforces turn order and legality.

---

# 3. High-Level Architecture

## 3.1 Data Flow

Backend → WebSocket state_update → StateMapper → RenderState → renderStore → UI

- Backend is the source of truth.
- StateMapper converts backend GameState into RenderState.
- renderStore holds the current RenderState.
- UI renders purely from renderStore.

---

## 3.2 Control Flow

UI → InteractionLayer → ActionTranslator → LiveController → Backend

- User interactions are captured by InteractionLayer.
- ActionTranslator converts UI intent into backend Action format.
- LiveController sends actions via WebSocket.

---

# 4. Module Structure

The frontend maintains the existing modular structure with selective adaptation.

## 4.1 Core

RenderState  
Defines the unified rendering schema.

StateMapper  
Maps backend GameState and events to RenderState.

ActionTranslator  
Converts UI input into backend Action format.

---

## 4.2 Controller

### LiveController

The only module responsible for backend communication.

Responsibilities:

- Perform initial game bootstrap via REST:
  - Create room
  - Join both seats with a single client_id
  - Set both seats to human
  - Start game
- Establish WebSocket connection
- Subscribe to room events
- Handle incoming events:
  - state_update
  - game_started
  - game_ended
- Forward user actions to backend via WebSocket

Constraints:

- No game logic
- No state inference
- No direct UI interaction

---

## 4.3 Stores

renderStore  
- Holds the current RenderState  
- Single source of truth for rendering  

uiStore  
- Stores ephemeral UI state (hover, selection)  
- Must not contain rule-level state  

No sessionStore or replayStore is required in local mode.

---

## 4.4 View Layer

Board.tsx  
- Main rendering component  
- Purely renders from RenderState  

Rendering Layers:

- PawnLayer  
- WallLayer  
- HighlightLayer  
- InteractionLayer  

Rules:

- No backend calls  
- No game logic  
- No mode branching  

---

## 4.5 Utilities

coord.ts  
- Handles coordinate mapping using 0-based grid  
- No notation conversion is required  

---

# 5. Backend Integration

## 5.1 REST (Bootstrap Only)

Used only during initialization:

- POST /api/rooms
- POST /api/rooms/{room_id}/join
- POST /api/rooms/{room_id}/select_actor
- POST /api/rooms/{room_id}/start_game

REST responses are not authoritative for game progression.

---

## 5.2 WebSocket (Runtime)

Endpoint:

ws://{host}/ws/{room_id}

Events:

- subscribe (client → server)
- state_update (server → client)
- action_result (server → client)
- game_started (server → client)
- game_ended (server → client)

Rules:

- state_update fully replaces frontend state
- action_result is acknowledgment only
- No state reconstruction from partial events

---

## 5.3 Action Format

All actions follow the backend contract:

- player: 1 or 2
- type: pawn | horizontal | vertical
- target: [row, col]

Frontend must construct actions without modifying semantics.

---

# 6. RenderState

RenderState is a projection of backend GameState and WebSocket metadata.

Key fields:

- boardSize
- pawns
- walls
- currentSeat
- stepCount
- lastAction
- isTerminal
- result

Constraints:

- Must originate from backend data
- Must not infer missing information
- Must not be mutated by UI

---

# 7. Removed Components

The following components are not part of the new architecture:

- ReplayController
- replayStore
- ReplayPage
- Session-based UI (room creation, player selection)
- Agent-related UI

These responsibilities are either removed or handled automatically in the backend bootstrap process.

---

# 8. Interaction Model

## Live Interaction

- User interacts with board
- InteractionLayer captures input
- ActionTranslator generates Action
- LiveController sends action via WebSocket
- Backend processes action and emits state_update
- UI updates via RenderState

---

# 9. Non-Negotiable Rules

- No rule logic in frontend
- WebSocket state_update is the only authority
- RenderState is the only rendering source
- LiveController is the only backend communication layer
- Seat is the only rule-level identity
- UI must not infer or reconstruct game state

---

# 10. Summary

This architecture preserves the original render-driven design while adapting the protocol layer to the new backend.

The frontend is:

- A pure rendering system
- A thin transport adapter to backend APIs
- Fully aligned with backend authority and engine semantics

All complexity related to game rules, state transitions, and validation remains outside the frontend.

---

# Changelog

Version 1 (2026-04-04)
- Initial definition of frontend architecture aligned with new backend API
- Introduced local mode operation with single connection controlling both seats
- Removed replay and session-based interaction model
- Preserved render-driven architecture and module boundaries
