# Quoridor Backend System Design

Author: Ji Hua  
Created Date: 2026-04-03  
Last Modified: 2026-04-05  
Current Version: 2  
Document Type: Design  
Document Subtype: Backend System Design  
Document Status: In Development  
Document Authority Scope: Backend module  
Document Purpose:  
This document defines the structural design of the Quoridor backend system under the new architecture. It specifies system decomposition, communication model, lifecycle structure, and user interaction flow. It establishes clear responsibility boundaries between backend orchestration and engine rule authority, without defining API schemas or implementation details.

---

# 1. Overview

The backend system serves as the **orchestration layer** of the Live Game System.

It coordinates:

- Human clients (via Frontend WebSocket)
- Agent Service (external standalone service)
- The Engine module (authoritative rule system)

The backend does not define game rules. All rule validation and state transitions are delegated to the Engine.

The backend interacts with external systems through two communication planes:

- **Control Plane** — Configuration, lifecycle management, and agent orchestration
- **Gameplay Plane** — Action submission and state broadcast during active games

---

# 2. Design Goals

The backend is designed to:

- Maintain a single authoritative game state via Engine
- Support real-time gameplay for human users
- Provide a clean separation between configuration and runtime
- Enable both human and agent participation through a unified gameplay contract
- Support sequential games within a single room
- Ensure deterministic and reproducible game progression
- Separate control operations (room setup, agent management) from gameplay operations (action submission, state broadcast)
- Treat Agent Service as an external module with clear communication boundaries

---

# 3. System Architecture

The backend is structured into five layers:

## 3.1 Transport Layer

Responsible for external communication.

- REST Controller (Frontend Control + Agent Control)
- WebSocket Gateway (Gameplay + Frontend Control events)
- Agent Service Client (Backend → Agent Service communication)

## 3.2 Application Layer

Responsible for orchestration logic.

- Room Manager
- Game Manager
- Seat Binding Manager
- Agent Lifecycle Manager

## 3.3 Runtime Layer

Responsible for live execution.

- Turn Orchestrator
- Agent Proxy (routes gameplay to/from Agent Service)
- Broadcast Hub

## 3.4 Domain Adapter Layer

Responsible for interfacing with other modules.

- Engine Adapter
- Display Coordinate Adapter
- Agent Service Adapter

## 3.5 Storage Layer

Responsible for in-memory state and persistence (optional).

- Room Registry
- Game Registry
- Replay Storage

---

# 4. Communication Model

The backend uses a dual-protocol, dual-plane communication model.

## 4.1 Communication Planes

### Control Plane

The Control Plane handles all configuration, lifecycle management, and agent orchestration.

Participants:
- Frontend → Backend (room setup, seat binding, game lifecycle, surrender)
- Frontend → Backend → Agent Service (agent creation, configuration, start/stop)

Protocol: REST (primary), WebSocket (surrender only)

### Gameplay Plane

The Gameplay Plane handles action submission and state broadcast during active games.

Participants:
- Frontend ↔ Backend (human action submission, state broadcast)
- Agent Service ↔ Backend (agent action submission, state delivery)
- Backend → Engine (action validation and state transition)

Protocol: WebSocket (primary), REST (state query)

**Key constraint:** The Gameplay Plane uses a single shared contract for both Frontend and Agent Service. The Backend does not distinguish between human and agent actions at this layer.

## 4.2 REST

REST is used for:

- Room creation and management (Control Plane)
- Seat binding and actor configuration (Control Plane)
- Game start and lifecycle control (Control Plane)
- Agent lifecycle management (Control Plane)
- Game state query (Gameplay Plane)
- Post-game operations (Control Plane)

REST interactions are low-frequency and request-response based.

## 4.3 WebSocket

WebSocket is used for:

- Real-time action submission — human players and agent proxy (Gameplay Plane)
- Game state broadcasting (Gameplay Plane)
- Event notifications — game start, game end, errors (Gameplay Plane)
- Surrender (Control Plane — Frontend only)

WebSocket is the authoritative runtime channel for gameplay.

## 4.4 Backend ↔ Agent Service Communication

The Backend communicates with the Agent Service as an external module.

Control Plane:
- Backend sends agent lifecycle commands (create, start, stop) to Agent Service

Gameplay Plane:
- Backend sends GameState to Agent Service when it is an agent's turn
- Agent Service responds with an Action
- Backend routes the Action through Engine like any other player action

The concrete protocol (REST, WebSocket, or internal RPC) between Backend and Agent Service is not yet defined. The architectural constraint is that all agent actions must flow through the Backend.

---

# 5. Protocol Structure

## 5.1 REST Domain Separation

REST APIs are logically divided into:

### Room Domain

- Create room
- Join room
- Bind seat
- Select actor type
- Start game
- Close room

### Game Domain

- Query current game state
- Start new game
- Swap seats
- Force end game
- Retrieve replay

## 5.2 WebSocket Event Model

WebSocket uses an event-based envelope:

- Client → Server:
  - subscribe
  - take_action
  - validate_action
  - surrender

- Server → Client:
  - room_snapshot
  - state_update
  - action_result
  - validate_result
  - game_started
  - game_ended
  - error

## 5.3 Core Principle

- Seat (1|2) is the only rule-level identity
- All actions are routed through backend → engine → broadcast
- state_update is the only authoritative progress signal

---

# 6. Lifecycle Model

The system uses a dual-layer lifecycle:

## 6.1 Room Lifecycle

Room.status ∈ { config, using, closed }

- config:
  - Room is configurable
  - Seats and actor types can be modified
  - Game can be started

- using:
  - A game is active
  - Configuration is locked

- closed:
  - Terminal state

Transitions:

- config → using: start_game
- using → config: game finished (automatic)
- config → closed: close_room

---

## 6.2 Game Lifecycle

Game.phase ∈ { starting, running, ending, finished }

- starting:
  - Engine initialization
  - Agent preparation

- running:
  - Normal gameplay loop
  - Accepts actions

- ending:
  - Terminal condition reached
  - No new actions accepted

- finished:
  - Result finalized
  - Replay generated

Transitions:

- starting → running: initialization complete
- running → ending: goal / surrender / forced end
- ending → finished: cleanup complete

---

## 6.3 Lifecycle Relationship

- Room.status = using while Game.phase ∈ {starting, running, ending}
- Game.phase = finished triggers Room.status → config

---

# 7. User Interaction Model

## 7.1 Human vs Human

1. User creates room
2. Both users join and bind to seats
3. Actor types set to human
4. Game is started
5. Clients connect via WebSocket
6. Players submit actions via WebSocket
7. Backend routes actions to Engine
8. Backend broadcasts state_update
9. Game ends via goal or surrender
10. Users may start a new game or swap seats

---

## 7.2 Human vs Agent

1. Human binds to one seat
2. Other seat configured as agent (via Agent Control API)
3. Backend requests agent creation from Agent Service
4. Game starts
5. When agent's turn: Backend sends GameState to Agent Service, receives Action
6. Backend routes agent Action through Engine (same path as human)
7. Human interacts via WebSocket
8. On game end: Backend instructs Agent Service to stop agent

## 7.3 Agent vs Agent

1. Both seats configured as agent (via Agent Control API)
2. Backend requests agent creation from Agent Service for both seats
3. Game starts
4. Backend alternates sending GameState to Agent Service for each seat's agent
5. Each agent responds with an Action
6. Backend routes actions through Engine
7. Frontend (if connected) receives state_update broadcasts as observer
8. On game end: Backend instructs Agent Service to stop both agents

---

## 7.4 Local Two-Player Mode

- Both players operate from the same frontend instance
- Backend still distinguishes seats (1 and 2)
- All actions are submitted through backend
- Backend maintains authoritative state

---

# 8. Authority Model

- Engine is the sole rule authority
- Backend is orchestration authority
- Frontend is interaction layer only

Constraints:

- Backend must not implement rule logic
- Backend must not mutate GameState directly
- All state transitions must occur via Engine

---

# 9. Concurrency Guarantees

The backend must ensure:

- Per-room mutual exclusion during action execution
- Illegal actions do not mutate state
- Exactly one state_update per successful action
- No actions accepted outside valid lifecycle phases

---

# 10. Summary

The backend system:

- Uses REST for configuration and WebSocket for runtime
- Separates Room lifecycle from Game execution lifecycle
- Maintains a single authoritative Engine instance per game
- Treats seat as the only rule-level identity
- Supports both human and agent participation
- Enables sequential games within a single room

This design prioritizes:

- Simplicity for initial implementation
- Clear separation of responsibilities
- Compatibility with future extensions

---

# Changelog

Version 2 (2026-04-05)
- Added Agent Service as external module in system architecture
- Introduced Control Plane vs Gameplay Plane separation
- Added Agent Lifecycle Manager to Application Layer
- Added Agent Proxy and Agent Service Adapter to Runtime and Adapter layers
- Added Backend ↔ Agent Service communication model (Section 4.4)
- Added Agent vs Agent interaction model (Section 7.3)
- Updated Human vs Agent flow to use Agent Service
- Renumbered Local Two-Player Mode to Section 7.4

Version 1 (2026-04-03)
- Initial backend system design aligned with new architecture
