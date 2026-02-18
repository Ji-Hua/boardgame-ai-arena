# Quoridor System Architecture

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 1

Document Type: Design
Document Subtype: System Architecture
Document Status: Draft
Document Authority Scope: Global
Document Purpose:
This document defines the high-level system architecture of the Quoridor project. It establishes the structural decomposition of the system into major subsystems, clarifies module responsibilities, and defines authority boundaries and inter-module relationships. This document does not define protocols, APIs, or implementation details.

---

# 1. System Overview

Quoridor is designed as a rule-centric AI platform built around a deterministic game engine. The system supports two primary operational domains:

1. Live Game System
2. Training System

The Live Game System enables real-time matches between human and AI players, including visualization and interaction.

The Training System enables self-play, evaluation, and experimentation. The Training System is acknowledged but not yet fully specified. Detailed architectural design of the Training System is outside the scope of this document and will be defined in a future revision.

This document focuses primarily on the Live Game System.

---

# 2. High-Level System Decomposition

The system is decomposed into two major subsystems:

## 2.1 Live Game System (Current Focus)

The Live Game System supports real-time, authoritative matches and visualization.

It is composed of four primary modules:

- Engine Module
- Backend Module
- Frontend Module
- Agent Module

## 2.2 Training System (Future Scope)

The Training System will support:

- Self-play simulation
- Model evaluation
- ELO computation
- Experimental analysis

The Training System is expected to reuse the Rule Layer of the Engine but does not depend on the Live Game orchestration layer.

The architectural design of the Training System is intentionally not defined in this version of the document.

---

# 3. Live Game System Design Goals

The Live Game System is designed with the following goals:

- Single authoritative live state.
- Clear separation of rule authority and orchestration.
- Strict module responsibility boundaries.
- Agent-human equivalence at the gameplay layer.
- Support for local simulation without compromising live authority.
- Deterministic and reproducible game progression.

---

# 4. Live Game System Structure

## 4.1 Engine Module

The Engine Module is the canonical rule authority of the system.

It is internally structured as two conceptual layers:

- Rule Layer (stateless)
- Game Layer (stateful)

### Responsibilities

- Maintain authoritative live GameState (Game Layer).
- Validate and apply Actions via the Rule Layer.
- Produce deterministic state transitions.
- Provide rule-derived utilities.

### Authority Model

- The Engine is the only module that defines valid state transitions.
- The Game Layer holds the authoritative live GameState.
- No external module may directly mutate GameState.
- All modifications must occur through Engine-defined interfaces.

### Non-Responsibilities

- Session management
- Networking
- UI rendering
- Agent lifecycle management
- Training orchestration

---

## 4.2 Backend Module

The Backend Module is the Live Game orchestration and coordination layer.

### Responsibilities

- Manage Room lifecycle.
- Manage Game lifecycle.
- Instantiate and manage the authoritative Engine instance per live game.
- Route player actions to the Engine.
- Receive updated GameState from the Engine.
- Broadcast state updates to connected clients.
- Manage player proxies (Human or Agent).

### Authority Model

- Backend is the orchestration authority.
- Backend does not define rules.
- Backend does not directly mutate GameState.
- Backend interacts with Engine exclusively through its public interfaces.

Backend may maintain read-only state views or cached representations, but such views are not authoritative.

### Non-Responsibilities

- Rule evaluation logic
- Agent implementation
- Training logic
- Local simulation logic

---

## 4.3 Frontend Module

The Frontend Module provides user interaction and visualization.

### Responsibilities

- Create and join Rooms.
- Select seats.
- Configure matches.
- Display GameState received from Backend.
- Submit player Actions to Backend.

### Non-Responsibilities

- Rule validation
- Authoritative state maintenance
- Direct interaction with the authoritative Engine
- Agent orchestration

Frontend may optionally maintain local simulation engines for analysis or visualization purposes. Such engines are strictly non-authoritative.

---

## 4.4 Agent Module

The Agent Module is an externally managed decision system.

It hosts AI agents, replay agents, and other automated decision systems.

### Responsibilities

- Receive GameState from Backend.
- Produce Action decisions.
- Optionally maintain local Engine instances for simulation, search, or analysis.
- Execute replay sequences as deterministic action providers.

### Authority Constraints

- Agent-local engines are shadow simulations only.
- The Agent Module has no authority over live GameState.
- All Actions must be submitted through Backend.
- Agents and human players are equivalent at the gameplay interaction layer.

Replay functionality is modeled as a specialized Agent that emits predetermined action sequences.

---

# 5. Module Relationships and Communication Boundaries

## 5.1 Authoritative Execution Chain (Live Game Mode)

In Live Game mode, the authoritative execution chain is:

Frontend / Agent
        ↓
     Backend
        ↓
      Engine

Only the Backend-managed Engine instance is authoritative.

No other Engine instance in the system may modify live state.

---

## 5.2 Engine ↔ Backend

- Backend invokes Engine through the canonical Engine contract.
- Engine returns updated GameState or structured error.
- Engine does not initiate communication.

This boundary enforces rule authority.

---

## 5.3 Backend ↔ Frontend

Two logical interaction domains exist:

Room Control:
- Create Room
- Join Room
- Seat selection
- Match start

Game Play:
- Backend pushes updated GameState.
- Frontend submits Action.

Frontend never interacts directly with the authoritative Engine.

---

## 5.4 Backend ↔ Agent Module

Two logical interaction domains exist:

Agent Control:
- Seat assignment
- Model selection
- Lifecycle management

Game Play:
- Backend pushes GameState.
- Agent submits Action.

From the Backend perspective, Agents and Human players are indistinguishable at the gameplay layer.

---

## 5.5 Agent ↔ Local Engine (Optional)

Within the Agent Module:

- Agents may maintain local Engine instances.
- Local Engines may be used for search, rollout, or heuristic evaluation.
- Local Engines must not override Backend authority.

---

# 6. Current Architectural Uncertainties

This document is marked In Development. The following areas are intentionally not finalized:

- Exact communication protocols (WebSocket, RPC, etc.).
- Backend state caching strategy.
- Internal Engine API signatures.
- Detailed Training System architecture.
- Deployment topology (single process vs distributed).

These areas will be defined in future Architecture or Protocol documents.

---

# Changelog

Version 1 (2026-02-17)
- Initial architecture definition separating Live Game and Training systems.
- Defined Engine as rule authority and Backend as orchestration authority.
- Established module responsibilities and communication boundaries.
