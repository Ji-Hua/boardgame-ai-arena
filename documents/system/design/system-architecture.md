# Quoridor System Architecture

Author: Ji Hua  
Created Date: 2026-02-17  
Last Modified: 2026-03-29  
Current Version: 3  

Document Type: Design  
Document Subtype: System Architecture  
Document Status: In Development  
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

This document focuses primarily on the Live Game System, while also clarifying how the Training System relates to the Engine.

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

The Training System reuses the Rule Engine as the canonical rule kernel and interacts with it through Rust-facing interfaces for performance-sensitive workloads.

The Training System does not depend on Live Game orchestration components.

---

# 3. Live Game System Design Goals

The Live Game System is designed with the following goals:

- Single authoritative live state  
- Clear separation of rule authority and orchestration  
- Strict module responsibility boundaries  
- Agent-human equivalence at the gameplay layer  
- Support for local simulation without compromising live authority  
- Deterministic and reproducible game progression  
- Reusable stateless rule logic across live play, replay, and training workflows  

---

# 4. Terminology Clarification

This document uses the term Engine in the broad system-architecture sense.

Within the Quoridor project, Engine has multiple related meanings:

## 4.1 Rule Engine

The Rule Engine is the stateless rule kernel.

It is responsible for:

- Rule evaluation  
- Geometric legality  
- Path-preservation checks  
- Deterministic state transitions  

## 4.2 Rust Engine

The Rust Engine refers to the concrete Engine Module implementation.

It includes:

- Rule Engine  
- Engine Interfaces  

## 4.3 Game Manager

Game Manager refers to a stateful orchestration layer that maintains game state and coordinates action execution using the Rule Engine.

Game Manager exists in multiple forms depending on system context:

- Live Game Manager — used in the Live Game System, responsible for authoritative state progression, action routing, and integration with Backend and external systems  
- Training Game Manager — used in the Training System, responsible for high-throughput simulation, replay, and search-oriented state evolution  

All Game Manager implementations:

- Must rely exclusively on the Rule Engine for rule semantics  
- Must not redefine rule behavior  

---

# 5. Live Game System Structure

## 5.1 Engine Module

The Engine Module is the canonical rule authority of the system.

It is internally structured into:

- Rule Engine (stateless)  
- Engine Interfaces (stateful access boundary)  

### Responsibilities

- Define all valid state transitions  
- Provide deterministic rule evaluation  
- Expose callable interfaces for external orchestration layers  
- Guarantee consistency of rule semantics across systems  

### Authority Model

- The Engine is the only module that defines valid state transitions  
- The Rule Engine is stateless and does not own GameState  
- No external module may directly mutate GameState  
- All state transitions must occur through Engine-defined interfaces  

### Non-Responsibilities

- Session management  
- Networking  
- UI rendering  
- Agent lifecycle management  
- Training orchestration  

---

## 5.2 Engine Internal Decomposition

The Engine Module is conceptually decomposed into:

- Rule Engine  
- Engine Interfaces  

### Rule Engine

- Stateless rule evaluation core  
- Accepts state and action  
- Produces deterministic results  

### Engine Interfaces

- Define how external systems interact with the Rule Engine  
- Provide a stable contract boundary  
- Do not impose a single stateful implementation model  

Game Management is not enforced as a single internal structure.

Instead, it is implemented externally depending on system context.

---

## 5.3 Rule Engine

The Rule Engine is a stateless computation layer responsible for rule evaluation.

### Core Characteristics

- Stateless  
- Deterministic  
- Side-effect free  
- Pure function semantics  

### Responsibilities

- Validate actions  
- Produce next state  
- Enforce geometric constraints  
- Ensure path existence  
- Construct valid state transitions  

### Internal Components

- APIs  
- Rule  
- Topology  
- Model  
- Calculation  

### Scope Constraints

The Rule Engine must not:

- Maintain live state  
- Manage sessions  
- Handle networking  
- Perform orchestration  
- Store history  

---

## 5.4 Game Management Patterns

Game state orchestration is not implemented as a single canonical structure inside the Engine Module.

Instead, multiple Game Management patterns exist.

---

### 5.4.1 Live Game Management

Used in the Live Game System.

Characteristics:

- Maintains authoritative live GameState  
- Receives user and agent actions  
- Coordinates with Backend  
- Supports observability and logging  

Implementation:

- Typically implemented outside the Engine Module  
- May be implemented in Python or other orchestration-friendly environments  
- Uses Engine Interfaces to invoke Rule Engine  

---

### 5.4.2 Training Game Management

Used in the Training System.

Characteristics:

- High-frequency simulation  
- Stateless or lightweight state cloning  
- Optimized for performance  
- Supports MCTS and self-play  

Implementation:

- Implemented in Rust  
- Directly uses Rule Engine APIs  
- Avoids external orchestration dependencies  

---

### 5.4.3 Shared Constraints

All Game Management implementations must satisfy:

- Rule Engine is the single source of truth  
- State transitions are deterministic  
- Identical action sequences produce identical final states  
- Rule semantics must not be redefined or bypassed  

---

## 5.5 Backend Module

The Backend Module is the orchestration layer for live gameplay.

### Responsibilities

- Manage Room lifecycle  
- Manage Game lifecycle  
- Instantiate Engine instances  
- Route player actions  
- Broadcast GameState updates  
- Manage player proxies  

### Authority Model

- Backend is orchestration authority  
- Backend does not define rules  
- Backend does not mutate GameState directly  
- Backend interacts only through Engine Interfaces  

---

## 5.6 Frontend Module

The Frontend Module provides user interaction.

### Responsibilities

- Display GameState  
- Submit actions  
- Manage UI state  

### Non-Responsibilities

- Rule validation  
- State authority  
- Direct Engine interaction  

---

## 5.7 Agent Module

The Agent Module is a decision-making system.

### Responsibilities

- Receive GameState  
- Produce actions  
- Optionally simulate locally  

### Constraints

- No authority over live state  
- Must route through Backend  
- Local engines are non-authoritative  

---

# 6. Module Relationships and Communication Boundaries

## 6.1 Authoritative Execution Chain

Frontend / Agent  
        ↓  
     Backend  
        ↓  
      Engine  

Only the Engine holds authoritative state.

---

## 6.2 Engine ↔ Backend

- Backend calls Engine Interfaces  
- Engine returns state or error  
- Engine does not initiate communication  

---

## 6.3 Backend ↔ Frontend

- Backend pushes GameState  
- Frontend submits actions  

---

## 6.4 Backend ↔ Agent

- Backend pushes state  
- Agent submits actions  

---

## 6.5 Agent ↔ Local Engine

- Local engines allowed for simulation  
- Must not override authoritative state  

---

## 6.6 Training System ↔ Engine

The Training System interacts directly with the Rule Engine through Rust-native interfaces.

Training workflows:

- Implement their own Game Manager  
- Do not depend on Backend  
- Must use the same Rule Engine  

Consistency requirement:

Identical inputs must produce identical outputs across Live and Training systems.

---

# 7. Architectural Invariants

The following invariants must hold:

- Rule Engine is stateless  
- Game Management is externalized  
- Engine defines all valid transitions  
- Backend does not define rules  
- Backend does not mutate state directly  
- Frontend is non-authoritative  
- Agent is non-authoritative  
- Live state exists only in Engine context  
- Training and Live systems share identical rule semantics  

---

# 8. Current Architectural Direction

Current implementation uses a single Rust crate with modular separation:

- rule_engine  
- engine_interfaces  

Game Management is intentionally externalized.

Future evolution may include:

- Python Live Game Manager  
- Rust Training Game Manager  
- FFI-based Engine Interfaces  

---

# 9. Architectural Uncertainties

The following areas remain undefined:

- Communication protocols  
- Backend caching strategy  
- Engine API signatures  
- Training system architecture  
- Deployment topology  
- Crate-level decomposition  

---

# Changelog

Version 3 (2026-03-29)
- Introduced explicit separation between Rule Engine and Game Manager  
- Defined dual Game Manager architecture (Live vs Training)  
- Removed single Game Context as canonical structure  
- Introduced Engine Interfaces abstraction  
- Clarified externalization of Game Management  
- Strengthened Training vs Live separation  
- Added shared invariants across Game Managers  

Version 2 (2026-03-29)
- Clarified Rule Engine vs Game Management Layer  
- Added terminology clarification  
- Added Engine internal decomposition  

Version 1 (2026-02-17)
- Initial architecture definition  
