# Quoridor System Architecture

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-04-19
Current Version: 4

Document Type: Design
Document Subtype: System Architecture
Document Status: In Development
Document Authority Scope: Global
Document Purpose:
This document defines the high-level system architecture of the Quoridor project. It establishes the structural decomposition of the system into major systems and modules, clarifies module responsibilities, and defines authority boundaries and inter-system relationships. This document does not define protocols, APIs, or implementation details.

---

# 1. System Overview

Quoridor is designed as a rule-centric AI platform built around a deterministic game engine.

The system consists of one shared foundational rule kernel and two major upper-layer systems:

1. Engine Module
2. Application System
3. Agent System

The Engine Module is the bottom-most rule authority of the entire platform.

The Application System is the complete user-facing entry system. It is the surface through which users interact with the platform. Its current implemented focus is live gameplay, but its long-term scope also includes user-facing workflows such as agent definition, parameter adjustment, evaluation, replay, and training-pipeline access.

The Agent System is the system in which agents are defined, evaluated, deployed, and evolved. It owns agent logic and lifecycle concerns. Training remains part of this system, but only as one internal subdomain rather than as a top-level system.

This document focuses primarily on the Application System, while also clarifying how the Agent System relates to the Engine Module.

---

# 2. High-Level System Decomposition

The system is decomposed into one foundational module and two major upper-layer systems.

## 2.1 Engine Module (Shared Foundation)

The Engine Module is the shared foundational rule kernel of the platform.

It is reused by both the Application System and the Agent System.

It defines the canonical game rules and the only valid state-transition semantics.

---

## 2.2 Application System (Current Focus)

The Application System is the complete user-facing entry layer of the platform.

Its current implemented focus is live gameplay and visualization.

Its long-term scope may include user-facing workflows such as:

- Live gameplay
- Replay
- Agent selection
- Agent definition
- Parameter adjustment
- Evaluation access
- Training-pipeline access

The Application System is composed of the following primary modules:

- Backend Module
- Frontend Module
- Agent Service Interface

The Application System does not define agent logic. It accesses agent capability only through service boundaries.

---

## 2.3 Agent System

The Agent System is the lifecycle system for agents.

It supports:

- Agent creation
- Agent evaluation
- Agent deployment
- Experimental analysis
- Training workflows

The Agent System reuses the Engine Module as the canonical rule kernel and interacts with it through Engine-facing interfaces appropriate to its workload.

The Agent System does not depend on Application-System orchestration components.

Within the Agent System, Training System is retained as a narrower internal submodule responsible for training-oriented workflows. It is no longer a top-level system parallel to the Application System.

---

# 3. Application System Design Goals

The Application System is designed with the following goals:

- Single authoritative live state
- Clear separation of rule authority and orchestration
- Strict module responsibility boundaries
- Human-agent gameplay interaction without exposing agent internals to the Application layer
- Support for user-facing workflows without compromising live authority
- Deterministic and reproducible game progression
- Reusable stateless rule logic across live play, replay, evaluation, and agent-system workflows

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

- Application Game Manager — used in the Application System, responsible for authoritative state progression, action routing, and integration with Backend and external systems
- Agent-System Game Management workflows — used in the Agent System, responsible for simulation, replay, search-oriented state evolution, evaluation, and future training workflows

All Game Manager implementations:

- Must rely exclusively on the Rule Engine for rule semantics
- Must not redefine rule behavior

---

# 5. Application System Structure

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
- Agent evaluation
- Agent deployment
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

### 5.4.1 Application Game Management

Used in the Application System.

Characteristics:

- Maintains authoritative live GameState
- Receives user and externally served agent actions
- Coordinates with Backend
- Supports observability and logging

Implementation:

- Typically implemented outside the Engine Module
- May be implemented in Python or other orchestration-friendly environments
- Uses Engine Interfaces to invoke Rule Engine

---

### 5.4.2 Agent-System Game Management

Used in the Agent System.

Characteristics:

- Simulation, replay, and evaluation support
- Stateless or lightweight state cloning
- Optimized for agent-oriented workflows
- Supports search, self-play, and future training

Implementation:

- May be implemented in Rust or other agent-system-appropriate environments
- Directly uses Rule Engine APIs or Engine Interfaces as needed
- Avoids Application-System orchestration dependencies

---

### 5.4.3 Shared Constraints

All Game Management implementations must satisfy:

- Rule Engine is the single source of truth
- State transitions are deterministic
- Identical action sequences produce identical final states
- Rule semantics must not be redefined or bypassed

---

## 5.5 Backend Module

The Backend Module is the orchestration layer for Application-System live gameplay.

### Responsibilities

- Manage Room lifecycle
- Manage Game lifecycle
- Instantiate Engine instances
- Route player actions
- Broadcast GameState updates
- Coordinate with the Agent Service Interface for agent-backed gameplay

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
- Serve as the current user-facing surface for live gameplay

### Non-Responsibilities

- Rule validation
- State authority
- Direct Engine interaction
- Agent definition
- Agent evaluation
- Agent training

---

## 5.7 Agent Service Interface

The Agent Service Interface is the Application-System boundary through which agent capability is exposed.

It serves as the live-game agent adapter layer between the Application System and the Agent System.

### Responsibilities

- Expose deployed agents to Application-System workflows
- Forward state and action-related requests across the service boundary
- Keep agent internals hidden from the Application System

### Non-Responsibilities

- Define agents
- Evaluate agents
- Train agents
- Decide deployment policy

---

# 6. Module Relationships and Communication Boundaries

## 6.1 Authoritative Execution Chain

In Application live gameplay, the authoritative execution chain is:

Frontend / Agent Service Interface
        ↓
     Backend
        ↓
      Engine

Only the Engine defines valid state transitions.

No Application-System component may override Engine authority.

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

## 6.4 Backend ↔ Agent Service Interface

- Backend coordinates agent-backed gameplay through the Agent Service Interface
- Backend provides game state and receives agent-produced actions through that boundary
- Backend does not depend on internal agent definitions

---

## 6.5 Agent Service Interface ↔ Agent System

- Agent Service Interface exposes only deployed agents from the Agent System
- Application workflows must not directly access agent internals
- Agent definitions, evaluation logic, deployment policy, and training workflows remain inside the Agent System

---

## 6.6 Agent System ↔ Local Engine Usage

Within the Agent System:

- Local Engine instances may be used for simulation, replay, search, evaluation, or training workflows
- Such Engine usage is non-authoritative with respect to Application live state
- Agent-System-local game evolution must still preserve Rule Engine semantics

---

## 6.7 Agent System ↔ Engine

The Agent System reuses the same Engine Module as the Application System.

Agent-System workflows:

- Implement their own game-management structures as needed
- Do not depend on Backend
- Must use the same Rule Engine semantics

Consistency requirement:

Identical inputs must produce identical outputs across Application and Agent-System workflows.

---

# 7. Architectural Invariants

The following invariants must hold:

- Rule Engine is stateless
- Engine defines all valid transitions
- Application Game Management is externalized from the Engine Module
- Agent-System game-management workflows are externalized from the Engine Module
- Backend does not define rules
- Backend does not mutate state directly
- Frontend is non-authoritative
- Application System does not define agent logic
- Agent System is the system of record for agent logic
- Training System is a submodule within Agent System, not a top-level parallel system
- Agent Service Interface exposes only deployed agents from Agent System
- Live application state exists only in Engine-mediated context
- Application and Agent systems share identical rule semantics

---

# 8. Current Architectural Direction

Current implementation uses a single Rust crate with modular separation:

- rule_engine
- engine_interfaces

Game Management is intentionally externalized.

Current implemented Application-System functionality is focused on live gameplay.

Future evolution may include:

- Broader Application-System user-entry workflows beyond live gameplay
- Python or other orchestration-friendly Application Game Management
- Expanded Agent-System evaluation, deployment, and training workflows
- FFI-based Engine Interfaces

---

# 9. Architectural Uncertainties

The following areas remain undefined:

- Communication protocols
- Backend caching strategy
- Engine API signatures
- Agent-System internal architecture
- Application-System expansion path beyond current live gameplay
- Deployment topology
- Crate-level decomposition

---

# Changelog

Version 4 (2026-04-19)
- Reframed the system into one shared Engine Module plus two upper-layer systems: Application System and Agent System.
- Renamed the former top-level Training System into Agent System.
- Reintroduced Training System only as a narrower internal submodule within Agent System.
- Reframed the former Live Game System as Application System, with live gameplay as the current implemented user-facing module.
- Replaced the former Agent Module inside live gameplay with Agent Service Interface as the boundary adapter between Application System and Agent System.
- Clarified that Application System does not define agent logic and that only deployed agents are exposed across the service boundary.

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
