# Agent Server Architecture

Author: Ji Hua
Created Date: 2026-04-05
Last Modified: 2026-04-05
Current Version: 1
Document Type: Design
Document Subtype: Agent Server Architecture
Document Status: In Development
Document Authority Scope: Agent module
Document Purpose:
This document defines the architectural role, responsibilities, and system integration model of the Agent Server within the Quoridor system. It formalizes the Agent Server as a standalone module that provides decision-making capabilities for automated players (e.g., AI agents and replay agents), and defines its interaction boundaries with the Backend and Frontend systems under the authoritative game execution model.

---

# 1. Overview

The Agent Server is a standalone service within the Agent Module that provides automated decision-making capabilities for gameplay.

It acts as a unified execution layer for all non-human players, including:

- AI Agents
- Replay Agents
- Future automated strategies

The Agent Server does not participate in rule evaluation or state authority. It operates strictly as a decision provider.

---

# 2. Architectural Position

Within the Quoridor system architecture, the Agent Server belongs to the Agent Module and interacts exclusively with the Backend Module.

It does not interact directly with:

- The Engine Module
- The Frontend Module

All interactions with the game system must pass through the Backend.

---

# 3. Design Principles

The Agent Server is governed by the following principles:

## 3.1 Non-Authoritative Behavior

- The Agent Server does not own or mutate GameState.
- The Agent Server does not validate rules.
- The Agent Server does not interact with the Engine.

All authoritative state transitions are controlled by the Backend and Engine.

---

## 3.2 Backend-Orchestrated Execution

- The Agent Server is fully controlled by the Backend.
- The Agent Server cannot initiate gameplay actions.
- The Agent Server only responds to Backend requests.

---

## 3.3 Agent-Human Equivalence

- Agents and human players are treated equivalently at the gameplay layer.
- Both provide actions in response to Backend requests.
- The difference lies only in how decisions are generated.

---

## 3.4 Opaque Decision Model

- The Agent Server is treated as an opaque decision provider.
- It may be internally stateful or stateless.
- The Backend must not depend on internal Agent state.

The only reliable input to the Agent Server is the GameState provided by the Backend.

---

## 3.5 Unified Agent Abstraction

All automated gameplay behaviors must be implemented as Agents within the Agent Server.

This includes:

- AI-based decision systems
- Replay systems
- Scripted agents

Replay is formally modeled as a deterministic Agent.

---

# 4. Internal Structure

The Agent Server may host multiple agent implementations.

Example structure:

Agent Server
  ├── AI Agent
  ├── Replay Agent
  └── (Future Agents)

Each agent implementation is responsible for:

- Receiving GameState input
- Producing a valid Action
- Managing its own internal state (if any)

The internal structure of agent implementations is outside the scope of this document.

---

# 5. Communication Model

The Agent Server participates in two distinct communication planes:

---

## 5.1 Control Plane

The Control Plane manages agent lifecycle and configuration.

Flow:

User → Frontend → Backend → Agent Server

Responsibilities:

- Agent registration
- Agent selection (AI / Replay)
- Seat binding
- Agent initialization
- Agent start / stop
- Parameter configuration

The Agent Server must not independently manage its lifecycle; it is fully controlled by the Backend.

---

## 5.2 Gameplay Plane

The Gameplay Plane handles decision-making during gameplay.

Execution cycle:

1. Backend sends GameState to Agent Server
2. Agent Server produces an Action
3. Backend receives Action
4. Backend invokes Engine to validate and apply Action
5. Backend broadcasts updated GameState to Frontend

Formal flow:

Backend → Agent Server → Backend → Engine → Backend → Frontend → User

---

# 6. Authority Boundaries

The Agent Server operates under strict authority constraints:

- It cannot modify GameState
- It cannot bypass Backend
- It cannot interact with Engine directly
- It cannot initiate gameplay actions

The Backend remains the sole orchestrator of gameplay.

The Engine remains the sole authority for rule validation and state transitions.

---

# 7. Relationship with Replay

Replay functionality is implemented as a specialized Agent.

Replay Agents:

- Receive GameState from Backend
- Emit predetermined actions based on a replay sequence
- Follow the same communication path as any other Agent

Replay must not be executed on the Frontend or bypass Backend validation.

---

# 8. Scope

This document defines:

- The architectural role of the Agent Server
- Its responsibilities and constraints
- Its interaction model with Backend
- Its internal abstraction model for agents

This document does not define:

- Communication protocols or message schemas
- API definitions
- Serialization formats
- Internal agent algorithms
- Performance strategies

These concerns are defined in Interface or Implementation documents.

---

# Changelog

Version 1 (2026-04-05)
- Initial definition of Agent Server architecture
- Defined Control Plane and Gameplay Plane separation
- Established authority boundaries and interaction model
- Formalized Replay as a specialized Agent
