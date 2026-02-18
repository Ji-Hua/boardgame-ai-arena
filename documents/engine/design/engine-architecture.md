# Quoridor Engine Architecture

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17

Current Version: 2
Document Type: Design
Document Subtype: Engine Architecture
Document Status: Draft
Document Authority Scope: Engine module
Document Purpose:
This document defines the structural architecture of the Quoridor Engine. It specifies the decomposition of topology, model, rule, and engine runtime components, and clarifies their responsibilities, lifecycle semantics, and trust model. It does not define concrete API signatures or implementation details.

---

# 1. Architectural Overview

The Quoridor Engine is structured as a two-level engine system:

1. Rule Engine (stateless rule kernel)
2. Game Engine (stateful game runtime container)

The Rule Engine internally follows a layered rule architecture composed of:

Topology → Model → Rule

The Game Engine depends on a Rule Engine instance but does not redefine rules.

Lower layers define structural truth.
Higher layers define runtime orchestration.

The engine is designed to function as a deterministic rule kernel suitable for live gameplay, replay systems, and AI training environments.

---

# 2. Rule Engine

## 2.1 Purpose

The Rule Engine is a stateless rule kernel responsible for validating and transforming game state under Quoridor rules.

It does not maintain game runtime information such as current state or history.

Each reference must hold its own Rule Engine instance. Rule Engines are not shared across games or contexts.

---

## 2.2 Construction Semantics

The Rule Engine is constructed from a GameConfig.

During construction, it initializes:

- Topology (geometric structure of the board)
- Model definitions (data structures)
- Rule logic
- Calculation utilities (e.g., BFS)

Topology is immutable once constructed and is private to the Rule Engine instance.

---

## 2.3 Internal Layering

Within the Rule Engine, the internal dependency direction is:

Topology → Model → Rule

Calculation utilities support Rule logic but do not define gameplay semantics.

---

# 3. Topology Layer

## 3.1 Purpose

The Topology layer defines the geometric structure of the board independently of game rules.

It describes connectivity and spatial relationships but does not encode rule semantics such as wall legality or turn order.

---

## 3.2 Responsibilities

The Topology layer defines:

- Board dimensions
- Valid square positions
- Valid wall anchor positions
- Adjacency relationships
- Connectivity graph
- Start positions and goal regions
- Geometric boundary constraints

Topology is immutable and independent of GameState.

---

# 4. Model Layer

## 4.1 Purpose

The Model layer defines structural data representations.

It contains no rule enforcement logic.

---

## 4.2 Core Entities

The Model layer defines representations for:

- Square
- Wall
- Pawn
- Player
- Action
- GameConfig
- GameState

These are structural definitions only.

---

## 4.3 GameState Structure

GameState consists of two conceptual components:

### 4.3.1 Raw State (Canonical)

The raw state contains authoritative rule-relevant information:

- Current player
- Pawn positions
- Wall positions
- Wall ownership

The raw state is the sole input used by the Rule Engine for state transitions.

Rule semantics depend exclusively on the raw state.

---

### 4.3.2 Derived Views (Non-Canonical)

Derived information includes:

- Remaining walls per player
- Game-over condition
- Winning player
- Shortest path lengths
- Connectivity checks

Derived views must satisfy:

Derived = f(Raw State, Topology)

Derived values:

- Do not participate in rule validation
- Do not affect state transitions
- Do not define state equivalence
- May be omitted without affecting rule correctness

The Rule Engine ignores derived values during validation and transition.

---

# 5. Rule Layer

## 5.1 Purpose

The Rule layer defines legal state transitions under Quoridor rules.

It operates as a pure state transformation system:

(Raw State, Action) → Raw State' | Rule Error

The Rule layer does not mutate input state.

---

## 5.2 Trust Model

The Rule Engine follows an explicit trust model:

- It assumes the input Raw State is valid.
- It does not implicitly re-validate state invariants on every transition.
- It validates only the legality of the proposed Action.

If the initial state is valid and all transitions are legal, all subsequent states remain valid.

Optional state validation may exist but is not part of the default transition path.

Performance-critical execution and safety validation are intentionally separated.

---

## 5.3 Responsibilities

The Rule layer is responsible for:

- Action validation
- Pawn movement legality
- Wall placement legality
- Turn enforcement
- Path existence enforcement
- Game termination detection
- Deterministic state transitions

Rule computations may rely on:

- BFS or equivalent graph traversal
- Shortest path computation
- Connectivity analysis

All rule computations must be deterministic.

---

## 5.4 Error Semantics

Rule errors represent rule-level violations.

Errors must be:

- Structured
- Machine-readable
- Independent of internal topology representation details

---

# 6. Game Engine

## 6.1 Purpose

The Game Engine represents a single running game.

It is a stateful runtime container built on top of a Rule Engine instance.

---

## 6.2 Responsibilities

The Game Engine:

- Maintains the authoritative current GameState
- Maintains action history
- Manages game progression
- Invokes the Rule Engine to perform state transitions

The Game Engine does not redefine rule semantics.

---

## 6.3 Lifecycle

The Game Engine lifecycle is bound to a single game instance.

The Rule Engine lifecycle is managed externally and may outlive or be reused independently of any specific game runtime.

---

# 7. Training Context

Training workflows interact directly with the Rule Engine.

They:

- Manage GameState externally
- Do not require a Game Engine
- Rely exclusively on deterministic rule transformations

---

# 8. Architectural Guarantees

This architecture guarantees:

- Clear separation between rule logic and game runtime
- Deterministic state transitions
- Explicit trust boundaries
- Single canonical rule truth based on Raw State
- Derived views as observational projections
- No hidden mutable global state

The engine answers:

“What is legally true in this game world?”

It does not answer:

“What move should be chosen?”

---

# Changelog

Version 2 (2026-02-17)
- Introduced two-level engine structure (Rule Engine and Game Engine).
- Clarified GameState raw vs derived semantics.
- Added explicit trust model for state validation.
- Separated runtime container responsibilities from rule logic.

Version 1 (2026-02-17)
- Initial draft of layered engine architecture with Topology, Model, Rule, and API separation.
