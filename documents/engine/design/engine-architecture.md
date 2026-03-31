# Quoridor Engine Architecture

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-18

Current Version: 6
Document Type: Design
Document Subtype: Engine Architecture
Document Status: Draft
Document Authority Scope: Engine module
Document Purpose:
This document defines the structural architecture of the Quoridor Engine system. It specifies the decomposition of the Rule Engine and Game Manager, clarifies their responsibilities, lifecycle semantics, trust model, and internal structure. It does not define concrete API signatures or language-specific implementation details.

---

# 1. Engine System Structure

## 1.1 Two-Level Engine Model

The Quoridor Engine system is structured as a two-level engine model:

1. Rule Engine (stateless rule kernel)
2. Game Manager (stateful runtime container)

The Rule Engine internally follows a layered structure:

Topology → Model → Rule

The Game Manager depends on a Rule Engine instance but does not redefine rule semantics.

Lower layers define structural and rule truth.
Higher layers define runtime orchestration.

The system is designed to support live gameplay, replay systems, AI training, and simulation workflows.

---

## 1.2 Dependency Relationships

Dependency direction:

Game Manager → Rule Engine
Rule Engine → Topology → Model → Rule

Calculation utilities support Rule logic but do not define gameplay semantics.

The Rule Engine is self-contained and stateless.

Training workflows may interact directly with the Rule Engine and do not require a Game Manager.

---

## 1.3 Responsibility Separation

### Rule Engine

The Rule Engine is a stateless rule kernel responsible for validating and transforming Raw State under Quoridor rules.

It defines:

- What is legally true
- How state transitions occur
- How rule invariants are enforced

It does not:

- Maintain current game state
- Record history
- Manage sessions

Each external reference must construct and hold its own Rule Engine instance.
Rule Engine instances are not shared across games or contexts.

From a rule perspective, the Rule Engine is functionally complete.

---

### Game Manager

The Game Manager represents a single running game instance.

It defines:

- Runtime state container
- Game progression management
- Action history tracking
- Invocation of the Rule Engine

It introduces no new rule authority and does not extend rule capabilities.

It serves purely as a runtime orchestration layer built on top of a Rule Engine.

---

# 2. Rule Engine — Functional Specification

## 2.1 Core Transition Model

The Rule Engine operates as a pure state transformation system:

(Raw State, Action) → Raw State' | Rule Error

It guarantees:

- No mutation of input Raw State
- Deterministic output for identical inputs
- No hidden mutable state
- No side effects
- No runtime memory of prior invocations

Each invocation is independent.

---

## 2.2 Trust Model

The Rule Engine assumes the input Raw State is valid.

It:

- Does not implicitly re-validate state invariants on every transition
- Validates only the legality of the proposed Action

If the initial state is valid and all transitions are legal, all subsequent states remain valid.

Optional explicit state validation may exist but is not part of the default transition path.

Performance-critical execution and safety validation are intentionally separated.

---

# 3. Rule Engine Internal Structure

Within the Rule Engine, internal dependency direction is:

Topology → Model → Rule

Calculation utilities support Rule logic but do not define gameplay semantics.

---

## 3.1 Topology Layer

### Purpose

Defines the geometric structure of the board independently of rule logic.

Topology describes connectivity and spatial relationships but does not encode rule semantics.

---

### Responsibilities

Topology defines:

- Board dimensions
- Valid square positions
- Valid wall anchor positions
- Adjacency relationships
- Connectivity graph
- Start positions and goal regions
- Geometric boundary constraints

Topology is immutable and independent of GameState.

---

## 3.2 Model Layer

### Purpose

Defines structural data representations.

Contains no rule enforcement or rule-derived computation logic.

---

### Core Entities

The Model layer defines representations for:

- Square
- Wall
- Pawn
- Player
- Action
- GameConfig
- GameState

---

### GameState Structure

GameState consists of:

#### Raw State (Canonical)

Contains rule-relevant information:

- Current player
- Pawn positions
- Wall positions
- Remaining Walls per player

Rule semantics depend exclusively on Raw State.
Note: Remaining walls are canonical rule constraints and MUST be maintained by Rule Engine transitions.

---

#### Derived Views (Non-Canonical)

Derived information may include:

- Game-over condition
- Winning player
- Shortest path lengths
- Connectivity checks

Derived must satisfy:

Derived = f(Raw State, Topology, Rule Semantics)

Derived:

- Does not participate in rule validation
- Does not affect state transitions
- Does not define state equivalence
- Is computed by the Rule Engine

The Model layer does not compute derived values.
The Rule Engine may compute derived projections when requested.

---

## 3.3 Rule Layer

### Purpose

Defines legal state transitions and rule-derived semantics under Quoridor rules.

Operates as:

(Raw State, Action) → Raw State' | Rule Error

Additionally, the Rule layer is responsible for computing rule-derived semantic projections from Raw State.

---

### Transition Phases

1. Structural Precondition Checks
2. Geometric Validation (Topology)
3. Rule Validation (turn, occupancy, wall stock)
4. Invariant Validation (path existence via Calculation)
5. Raw Transformation
6. Result Return

The Rule layer defines semantic transformation.
Object construction is delegated to the Model layer.

---

### Invariant Closure Principle

If:

- The initial Raw State is valid
- Every transition is legal

Then:

- All subsequent Raw States remain valid

---

### Legal Action Evaluation

Action validation reuses transition logic without producing new state.

Legal action generation:

1. Enumerate candidate actions
2. Validate each
3. Collect valid actions

Correctness is guaranteed; performance strategy is not defined here.

---

## 3.4 Calculation Module

### Role

Provides algorithmic support:

- Graph traversal
- Shortest path computation
- Connectivity validation

Does not define rule semantics.

All computations must be deterministic.

---

# 4. Rule Engine Lifecycle

## 4.1 Construction Phase

Constructed from GameConfig.

Initializes:

- Topology
- Rule logic
- Calculation utilities

After construction:

- Structure is fixed
- Topology is immutable
- Rule semantics cannot be altered
- No GameState is created

Each external owner manages its own instance.

---

## 4.2 Initial State Generation

The Rule Engine provides canonical Raw State generation.

Initial Raw State:

- Derived solely from GameConfig
- Satisfies all rule invariants
- Contains no derived projections

Forms the base of invariant closure.

---

## 4.3 Operational Phase

During operation:

- Accepts Raw State and Action
- Produces new Raw State or Rule Error
- Maintains no internal runtime state

---

## 4.4 Destruction Phase

The Rule Engine:

- Holds no runtime state
- Requires no special teardown
- Does not affect existing GameState instances

---

# 5. Game Manager

## 5.1 Purpose

The Game Manager represents a single running game instance.

It is a stateful runtime container built on top of a Rule Engine.

---

## 5.2 Responsibilities

The Game Manager:

- Maintains the authoritative current GameState
- Maintains action and state history
- Delegates all rule semantics to the Rule Engine
- Invokes the Rule Engine for transitions
- Ensures sequential state updates
- Supports undo (revert last action)

It does not:

- Redefine rule semantics
- Implement or infer rule validation logic
- Independently track game-over state (delegates to Rule Engine queries)
- Modify Raw State directly
- Bypass the Rule Engine
- Record invalid actions

---

## 5.3 State Ownership and History Model

The Game Manager is the sole owner of runtime state.

Rule Engine:

- Does not store state
- Does not modify external state
- Returns new Raw State instances only

All state mutation occurs through controlled replacement:

current_state ← RuleEngine transition result

History model:

- initial_state is stored separately and is immutable
- actions is an append-only list of accepted actions (source of truth)
- states is a list of non-initial states derived from actions
- len(actions) == len(states)
- states[i] corresponds to the result of applying actions[0..=i]
- Invalid actions are rejected and never recorded

---

## 5.4 Game Manager Lifecycle

The Game Manager operates under a three-stage lifecycle:

UNINITIALIZED → RUNNING → TERMINAL

### UNINITIALIZED

- No state exists
- Only initialization is allowed

---

### RUNNING

For each action:

1. Retrieve current Raw State
2. Invoke Rule Engine
3. Receive new Raw State or Rule Error
4. If successful:
   - Update current_state
   - Append action and resulting state to history

Undo is permitted during this stage.

---

### TERMINAL

Terminal is externally controlled and represents a mutation freeze.

- All mutation operations (submit_action, undo) are disabled
- All query and read operations remain available
- Terminal is idempotent

Destruction of Game Manager does not affect the Rule Engine.

---

# 6. Architectural Guarantees

This architecture guarantees:

- Clear separation between rule logic and runtime management
- Deterministic state transitions
- Explicit trust boundaries
- Canonical rule truth based on Raw State
- Derived views as pure projections
- No hidden mutable global state
- Single authoritative runtime state owner

The system answers:

“What is legally true in this game world?”

It does not answer:

“What move should be chosen?”

---

# Changelog

Version 6 (2026-02-18)
- Refined the semantic definition and responsibility boundaries of Derived GameState projections.
- Clarified that derived values are computed by the Rule Engine and not by the Model layer

Version 5 (2026-02-18)
- Integrated full Game Manager structural specification into the architecture document.
- Clarified state ownership and lifecycle boundaries between Rule Engine and Game Manager.
- Expanded runtime orchestration responsibilities.

Version 4 (2026-02-17)
- Renamed Game Engine to Game Manager.
- Clarified Rule Engine completeness.

Version 3 (2026-02-17)
- Integrated Rule Engine Design content into architecture document.

Version 2 (2026-02-17)
- Introduced two-level engine structure and trust model.

Version 1 (2026-02-17)
- Initial draft of layered engine architecture.
