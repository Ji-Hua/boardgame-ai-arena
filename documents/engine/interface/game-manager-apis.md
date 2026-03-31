# GameManager Interface

Author: Ji Hua
Created Date: 2026-03-29
Last Modified: 2026-03-29
Current Version: 2
Document Type: Interface
Document Subtype: Game Lifecycle and State Management
Document Status: In Development
Document Authority Scope: Engine module
Document Purpose:
This document defines the external interface and behavioral contract of the GameManager component. It specifies lifecycle control, state management, history handling, and rule-engine delegation boundaries without defining internal implementation or extending system responsibilities.

---

# 1. Overview

GameManager is the stateful orchestration component responsible for managing a single game instance in the Live Game System.

GameManager:

- Owns and manages the current game state
- Maintains action and state history
- Delegates all rule semantics to the Rule Engine
- Supports undo and replay
- Exposes a controlled mutation boundary

GameManager does not:

- Implement rule validation logic
- Define rule semantics
- Manage networking or backend orchestration

This separation aligns with the Engine–Backend authority boundary defined in system architecture :contentReference[oaicite:0]{index=0}

---

# 2. Lifecycle Model

GameManager operates under a three-stage lifecycle:

UNINITIALIZED → RUNNING → TERMINAL

## 2.1 UNINITIALIZED

- No state exists
- Only initialization is allowed

## 2.2 RUNNING

- Actions may be submitted
- Undo is allowed
- State evolves deterministically

## 2.3 TERMINAL

- All mutation operations are disabled
- State and history are read-only

---

# 3. Data Exposure Contract

GameManager exposes the following conceptual data:

- initial_state
- current_state
- actions
- states
- terminal flag
- initialized flag

Constraints:

- initial_state is immutable
- actions is the canonical history
- states is derived from actions
- states[i] corresponds to the result of actions[:i+1]

---

# 4. API Specification

APIs are grouped by lifecycle stage.

---

## 4.1 Initialization

### initialize() -> None

Initializes the GameManager.

Behavior:

- Creates initial_state via Rule Engine
- Sets current_state
- Clears history
- Marks initialized

Constraints:

- Must fail if already initialized

---

### is_initialized() -> bool

Returns initialization status.

---

## 4.2 Lifecycle Control

### terminate() -> None

Marks the GameManager as terminal.

Behavior:

- Disables all mutation operations

Constraints:

- Must be idempotent

---

### is_terminal() -> bool

Returns terminal status.

---

## 4.3 Core Game Control

### submit_action(action: Action) -> ActionResult

Submits an action.

Behavior:

- Reject if not initialized
- Reject if terminal
- Delegates validation to Rule Engine
- On success:
  - Updates current_state
  - Appends to actions and states

---

### undo() -> bool

Reverts the last action.

Behavior:

- Reject if not initialized
- Reject if terminal
- Reject if no actions
- Updates current_state and history

---

## 4.4 State Query

### current_state() -> State

Returns current state.

---

### initial_state() -> State

Returns initial state.

---

### legal_actions() -> List[Action]

Returns legal actions for current state via Rule Engine.

---

## 4.5 Rule Semantic Query (Pass-through)

All functions in this section delegate to Rule Engine.

### is_game_over() -> bool

### winner() -> Optional[Player]

### remaining_walls(player: Player) -> int

### goal_cells(player: Player) -> Set[Square]

### path_exists(player: Player) -> bool

### shortest_path_len(player: Player) -> int

These functions are derived from the geometric and rule model defined in the engine design :contentReference[oaicite:1]{index=1}

---

## 4.6 History Query

### actions() -> List[Action]

Returns all actions.

---

### states() -> List[State]

Returns all non-initial states.

---

### step_count() -> int

Returns number of actions.

---

### get_state_at(step: int) -> State

Behavior:

- step = 0 → initial_state
- step > 0 → states[step - 1]

Constraints:

- Must fail if out of range

---

## 4.7 Debug / Consistency

### replay() -> State

Recomputes final state from actions.

Used for validation only.

---

# 5. Behavioral Constraints

## 5.1 Mutation Rules

| Operation      | RUNNING | TERMINAL |
|---------------|--------|----------|
| submit_action | ✔      | ❌       |
| undo          | ✔      | ❌       |
| query         | ✔      | ✔        |

---

## 5.2 Rule Delegation

GameManager must not:

- Implement rule logic
- Perform independent validation
- Modify rule semantics

All rule behavior is delegated to the Rule Engine.

---

## 5.3 Consistency Guarantee

GameManager ensures:

- actions and states remain synchronized
- no partial updates occur
- state history remains consistent

---

# 6. Non-Goals

GameManager does not:

- Manage sessions or rooms
- Handle networking
- Implement AI logic
- Support branching timelines
- Store invalid actions

---

# 7. Summary

GameManager is a deterministic state orchestration layer that:

- Applies actions safely
- Maintains history
- Exposes state
- Delegates all rule semantics

It serves as the boundary between Backend orchestration and Rule Engine authority.

---

# Changelog

Version 2 (2026-03-29)
- Refactored to comply with documentation governance standard.
- Changed Document Type to Interface.
- Added required header fields and reordered structure.
- Clarified API grouping and behavioral constraints.
- Removed implementation-level ambiguity.

Version 1 (2026-03-29)
- Initial draft of GameManager interface specification.
