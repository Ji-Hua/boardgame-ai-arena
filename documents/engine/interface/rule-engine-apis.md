# Quoridor Rule Engine API

Author: Ji Hua
Created Date: 2026-02-18
Last Modified: 2026-02-18
Current Version: 1
Document Type: Interface
Document Subtype: Rule Engine API
Document Status: In Development
Document Authority Scope: Engine module
Document Purpose:
This document defines the public API contract of the Quoridor Rule Engine. It specifies function signatures, input and output structures, invocation semantics, and error handling behavior. This document operates strictly within the structural boundaries defined by the Engine Architecture Design and does not define internal algorithms or implementation details.

---

# 1. Overview

The Rule Engine is a stateless rule kernel responsible for validating and transforming RawState under Quoridor rule semantics.

All functions defined in this document:

- Are deterministic.
- Must not mutate input parameters.
- Must not maintain internal runtime state.
- Must not depend on hidden global state.
- Operate strictly within the responsibilities defined by the Engine Architecture Design.

Unless explicitly stated otherwise, functions return either a successful value or a RuleError.

---

# 2. Core Types

This section defines abstract types used in the API signatures. These are conceptual types and are not language-specific.

## 2.1 RuleEngine

An immutable rule kernel instance constructed from GameConfig.

A RuleEngine:

- Encapsulates topology.
- Encapsulates rule semantics.
- Encapsulates calculation utilities.
- Does not store or mutate RawState.

---

## 2.2 GameConfig

Configuration object used to construct a RuleEngine.

GameConfig defines:

- Board size.
- Initial wall counts.
- Any structural parameters required by the rule system.

GameConfig must be valid at construction time. Invalid configurations must cause construction failure.

---

## 2.3 RawState

Canonical rule-relevant game state.

RawState contains only rule-truth data, including:

- Current player.
- Pawn positions.
- Wall positions.
- Remaining walls.

RawState:

- Must be treated as immutable by consumers.
- Does not include derived projections.
- Does not include history.

---

## 2.4 Action

Canonical structured action representation.

An Action includes:

- player
- type
- target

Action must conform to the Action Schema defined in the Interface layer.

---

## 2.5 RuleError

Canonical structured error object.

RuleError includes:

- code (machine-readable)
- optional message
- optional details

Consumers must rely exclusively on the error code for programmatic behavior.

---

## 2.6 Result<T>

Abstract success-or-error return structure.

A function returning Result<T> returns either:

- Success containing a value of type T
- Failure containing a RuleError

The specific language representation of Result<T> is implementation-defined.

---

# 3. Construction API

## 3.1 create_rule_engine

Signature:

create_rule_engine(config: GameConfig) -> RuleEngine

Semantics:

- Constructs an immutable RuleEngine instance.
- Validates GameConfig.
- Initializes topology and rule semantics.
- Does not create RawState.
- Must fail immediately if config is invalid.

---

# 4. State Bootstrap API

## 4.1 initial_state

Signature:

initial_state(engine: RuleEngine) -> RawState

Semantics:

- Returns the canonical initial RawState.
- The returned RawState satisfies all rule invariants.
- Does not include derived projections.
- Does not depend on external state.

---

# 5. Transition API

## 5.1 apply_action

Signature:

apply_action(engine: RuleEngine, raw: RawState, action: Action)
    -> Result<RawState>

Semantics:

- Validates the action against the given RawState.
- If invalid, returns RuleError.
- If valid, returns a new RawState.
- Must not mutate the input RawState.
- Must be deterministic.
- Must internally rely on the same validation logic as validate_action.

---

## 5.2 validate_action

Signature:

validate_action(engine: RuleEngine, raw: RawState, action: Action)
    -> Result<void>

Semantics:

- Validates legality of the action under rule semantics.
- Performs all structural, geometric, rule, and invariant checks.
- Does not produce a new RawState.
- Must share rule-validation logic with apply_action.
- Must not modify input parameters.

---

## 5.3 legal_actions

Signature:

legal_actions(engine: RuleEngine, raw: RawState)
    -> List<Action>

Semantics:

- Returns all legal actions for the current RawState.
- Must include all and only valid actions.
- Must not include illegal actions.
- Must not attach heuristic metadata.
- Order of returned actions is not semantically guaranteed.

---

# 6. Defensive Validation API

## 6.1 validate_state

Signature:

validate_state(engine: RuleEngine, raw: RawState)
    -> Result<void>

Semantics:

- Validates whether RawState is a rule-legal state.
- Intended for non-trusted state sources.
- Must verify structural validity and rule invariants.
- Must not modify RawState.
- Must not be implicitly invoked by apply_action.
- Must not depend on historical state.

---

# 7. Rule Semantic Query API

## 7.1 is_game_over

Signature:

is_game_over(engine: RuleEngine, raw: RawState) -> bool

Semantics:

- Returns true if the game has reached a terminal state.
- Returns false otherwise.
- Must not modify state.

---

## 7.2 winner

Signature:

winner(engine: RuleEngine, raw: RawState)
    -> Optional<Player>

Semantics:

- Returns the winning player if the game is over.
- Returns None if the game is not over.
- Must not raise error for non-terminal states.

---

## 7.3 remaining_walls

Signature:

remaining_walls(engine: RuleEngine, raw: RawState, player: Player)
    -> Result<int>

Semantics:

- Returns the number of remaining walls for the given player.
- Must fail if player identifier is invalid.
- Must not modify state.

---

## 7.4 goal_cells

Signature:

goal_cells(engine: RuleEngine, player: Player)
    -> Result<Set<Square>>

Semantics:

- Returns the set of goal squares for the specified player.
- Depends only on topology and rule semantics.
- Must fail if player identifier is invalid.
- Independent of RawState.

---

## 7.5 path_exists

Signature:

path_exists(engine: RuleEngine, raw: RawState, player: Player)
    -> Result<bool>

Semantics:

- Returns whether at least one legal path to goal exists for the player.
- Must fail if player identifier is invalid.
- Must not expose internal traversal details.
- Must not modify state.

---

## 7.6 shortest_path_len

Signature:

shortest_path_len(engine: RuleEngine, raw: RawState, player: Player)
    -> Result<int>

Semantics:

- Returns the length of the shortest legal path to goal.
- Must fail if player identifier is invalid.
- Must not modify state.
- Must not expose internal algorithm details.

---

# 8. Non-Goals

This API does not define:

- Internal data structures.
- Internal algorithms.
- Performance guarantees.
- Caching strategies.
- Serialization formats.
- Replay formats.
- Text protocols.
- Training optimization interfaces beyond those explicitly defined.

---

# Changelog

Version 1 (2026-02-18)
- Initial Rule Engine API document created.
