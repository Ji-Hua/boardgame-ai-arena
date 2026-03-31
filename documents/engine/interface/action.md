# Quoridor Action Model (Logical Coordinate Based)

## Overview

An Action represents a single atomic player intent under the canonical logical coordinate system.

All coordinates used in Action are **engine-facing logical coordinates**, not display notation.

The logical coordinate system is defined as:

(x, y, kind)

where:

- x, y are zero-based indices
- kind ∈ { Square, Horizontal, Vertical }

---

## Action Structure

Action consists of three fields:

- player
- kind
- target

### Definition

Action:

- player: Player
- kind: ActionKind
- target: Coordinate

Coordinate:

- x: integer
- y: integer
- kind: CoordinateKind

---

## Enumerations

### Player

- 1
- 2

Currently only two-player games are supported.

---

### ActionKind

- MovePawn
- PlaceWall

---

### CoordinateKind

- Square
- Horizontal
- Vertical

---

## Semantic Constraints

### MovePawn

If:

- action.kind == MovePawn

Then:

- action.target.kind MUST be Square
- (x, y) MUST satisfy square bounds:
  - 0 ≤ x < N
  - 0 ≤ y < N

Violation results in:

- INVALID_ACTION_KIND or PAWN_MOVE_OUT_OF_BOUNDS

---

### PlaceWall

If:

- action.kind == PlaceWall

Then:

- action.target.kind MUST be Horizontal or Vertical
- (x, y) MUST satisfy wall bounds:

For Horizontal:
- 0 ≤ x < N
- 0 ≤ y < N-1

For Vertical:
- 0 ≤ x < N-1
- 0 ≤ y < N

Violation results in:

- INVALID_ACTION_KIND or WALL_OUT_OF_BOUNDS

---

## Design Principles

- Action expresses player intent only.
- Action does not encode rule validation.
- Action does not encode derived state.
- Action does not use display notation.
- Action does not depend on internal ID encoding.
- Action is immutable.
- Action semantics are determined exclusively by the Rule Engine.

---

## Determinism

For identical:

- PrimaryState
- Action

The Rule Engine MUST produce identical results.
