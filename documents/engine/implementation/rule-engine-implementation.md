# Quoridor Rule Engine Implementation

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-03-28
Current Version: 2
Document Type: Implementation
Document Subtype: Rule Engine Implementation
Document Status: In Development
Document Authority Scope: Engine module
Document Purpose:
This document defines the concrete internal implementation of the Quoridor Rule Engine, including data structures, algorithms, and execution flow. It specifies how the engine realizes rule validation, state transitions, and action enumeration without redefining system-level design constraints or interface contracts.

---

# 1. Implementation Scope

This document defines:

- Internal data structures
- Internal algorithms
- Execution flow
- Validation procedures

This document does NOT define:

- Architectural constraints
- Cross-module contracts
- Public APIs

---

# 2. RawState Representation

## 2.1 Structure

struct RawState {
    pawn_positions: [Position; 2],
    horizontal_walls: u128,
    vertical_walls: u128,
    remaining_walls: [u8; 2],
    current_player: Player,
}

---

## 2.2 Pawn Representation

- Stored as (x, y)
- Zero-based indexing

---

## 2.3 Wall Representation

Walls are encoded using bitsets.

Horizontal walls:

- Grid: N × (N-1)
- Index mapping:
  index = x * (N-1) + y

Vertical walls:

- Grid: (N-1) × N
- Index mapping:
  index = x * N + y

---

## 2.4 Remaining Walls

- Stored per player
- Updated during wall placement

---

## 2.5 Current Player

- Stored explicitly
- Updated after each valid action

---

## 2.6 Immutability

- RawState is treated as immutable
- All transitions produce a new instance

---

# 3. Action Representation

struct Action {
    player: Player,
    kind: ActionKind,
    target: Coordinate,
}

---

## 3.1 Properties

- Represents intent only
- Immutable
- Contains no validation logic

---

# 4. Connectivity Implementation

## 4.1 Algorithm

- Breadth-First Search (BFS)
- Graph constructed on-the-fly

---

## 4.2 Functions

fn path_exists(raw: &RawState, player: Player) -> bool

fn shortest_path_len(raw: &RawState, player: Player) -> Option<u32>

---

## 4.3 Implementation Notes

- No persistent graph storage
- Adjacency computed per expansion step

---

# 5. Wall Blocking Logic

## 5.1 Horizontal Wall

Blocks:

(x, y) ↔ (x, y+1)  
(x+1, y) ↔ (x+1, y+1)

---

## 5.2 Vertical Wall

Blocks:

(x, y) ↔ (x+1, y)  
(x, y+1) ↔ (x+1, y+1)

---

## 5.3 Reuse Strategy

- Same blocking logic is reused in:
  - Movement validation
  - BFS expansion

---

# 6. Pawn Movement Validation

## 6.1 Validation Order

1. Adjacent move
2. Forward jump (if blocked by opponent)
3. Side-step (if forward jump unavailable)

---

## 6.2 Checks

- Boundary check
- Wall blocking check
- Occupancy check

---

# 7. Wall Placement Validation

## 7.1 Validation Steps

1. Bounds check
2. Overlap check
3. Crossing check
4. Remaining wall check
5. Path existence check (via BFS)

---

# 8. Legal Action Generation

## 8.1 Strategy

Enumerate → Validate → Collect

---

## 8.2 Implementation

- Generate all candidate pawn moves
- Generate all candidate wall placements
- Validate each candidate
- Collect valid actions

---

## 8.3 Constraint

- Validation logic is reused
- No separate rule logic is introduced

---

# 9. Derived Computation

## 9.1 Examples

- Game termination detection
- Winner determination
- Shortest path length

---

## 9.2 Strategy

- Computed on-demand
- Stateless functions over RawState

---

# 10. Rule Engine Structure

## 10.1 Composition

struct RuleEngine {
    topology: Topology,
}

---

## 10.2 Internal Modules

- Validation logic
- BFS connectivity
- Action generation
- Derived computation

---

# 11. Execution Flow

## 11.1 apply_action

1. Validate action  
2. Apply state transition  
3. Produce new RawState  

---

## 11.2 validate_action

Performs:

- Structural validation
- Geometric validation
- Rule-specific checks

---

## 11.3 legal_actions

1. Enumerate candidates  
2. Validate each  
3. Return valid actions  

---

# 12. Non-Goals (v0)

The following are not implemented:

- Zobrist hashing
- Incremental updates
- Performance optimizations
- GameManager
- Replay system
- AI/search integration

---

# 13. Future Implementation Directions

Possible future improvements:

- Incremental BFS
- Cached derived values
- Alternative pathfinding algorithms

---

# 14. Determinism Guarantee

This implementation ensures:

- Deterministic execution
- No hidden state
- Consistent validation behavior

---

# Changelog

Version 2 (2026-03-28)
- Refactored document to comply with documentation governance standard.
- Removed design-level constraints and retained implementation-only content.
- Added required header fields and structured sections.
- Introduced explicit execution flow and validation breakdown.

Version 1 (2026-02-17)
- Initial implementation specification draft.
