# Quoridor — Geometric and Coordinate Model

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 3

Document Type: Design
Document Subtype: Internal Geometric Model
Document Status: In Development
Document Authority Scope: Engine module
Document Purpose:
This document defines the geometric model and coordinate systems used by the Quoridor engine. It formalizes the logical representation of squares, edges, and walls, and defines the human-facing display notation adopted by this system. The logical coordinate system is canonical. Display notation is a representation layer and must be converted into logical coordinates before rule evaluation.

---

## 1. Core Geometric Concepts

### 1.1 Square

- A square is a cell of the pawn grid.
- A pawn always occupies exactly one square.
- Squares are the only locations a pawn may occupy.

---

### 1.2 Edge

- An edge is the boundary segment between two adjacent squares.
- Pawn movement between adjacent squares crosses exactly one edge.
- If an edge is occupied by a wall segment, crossing that edge is forbidden.
- If an edge is not occupied, it does not restrict movement.

There are two edge orientations:

- horizontal — a horizontal boundary segment
- vertical — a vertical boundary segment

Edges are part of the logical coordinate system and represent geometric movement constraints.

---

### 1.3 Wall

- A wall consists of exactly two adjacent edges of the same orientation.
- A wall is atomic at the geometric and rule level.
- Geometrically, a wall blocks movement by occupying two edges.

A wall is represented using a head edge coordinate and extends along its orientation:

- A wall placed at (x, y, horizontal) occupies:
  - (x, y, horizontal)
  - (x, y+1, horizontal)

- A wall placed at (x, y, vertical) occupies:
  - (x, y, vertical)
  - (x+1, y, vertical)

Walls must respect board boundaries and must not overlap previously occupied edges.

In storage, walls are represented as two occupied edges.

---

## 2. Logical Coordinate System (Engine-Facing)

The logical coordinate system is the canonical geometric truth used by the engine.

### 2.1 Orientation

- Origin: bottom-left corner
- Origin coordinate: (0, 0, square)
- x increases to the right
- y increases upward
- The board lies entirely in the first quadrant

---

### 2.2 Board Size

- The board size is N × N
- N is defined by GameConfig

Pawn square coordinate bounds:

- 0 ≤ x < N
- 0 ≤ y < N

---

### 2.3 Unified Coordinate Representation

All logical entities are represented as:

(x, y, type)

where:

type ∈ { square, horizontal, vertical }

---

### 2.4 Square Coordinates

A pawn position is represented as:

(x, y, square)

with:

- 0 ≤ x < N
- 0 ≤ y < N

---

### 2.5 Edge Coordinates

Edges are anchored at the lower-left corner of a square.

For a square at (x, y):

- The horizontal edge above the square:
  (x, y, horizontal)

- The vertical edge to the right of the square:
  (x, y, vertical)

Edge bounds:

horizontal:
- 0 ≤ x < N
- 0 ≤ y < N-1

vertical:
- 0 ≤ x < N-1
- 0 ≤ y < N

Total number of edges:

2 × N × (N-1)

---

## 3. Display Coordinate System (Glendenning Notation)

Display notation is a human-facing representation layer used for logging, visualization, and user interaction.

The logical coordinate system remains canonical. All display coordinates must be converted into logical coordinates before rule evaluation.

This system adopts the widely used Glendenning-style notation by the Quoridor community.

Reference:
https://quoridorstrats.wordpress.com/notation/

In case of discrepancy, this document governs system behavior.

---

### 3.1 Board Notation

The board is labeled as follows:

- Files (columns): a, b, c, d, e, f, g, h, i
- Ranks (rows): 1, 2, 3, 4, 5, 6, 7, 8, 9

- a1 is the bottom-left square.
- i9 is the top-right square.

---

### 3.2 Pawn Move Notation

A pawn move is recorded by the destination square.

Example:

- e2 — move pawn to square e2

---

### 3.3 Wall Placement Notation

Walls are placed using intersection-based anchors.

Format:

<file><rank><orientation>

where:

- file ∈ {a–h}
- rank ∈ {1–8}
- orientation ∈ {h, v}

Examples:

- e3h — horizontal wall
- e3v — vertical wall

A horizontal wall (h) blocks vertical movement across two adjacent files.
A vertical wall (v) blocks horizontal movement across two adjacent ranks.

---

### 3.4 Conversion to Logical Coordinates

Square conversion:

Given square notation:

<file><rank>

Logical coordinates:

x = ord(file) - ord('a')
y = rank - 1

Logical representation:

(x, y, square)

---

Wall conversion:

Given wall notation:

<file><rank><orientation>

Let:

x = ord(file) - ord('a')
y = rank - 1

If orientation == 'h':

Logical head coordinate:

(x, y, horizontal)

If orientation == 'v':

Logical head coordinate:

(x, y, vertical)

Logical wall semantics are defined in Section 1.3.

---

## 4. Scope

This document defines the canonical geometric and coordinate model for the engine.

It does not define:

- Action schemas
- API structures
- Serialization formats
- Responsibility boundaries between modules

Those concerns are defined in Interface or Architecture documents.

---

# Changelog

Version 3 (2026-02-17)
- Renamed edge types from horizontal_edge/vertical_edge to horizontal/vertical.
- Rewrote display coordinate section to adopt Glendenning notation.
- Added external notation reference link.
- Clarified canonical logical system and conversion rules.

Version 2 (2026-02-17)
- Replaced right_edge/up_edge terminology with horizontal_edge/vertical_edge for geometric clarity.
- Clarified wall extension semantics using orientation-based definitions.

Version 1 (2026-02-17)
- Extracted geometric and coordinate model from coordination specification.
- Removed action schema and module responsibility content.
- Classified as Design document with subtype Internal Geometric Model.
