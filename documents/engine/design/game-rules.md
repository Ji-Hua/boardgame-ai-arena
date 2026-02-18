# Game Rules

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 1

Document Type: Design
Document Subtype: External Domain Specification
Document Status: In Development
Document Authority Scope: Engine module
Document Purpose:
This document formalizes the official rules of the board game Quoridor as external domain constraints adopted by this system. These rules define the semantic invariants and legality conditions that the Engine must enforce. This document does not define system architecture, APIs, or implementation details.

---

## 1. Overview

This document describes the standard rules for the board game Quoridor. It covers equipment, setup, pawn movement, wall placement, turn order, winning conditions, illegal moves, and common variants.

The rules defined here constitute domain-level constraints. All Engine implementations must enforce these rules. No implementation may alter or reinterpret these rules without formally updating this document.

---

## 2. Equipment

- Board: A 9x9 grid of squares (rows and columns numbered 1..9 in implementations).
- Pawns: One pawn per player. Standard 2-player game uses 2 pawns; 4-player variant uses 4 pawns (two teams of two or free-for-all).
- Walls: 20 fence pieces in a standard 2-player game (10 per player). Each wall occupies the gap between two adjacent pairs of squares and spans two such gaps (i.e., length 2). Walls can be placed either horizontally or vertically.

---

## 3. Setup

- Place the board so that each player starts centered on their home edge.
- For a 2-player game: White (or Player 1) starts on square (5,1) and Black (or Player 2) starts on square (5,9). Coordinates here use (column,row) with (1,1) at one corner; implementations must adopt a consistent coordinate mapping.
- Each player begins with 10 walls.

---

## 4. Objective

The goal is to be the first player to move your pawn to any square on the opposite side (the row occupied initially by your opponent). In a 2-player game, this means reaching the opponent's home row.

---

## 5. Turn Sequence

On a player's turn they must do exactly one of the following:

- Move their pawn according to the movement rules.
- Place one wall according to the wall placement rules (consuming one of their wall pieces).

Players may not pass. If a player has no legal pawn moves but can place a wall, placing a wall is allowed only if it is a legal placement.

---

## 6. Pawn Movement

- Basic move: A pawn may move one square orthogonally (up, down, left, right) onto an adjacent empty square when no wall blocks the path.
- Jumping over a pawn: If an opponent's pawn is on an adjacent square and the square immediately beyond it (in the same direction) is empty and not blocked by a wall, the player may jump over that pawn to the square directly behind it (a two-square straight move).
- Diagonal move when blocked: If an adjacent enemy pawn blocks a straight jump because a wall or the board edge blocks the square beyond, the moving pawn may move diagonally to one of the two squares that are adjacent to the blocking pawn. A diagonal move is permitted only when the adjacent pawn prevents a straight jump.

Additional constraints:

- All pawn moves are constrained by walls. If a wall lies on the boundary between two squares, movement across that boundary is not allowed.
- Pawns may not share the same square. A move must land on an empty square.

---

## 7. Wall Placement

- Walls occupy the gaps between squares and always cover two adjacent gaps (spanning two edges). They come in horizontal and vertical orientations.
- Walls must be placed aligned to the grid, covering exactly two adjacent edges between squares.
- Walls may not overlap or cross existing walls. A wall placement that overlaps or creates an invalid crossing with an existing wall is illegal.
- Walls must not completely block all possible paths for any player to reach their goal row. After placing a wall, the game must still allow at least one path (of any length) for every pawn to reach its goal row.
- Placing a wall consumes one of the player's wall pieces and ends their turn.

Implementations must validate that a proposed wall placement preserves at least one valid path for every pawn to reach its goal.

---

## 8. Illegal Moves

The following actions are illegal:

- Moving through a wall.
- Moving onto an occupied square.
- Placing a wall that overlaps or crosses an existing wall.
- Placing a wall that leaves any player without any path to their goal.
- Attempting to pass.
- Attempting to perform both a pawn move and a wall placement in the same turn.

If an illegal move is detected by an engine or referee, the move must be rejected and the player must be required to submit a legal move instead.

---

## 9. Winning and End of Game

- The first player to move their pawn onto any square of their target goal row (opponent's starting row) wins immediately.
- In multiplayer variants where teams exist, a team wins when one of its players reaches the opposing edge according to the variant's win rules.

---

## 10. Notation and Coordinate Conventions

Implementations must document the coordinate system used (e.g., (file,rank) or (x,y), zero- or one-indexed) and whether rows increase upward or downward. A consistent mapping must be used across all interfaces.

---

## 11. Examples (Illustrative Only)

The following ASCII diagrams illustrate conceptual positions. These examples are non-normative and provided for clarification only.

`.` indicates empty square.  
`P` indicates a pawn.  
`|` and `-` represent walls graphically for illustration.

Example: Basic move

	. . . . . . . . .
	. . . . . . . . .
	. . . . . . . . .
	. . P . . . . . .

Example: Jump over pawn

	. . . . .
	. P P . .
	. . . . .

Example: Wall placement (conceptual)

	. . . . .
	. . . . .
	. . - - .
	. . . . .

---

## 12. Common Variants

- 4-player Free-for-All: 4 pawns begin on each side center, each player has 5 walls (or other agreed counts). Winning conditions and wall counts vary by variant.
- Team Play (2 vs 2): Two teams of two, players sit opposite each other; a team wins when any team member reaches the opponents' side.
- Smaller or larger boards or different wall counts may be used, provided that core movement and wall rules are preserved.

Unless explicitly adopted by the system, variants are considered non-normative extensions.

---

## 13. Non-Normative Implementation Guidance

The following guidance does not constitute additional rules but may assist implementers:

- Path-checking: When validating wall placement, a pathfinding algorithm (e.g., BFS or A*) may be used to verify that at least one route to the goal remains for each pawn.
- Deterministic turns: Engines should enforce exactly one action per turn and validate legality before committing moves.
- Time controls and penalties: If timed play is supported, the handling of repeated illegal actions should be defined at the protocol level.

---

# Changelog

Version 1 (2026-02-17)
- Converted rule document into Design category with subtype External Domain Specification.
- Added governance-compliant header.
- Clarified normative vs non-normative sections.
