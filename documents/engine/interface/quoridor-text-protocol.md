# Quoridor Text Protocol (QTP)

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 1

Document Type: Interface
Document Subtype: Text Protocol
Document Status: In Development
Document Authority Scope: Protocol layer
Document Purpose:
This document defines the Quoridor Text Protocol (QTP), a human-readable
and replay-friendly command protocol for representing Quoridor games.
QTP defines command structure, grammar, and replay format.
It relies on the Display Coordinate System defined in the Design layer.
QTP does not define geometric semantics or engine internals.

---

## 1. Overview

QTP (Quoridor Text Protocol) is a lightweight textual protocol for:

- Game replay files
- CLI-based play
- Engine debugging
- Agent communication
- Human-readable logs

QTP is:

- Human-readable
- Backend-agnostic
- Stable across engine implementations
- Independent of internal logical coordinate models

QTP uses the Display Coordinate System defined in:

`documents/engine/design/cooridination.md`

---

## 2. Coordinate Layer

QTP uses Glendenning-style board notation:

- Files: A–I
- Ranks: 1–9
- A1 is bottom-left
- I9 is top-right

Pawn squares:
- E2
- D5
- etc.

Wall anchors:
- E3h (horizontal wall)
- E3v (vertical wall)

The mapping from display notation to logical coordinates is defined in the Design layer.
QTP does not define internal coordinate conversion rules.

---

## 3. Command Model

Each command represents exactly one legal game action.

Command syntax:

<Command> <Args...>

Commands are case-sensitive.

---

## 4. Command Set

### 4.1 Pawn Move

Command:

P <player> <square>

Example:

P 1 E2
P 2 E8

Meaning:

- Move pawn of <player> to <square>
- <square> must follow Display Coordinate syntax

---

### 4.2 Wall Placement

Horizontal Wall:

W <player> <square> h

Vertical Wall:

W <player> <square> v

Examples:

W 1 E3 h
W 2 D5 v

Meaning:

- Place a wall with head at <square>
- Orientation specified by final token
- Wall legality must be validated by the engine

---

## 5. Grammar

command_line:
  format: "<Command> <Args...>"

command_set:
  - P
  - W

allowed_players:
  - 1
  - 2

square_syntax:
  format: "<File><Rank>"
  file_range: "A–I"
  rank_range: "1–9"

wall_orientation:
  allowed:
    - h
    - v

---

## 6. Replay Format

A replay file is a sequence of QTP commands.

- One command per line
- Strictly ordered
- Represents exactly one complete game

Example:

P 1 E2
P 2 E8
W 1 D4 v
W 2 E6 h
P 1 E3
P 2 E7

---

## 7. Metadata (Optional)

Metadata may be included using comment lines beginning with "#".

Example:

# Game-ID: demo-1234
# Date: 2026-02-17
# Players: Human vs Engine-v1
# Result: P1-win
# Termination: goal

Metadata is ignored by the core parser unless explicitly consumed.

---

## 8. Machine Mapping

QTP commands must be converted into backend action structures.

Example mapping (illustrative only):

P p S:
  { player: p, type: "square", target: S }

W p S h:
  { player: p, type: "horizontal", head: S }

W p S v:
  { player: p, type: "vertical", head: S }

Conversion from Display Coordinate to logical grid
is an internal backend concern and not part of QTP.

---

## 9. Compatibility and Versioning

- Minor versions may extend metadata or optional commands.
- Breaking grammar changes require a major version increment.
- Changes to the Display Coordinate System require a major version bump.

QTP is intentionally independent of engine implementation details.
Logical coordinate representation changes must not affect QTP.

---

# Changelog

Version 1 (2026-02-17)
- Replaced QNP with QTP.
- Removed custom coordinate system definition.
- Adopted Display Coordinate System from Design layer.
- Simplified command set.
- Clarified separation from engine internals.
