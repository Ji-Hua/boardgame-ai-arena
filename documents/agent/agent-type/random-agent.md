# Random Agent Design

Author: Ji Hua  
Created Date: 2026-04-07  
Last Modified: 2026-04-07  
Current Version: 1  

Document Type: Implementation  
Document Subtype: Agent Strategy  
Document Status: In Development  
Document Authority Scope: Agent module  

Document Purpose:  
This document defines the Random Agent implementations. It covers both the basic Random Agent (V1) and the weighted Random Agent V2, which samples from the full action space including wall placements.

---

# 1. Overview

The Random Agent family provides baseline, non-strategic agents that select actions without evaluation.

Two variants exist:

- **Random Agent (V1)**: Uniformly samples from the legal pawn actions provided by the Backend.
- **Random Agent V2**: Samples from the full action space (pawn moves and wall placements) using a configurable pawn bias.

Both agents are stateless: each decision depends only on the current input.

---

# 2. Random Agent (V1)

## 2.1 Behavior

- Receives `legal_actions` from Backend (pawn moves only)
- Uniformly selects one action at random
- Returns the selected action

## 2.2 Properties

- Stateless
- Non-deterministic (random selection)
- Does not use the Rule Engine
- Does not place walls
- No configuration required

## 2.3 Registration

- type_id: `random`
- category: `scripted`

---

# 3. Random Agent V2

## 3.1 Behavior

- Reconstructs the current engine state from `game_state` (including wall bitmaps)
- Uses the local Rust Rule Engine to enumerate all legal actions (pawn moves and wall placements)
- Partitions actions into pawn moves and wall placements
- Applies weighted random selection:
  - With probability `threshold` (default: 0.8), selects a pawn move
  - With probability `1 - threshold`, selects a wall placement
- Falls back to whatever category is available when one is empty

## 3.2 Engine Usage

V2 uses the same Rust Rule Engine as the Backend, locally:

- `RuleEngine.legal_actions(state)` — enumerates all legal actions for the current player

This guarantees the agent only produces actions the Engine will accept.

Invariant:

For any state:

AgentEngine.legal_actions(state) ≡ BackendEngine.legal_actions(state)

## 3.3 Properties

- Stateless: state is reconstructed from `game_state` each call
- Non-deterministic (random with configurable bias)
- All produced actions are valid (derived from engine legal action set)
- Configurable pawn/wall ratio via threshold parameter

## 3.4 Registration

- type_id: `random_v2`
- category: `scripted`

---

# 4. Constraints

Both Random Agents MUST:

- Return exactly one action per request
- Treat GameState as read-only
- Not call Backend or Engine remotely

Random Agent V2 additionally MUST:

- Use only the local Rule Engine for action enumeration
- Not evaluate or rank actions

---

# 5. Limitations

- No strategic reasoning
- No opponent modeling
- V1 cannot place walls (limited to Backend-provided pawn moves)
- V2 wall placement selection is uniformly random among legal walls
- Games between random agents may take many turns to complete

---

# 6. Extensibility

The Random Agent V2 design demonstrates the pattern for agents that generate wall actions locally using the Rust Rule Engine. This pattern is reusable by any agent that needs the full action space beyond what the Backend provides in `legal_actions`.

---

# Changelog

Version 1 (2026-04-07)
- Initial Random Agent documentation
- Defined V1 and V2 behavior
- Documented V2 engine usage and weighted sampling
- Established constraints and limitations
