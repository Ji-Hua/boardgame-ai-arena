# Agent API Capabilities

Author: Ji Hua  
Created Date: 2026-04-05  
Last Modified: 2026-04-05  
Current Version: 1  
Document Type: Interface  
Document Subtype: Agent API Capabilities  
Document Status: In Development  
Document Authority Scope: Agent module  
Document Purpose:  
This document defines the capability-level API contract between the Backend and the Agent Service. It specifies what operations exist, their semantics, and behavioral expectations. It does NOT define transport protocols, request/response schemas, or wire formats.

---

# 1. Overview

The Agent Service exposes two API planes to the Backend:

- **Control Plane** — Agent lifecycle management (create, configure, destroy)
- **Gameplay Plane** — Decision request/response during active games

Both planes are Backend-initiated. The Agent Service never initiates communication.

---

# 2. Agent Lifecycle Capabilities (Control Plane)

## 2.1 Create Agent

Instantiate a new agent of a given type.

- Input: agent type identifier, room/seat binding, optional configuration
- Output: instance identifier
- Semantics: Creates an agent instance and binds it to a specific room and seat. The instance is initially inactive (not yet participating in gameplay).
- Error: Unknown agent type, invalid configuration

## 2.2 Configure Agent

Provide agent-specific parameters after creation.

- Input: instance identifier, configuration data
- Semantics: Updates the agent's internal configuration. For Replay Agents, this provides the action sequence. For AI agents, this may set difficulty or strategy parameters.
- Constraint: May only be called before the agent is started.
- Error: Unknown instance, invalid configuration

## 2.3 Destroy Agent

Tear down an agent instance and release resources.

- Input: instance identifier (or room identifier for bulk destroy)
- Semantics: Permanently removes the agent instance. Any in-progress decision is cancelled.
- Idempotent: Destroying a non-existent instance is a no-op.

---

# 3. Session / Seat Binding

## 3.1 Bind Agent to Game

Association between an agent instance and an active game.

- Binding is established at creation time via room/seat parameters.
- An agent is bound to exactly one room and one seat.
- Multiple agents may be bound to the same room (one per seat).

## 3.2 Bind Agent to Seat

Each agent instance is bound to a specific seat (1 or 2).

- The binding is immutable for the lifetime of the instance.
- The Backend is responsible for ensuring seat/agent type consistency (i.e., the seat's actor_type must be "agent").

## 3.3 Unbind Agent

Agents are unbound when destroyed.

- There is no explicit unbind operation separate from destroy.
- When a game ends, the Backend destroys all agents in the room.

---

# 4. Gameplay Capabilities (Gameplay Plane)

## 4.1 Receive GameState

The Backend delivers the current game state to the agent when it is the agent's turn.

- Input: room/seat identifier, full GameState, list of legal actions
- Semantics: The agent receives the same GameState that would be broadcast to a human player. Legal actions are provided as a convenience but the agent is not required to choose from them (it may attempt any action, including wall placements not in the legal actions list).

## 4.2 Produce Action

The agent returns a single action decision.

- Output: Action in the standard format (player, type, target)
- Semantics: The action represents the agent's decision for the current turn. It is submitted to the Backend, which routes it through the Engine for validation.
- The agent's action is treated identically to a human action at the Engine layer.

## 4.3 Handle Invalid Action Feedback (Reject)

When an agent's action is rejected by the Engine, the system must handle the failure gracefully.

- The Backend does NOT forward rejection details to the Agent Service.
- For standard agents (AI, scripted): a rejected action indicates a bug in the agent. The Backend may force-end the game.
- For Replay Agents: a rejected action is expected behavior (the original game may have contained rejected actions). The Replay Agent must re-emit the same action until it is accepted. The Backend must re-request the action from the Replay Agent without advancing.

---

# 5. Error / Retry Semantics

## 5.1 Agent May Produce Invalid Actions

Agents are not guaranteed to produce valid actions. The Engine remains the sole authority on action validity.

## 5.2 Timeout

The Backend imposes a timeout on agent decision requests. If the agent does not respond in time, the Backend force-ends the game.

## 5.3 Agent Error

If the agent raises an internal error during decision-making, the Backend force-ends the game. The Agent Service must catch and report errors rather than allowing them to propagate as unhandled exceptions.

## 5.4 Replay Agent Retry

The Replay Agent is a special case. When its action is rejected:

1. The Backend detects the rejection.
2. The Backend re-requests action from the same agent instance.
3. The Replay Agent re-emits the same action (cursor does not advance on rejection).
4. This continues until the action is accepted or a maximum retry count is reached.

This enables replays of games that contained invalid actions (e.g., a player attempting an illegal wall placement before making a valid move).

## 5.5 System Tolerance

The overall system must tolerate agent misbehavior without corrupting game state:

- Invalid actions do not modify GameState (Engine guarantee)
- Timeouts result in clean game termination
- Agent crashes do not crash the Backend
- The Gameplay Plane remains consistent regardless of agent behavior

---

# 6. Key Principles

1. **Agent-Human Equivalence**: Agents participate through the same Gameplay path as humans. No special Engine treatment.
2. **Backend Authority**: The Backend controls all agent lifecycle. Agents cannot self-activate or self-terminate.
3. **Engine Authority**: All actions (human or agent) are validated by the Engine. The Agent Service has no rule authority.
4. **Single Contract**: The Gameplay API is shared between Frontend (human) and Agent Service (automated). No duplication.
5. **Fail-Safe**: Agent failures result in clean game termination, never in corrupted or inconsistent state.

---

# Changelog

Version 1 (2026-04-05)
- Initial capability-level API document
- Defined lifecycle, session binding, gameplay, and error semantics
- Established Replay Agent retry behavior for rejected actions
