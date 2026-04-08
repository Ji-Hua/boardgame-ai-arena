# Agent Registration and Migration Design

Author: Ji Hua  
Created Date: 2026-04-05  
Last Modified: 2026-04-05  
Current Version: 1  
Document Type: Design  
Document Subtype: Agent Registration and Migration  
Document Status: In Development  
Document Authority Scope: Agent module  
Document Purpose:  
This document defines the agent registration mechanism for the Agent Service, the migration plan for legacy agents, and the redesign of Replay as a Replay Agent. It establishes how agents are identified, registered, and extended, and how existing agent code is adapted for the new architecture.

---

# 1. Agent Registration Mechanism

## 1.1 Overview

The Agent Service maintains a registry of available agent types. Each agent type is identified by a unique type identifier and associated with a factory that can instantiate agent instances.

Registration occurs at Agent Service startup. The registry is immutable at runtime — new agent types require a service restart (or a future hot-reload mechanism).

## 1.2 Agent Type Identification

Each registered agent type has the following identity attributes:

| Attribute | Description |
|-----------|-------------|
| `type_id` | Unique string identifier (e.g., `"random"`, `"greedy"`, `"replay"`) |
| `display_name` | Human-readable name (e.g., `"Random Agent"`) |
| `category` | Classification: `"ai"`, `"replay"`, or `"scripted"` |

Categories:
- **ai** — Agents that make decisions based on heuristics, search, or learned models (e.g., `greedy`, future ML agents)
- **replay** — Agents that emit predetermined actions from a recorded game (e.g., `replay`)
- **scripted** — Agents that follow fixed behavioral rules without evaluation (e.g., `random`, `dummy`)

## 1.3 Registration Model

Agent types are registered via a declarative registry pattern:

```
Registry
  "random"   → RandomAgent factory    [category: scripted]
  "dummy"    → DummyAgent factory     [category: scripted]
  "greedy"   → GreedyAgent factory    [category: ai]
  "replay"   → ReplayAgent factory    [category: replay]
```

Registration requirements:
- Each agent type must provide a factory that accepts optional configuration and returns an agent instance
- Each agent instance must implement the common agent interface (receive GameState, produce Action)
- Each agent type must declare its category

## 1.4 Extensibility

New agent types can be added by:
1. Implementing the common agent interface
2. Registering the type in the Agent Registry with a unique `type_id`
3. Restarting the Agent Service

No changes to the Backend or Frontend are required to support new agent types, as long as they conform to the existing interface contract.

The Backend discovers available types via the `List Types` operation (Agent Control API).

---

# 2. Legacy Agent Audit

## 2.1 Current Legacy Code

The legacy agent code resides at `quoridor_v0/quoridor-agents` and contains:

### Base Agent (`agents/base_agent.py`)
Abstract base class defining the agent interface:
- `make_action(state, legal_actions) → Action` — Core decision method
- `update_state(action, game_state)` — State update callback
- `reset()` — Optional reset between games
- `notify_result(result)` — Optional game result notification
- Constructor: `name`, `model_version`

### Agent Implementations

| Agent | File | Category | Behavior |
|-------|------|----------|----------|
| DummyAgent | `dummy_agent.py` | scripted | Always picks first pawn action |
| RandomAgent | `random_agent.py` | scripted | Random action selection |
| RandomAgentV2 | `random_agent.py` | scripted | Weighted random (80% pawn, 20% wall) |
| GreedyAgent | `greedy_agent.py` | ai | Placeholder heuristic evaluation |

### Agent Registry (`registry.py`)
Simple class-based registry:
- Maps string keys to agent classes
- `get(name)` instantiates and returns an agent
- `list()` returns available agent names
- Currently registers: `dummy`, `random`, `random_v2`

### Dependencies
- `quoridor_utils.models` — GameState and Action models (external)
- `quoridor_rust_engine` — Used by GreedyAgent for evaluation (direct Engine access)

---

# 3. Migration Plan

## 3.1 What Can Be Reused

| Component | Reusable | Notes |
|-----------|----------|-------|
| Agent base class interface | Yes (with adaptation) | `make_action` pattern maps directly to new service model |
| DummyAgent logic | Yes | Trivial decision logic, no external dependencies |
| RandomAgent logic | Yes | Trivial decision logic, no external dependencies |
| RandomAgentV2 logic | Yes | Trivial decision logic, no external dependencies |
| GreedyAgent logic | Partially | Decision logic reusable; direct Engine dependency must be removed |
| Registry pattern | Yes (with adaptation) | Declarative pattern maps to new registry; needs category support |

## 3.2 What Must Change

### 3.2.1 Agent Base Class

Current:
```python
make_action(self, state: GameState, legal_actions: list[Action]) → Action
update_state(self, action: Action, game_state: GameState)
```

Required changes:
- `make_action` signature remains conceptually the same but inputs will come from Backend-provided GameState (serialized JSON), not internal Python objects
- `update_state` may be removed or redesigned — agents receive full GameState on each turn, so incremental updates are optional
- Add `configure(config: dict)` method for agent-specific setup (e.g., replay data)
- Add `type_id` and `category` class attributes for registry integration

### 3.2.2 GreedyAgent — Remove Engine Dependency

The legacy GreedyAgent directly imports `quoridor_rust_engine.Engine` for action evaluation. This violates the architectural constraint that agents must not interact with the Engine directly.

Migration path:
- Remove direct Engine import
- If local simulation is needed, provide a lightweight evaluation utility within the Agent Service (non-authoritative)
- Alternatively, redesign to use only the GameState information provided by the Backend

### 3.2.3 Model Dependencies

Legacy agents depend on `quoridor_utils.models.GameState` and `Action`. In the new architecture:
- GameState and Action will be received as serialized JSON from the Backend
- Agent Service must define its own internal models or use a shared schema package
- Models must match the Backend's serialization format (defined in the Backend Interface document)

### 3.2.4 Registry Enhancement

The legacy registry is a simple dict mapping. Required enhancements:
- Add `category` attribute per agent type
- Add `display_name` attribute per agent type
- Support `list_types()` returning structured type information
- Support `create(type_id, config)` for parameterized instantiation

## 3.3 Migration Steps

1. **Copy agent implementations** from `quoridor_v0/quoridor-agents` into monorepo Agent Service
2. **Adapt base class** — update interface to match new service model
3. **Update models** — switch from `quoridor_utils` imports to Agent Service internal models
4. **Remove Engine dependency** from GreedyAgent
5. **Enhance registry** — add category, display_name, structured listing
6. **Add Replay Agent** — new agent type (see Section 4)
7. **Integrate with Agent Service** — wrap agents behind service interface
8. **Test** — verify each agent produces valid actions given serialized GameState input

---

# 4. Replay as Agent

## 4.1 Current State

Replay is currently implemented in the Frontend:
- Frontend loads replay data and drives playback locally
- Replay does not go through Backend → Engine validation
- This bypasses the authoritative execution chain

## 4.2 Target Design

Replay must be redesigned as a **Replay Agent** within the Agent Service.

The Replay Agent:
- Is a standard agent that follows the same lifecycle as all other agents
- Is configured with a replay action sequence at creation time
- On each turn, emits the next action from the replay sequence
- Is deterministic — the same replay data always produces the same action sequence
- Follows the standard data flow: Backend → Agent Service → Backend → Engine

## 4.3 Replay Agent Behavior

```
Configuration:
  - Receive replay data (ordered list of actions)
  - Initialize replay cursor to position 0

Each turn:
  - Receive GameState from Backend
  - Emit action at current cursor position
  - Advance cursor
  - If cursor exceeds replay length: signal completion (no more actions)
```

## 4.4 Replay Data Flow

```
1. User selects "replay" mode in Frontend
2. Frontend sends replay data to Backend via Agent Control API
3. Backend creates Replay Agent in Agent Service with replay data
4. Game starts with both seats as Replay Agents (or one seat)
5. Backend delivers GameState to Replay Agent on each turn
6. Replay Agent emits the next recorded action
7. Backend routes action through Engine for validation
8. Backend broadcasts state_update to Frontend
9. Frontend renders the replayed game in real-time
```

## 4.5 Benefits

- Replay actions are validated by the Engine (ensures correctness)
- Frontend receives state_update through the standard broadcast mechanism
- Replay uses the same rendering path as live games
- No special-case code in Frontend for replay
- Replay speed can be controlled by Backend pacing (future enhancement)

## 4.6 Frontend Replay Transition

The existing frontend-side replay remains functional for now. The transition to Replay Agent is:

1. **Phase 1 (current):** Frontend-driven replay continues to work
2. **Phase 2:** Backend + Replay Agent path implemented and tested
3. **Phase 3:** Frontend switches to Backend-driven replay; local replay code deprecated

This is a non-breaking migration — both paths can coexist during transition.

---

# Changelog

Version 1 (2026-04-05)
- Initial agent registration and migration design
- Defined registration mechanism with type_id, display_name, category
- Audited legacy agent code and identified reusable components
- Defined migration steps for each agent and the registry
- Designed Replay Agent architecture and data flow
- Defined phased frontend replay transition plan
