# Agent Service Design

Author: Ji Hua  
Created Date: 2026-04-05  
Last Modified: 2026-04-05  
Current Version: 1  
Document Type: Design  
Document Subtype: Agent Service Architecture  
Document Status: In Development  
Document Authority Scope: Agent module  
Document Purpose:  
This document defines the structural design of the Agent Service as a standalone service within the Quoridor system. It specifies the service's role, internal structure, communication model, and authority boundaries. It does not define API schemas, implementation details, or framework choices.

---

# 1. Overview

The Agent Service is a standalone service that provides automated decision-making capabilities for all non-human players in the Quoridor system.

It hosts and manages multiple agent implementations, providing a unified interface for the Backend to interact with any type of automated player.

The Agent Service is a pure decision provider. It does not participate in rule evaluation, state authority, or game orchestration.

---

# 2. Role

The Agent Service serves the following role within the Live Game System:

- **Decision Provider** — Given a GameState, produce an Action
- **Agent Host** — Host multiple agent implementations within a single service
- **Lifecycle Recipient** — Respond to Backend commands for agent creation, start, stop, and teardown

The Agent Service does not:

- Own or mutate GameState
- Validate rules
- Interact with the Engine directly
- Initiate gameplay actions independently
- Manage game lifecycle

---

# 3. Relationship with Backend

The Agent Service operates under full Backend control.

## 3.1 Backend as Orchestrator

The Backend is the sole orchestrator of agent lifecycle and gameplay flow:

- Backend creates agents (via Agent Service)
- Backend starts agents when a game begins
- Backend delivers GameState to agents when it is their turn
- Backend receives Actions from agents
- Backend stops agents when a game ends

The Agent Service never initiates communication with the Backend for gameplay purposes.

## 3.2 Communication Flow

Control Plane (lifecycle):
```
Frontend → Backend → Agent Service
```

Gameplay Plane (decision):
```
Backend → Agent Service → Backend → Engine → Backend → Frontend
```

The Agent Service participates in both planes but is always responding to Backend requests.

---

# 4. Internal Structure

The Agent Service hosts multiple agent implementations behind a unified service boundary.

```
Agent Service
  ├── Agent Registry (maps type_id → agent factory)
  ├── Agent Instance Manager (tracks active agent instances)
  │
  ├── Agent Implementations
  │   ├── Random Agent
  │   ├── Greedy Agent
  │   ├── Dummy Agent
  │   ├── Replay Agent
  │   └── (Future agents)
  │
  └── Service Interface (receives Backend requests)
```

## 4.1 Agent Registry

The Agent Registry maintains a mapping of agent type identifiers to agent factories.

Responsibilities:
- Register agent types at service startup
- Provide agent type listing to Backend on request
- Instantiate agents by type when requested

## 4.2 Agent Instance Manager

The Agent Instance Manager tracks active agent instances across rooms and games.

Responsibilities:
- Create agent instances on Backend request
- Track instance-to-room/seat binding
- Route GameState delivery to the correct instance
- Tear down instances when games end

## 4.3 Agent Implementations

Each agent implementation follows a common abstract interface:

- Receive GameState as input
- Produce a single Action as output
- Optionally maintain internal state (e.g., search trees, replay position)
- Provide a reset mechanism for new games

The internal logic of each agent is opaque to the Agent Service framework and to the Backend.

---

# 5. Communication Planes

## 5.1 Control Plane

The Control Plane manages agent lifecycle and configuration.

Operations:
- **Create** — Instantiate an agent of a given type for a specific room/seat
- **Configure** — Provide agent-specific parameters (e.g., replay data, difficulty settings)
- **Start** — Activate the agent for gameplay
- **Stop** — Deactivate the agent
- **Destroy** — Tear down the agent instance
- **List Types** — Return available agent types

Control flow:
```
User → Frontend → Backend → Agent Service
```

The Agent Service must not independently manage its lifecycle. All lifecycle transitions are initiated by the Backend.

## 5.2 Gameplay Plane

The Gameplay Plane handles decision-making during active games.

Execution cycle:
1. Backend determines it is an agent-controlled seat's turn
2. Backend sends GameState to Agent Service (identifying which agent instance)
3. Agent Service routes GameState to the correct agent instance
4. Agent instance produces an Action
5. Agent Service returns Action to Backend
6. Backend routes Action through Engine (same path as human actions)
7. Backend broadcasts state_update to all subscribers

The agent's action submission follows the same Gameplay API contract used by human players. The Backend does not distinguish between human and agent actions at the Engine layer.

---

# 6. Authority Boundaries

The Agent Service operates under strict authority constraints:

| Concern | Authority |
|---------|-----------|
| Game rules | Engine (exclusive) |
| Game state | Engine (via Backend) |
| Game orchestration | Backend (exclusive) |
| Agent lifecycle | Backend (via Agent Service) |
| Decision-making | Agent Service (exclusive) |

Constraints:
- Agent Service cannot modify GameState
- Agent Service cannot bypass Backend
- Agent Service cannot interact with Engine directly
- Agent Service cannot initiate gameplay actions
- Agent Service cannot refuse Backend lifecycle commands

---

# 7. Agent-Human Equivalence

At the Gameplay Plane, agents and human players are treated identically:

- Both produce Actions in response to GameState
- Both submit Actions through the same Backend pathway
- Both are subject to the same Engine validation
- The difference is only in how decisions are generated

This equivalence is a system-level invariant. The Backend must not apply different gameplay rules based on whether a seat is human or agent.

---

# 8. Replay as Agent

Replay functionality is implemented as a specialized Replay Agent within the Agent Service.

Replay Agent characteristics:
- Receives a replay action sequence during configuration
- On each GameState delivery, emits the next predetermined action from the sequence
- Is deterministic — always produces the same action for the same replay position
- Follows the same Backend → Agent → Backend → Engine path as all other agents

Replay must not:
- Execute on the Frontend
- Bypass Backend validation
- Directly interact with the Engine

This ensures replay correctness is validated by the same authority chain as live play.

---

# 9. Deployment Model

The Agent Service is designed as an independent, deployable service.

Characteristics:
- Runs as a separate process from the Backend
- Communicates with Backend over network (protocol TBD)
- Can be scaled independently
- Can be developed and tested independently

The exact deployment topology (co-located, containerized, etc.) is not defined by this document.

---

# 10. Scope and Non-Scope

This document defines:
- The role and responsibilities of the Agent Service
- Its internal structural model
- Its communication model with the Backend
- Its authority boundaries
- The Replay Agent as a specialized agent type

This document does not define:
- API schemas or protocol details
- Framework or language choices
- Internal algorithm implementations
- Training system integration
- Performance requirements

---

# Changelog

Version 1 (2026-04-05)
- Initial Agent Service design document
- Defined service role, internal structure, communication planes
- Established authority boundaries and agent-human equivalence
- Defined Replay as Agent model
