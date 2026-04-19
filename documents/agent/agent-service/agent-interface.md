# Agent Interface Contract

Author: Ji Hua
Created Date: 2026-04-07
Last Modified: 2026-04-07
Current Version: 1
Document Type: Interface
Document Subtype: Backend ↔ Agent Service Contract
Document Status: In Development
Document Authority Scope: Agent module, Backend module
Document Purpose:
This document defines the formal interface contract between the Backend and the Agent Service. It specifies what each side provides, what each side expects, and the precise semantics of all interaction points. It does not define transport protocols, wire formats, or internal implementation details.

---

# 1. Scope and Authority

## 1.1 What This Contract Covers

This document covers:

- **Control Plane**: The lifecycle operations the Backend executes against the Agent Service (create, configure, destroy)
- **Gameplay Plane**: The decision loop executed during an active game (state delivery → action return)
- **Responsibility boundaries**: What the Backend owns vs. what the Agent Service owns

## 1.2 What This Contract Does Not Cover

- Transport protocols or wire formats (HTTP, WebSocket, etc.)
- Internal agent algorithm implementations
- Frontend↔Backend communication
- Engine↔Backend communication
- Training system integration

## 1.3 Authority Invariants

These invariants are non-negotiable and are enforced by the contract structure:

| Concern | Owner |
|---|---|
| Game rules and action validity | Engine (exclusive) |
| GameState (authoritative) | Engine via Backend |
| Agent lifecycle | Backend (exclusive) |
| Action selection | Agent Service (exclusive) |

The Agent Service MUST NOT:
- Interact with the Engine directly
- Modify or store authoritative GameState
- Initiate any gameplay or lifecycle action toward the Backend
- Refuse or delay a lifecycle command from the Backend

---

# 2. Shared Data Types

These types are used in both planes and are defined here to be referenced throughout.

## 2.1 AgentTypeDescriptor

Describes a registered agent type. Produced by the Agent Service; consumed by the Backend.

```
AgentTypeDescriptor:
  type_id:       string            -- Unique identifier (e.g., "random", "greedy", "replay")
  display_name:  string            -- Human-readable label
  category:      "ai"              -- Heuristic, search, or learned strategy
               | "scripted"        -- Fixed behavioral rules (e.g., random, dummy)
               | "replay"          -- Emits predetermined action sequences
```

## 2.2 AgentInstanceId

An opaque, service-assigned identifier for a live agent instance.

```
AgentInstanceId: string (opaque)
```

The Backend must treat this as an opaque handle. The Agent Service may use any scheme internally.

## 2.3 SeatBinding

Identifies which game seat an agent is bound to.

```
SeatBinding:
  room_id:  string    -- Room identifier (assigned by Backend)
  seat:     1 | 2     -- Seat number
```

A binding is immutable for the lifetime of an instance. One agent instance maps to exactly one seat in one room.

## 2.4 GameState

The authoritative game state delivered by the Backend to the Agent Service. The Backend serializes this from engine state; the Agent Service must not assume additional fields.

```
GameState:
  current_player:   1 | 2
  pawns:
    "1": { row: integer[0–8], col: integer[0–8] }
    "2": { row: integer[0–8], col: integer[0–8] }
  walls_remaining:
    "1": integer[0–10]
    "2": integer[0–10]
  game_over:  boolean
  winner:     1 | 2 | null
```

The Agent Service treats GameState as read-only input. It does not store, modify, or forward it.

## 2.5 Action

The action produced by the Agent Service and returned to the Backend.

```
Action:
  player:  1 | 2
  type:    "pawn"        -- Pawn movement
         | "horizontal"  -- Horizontal wall placement
         | "vertical"    -- Vertical wall placement
  target:  [row, col]    -- Logical coordinates
```

- For pawn moves: row, col ∈ [0, 8]
- For wall placements: row, col ∈ [0, 7]

The Agent Service does not validate actions. All validation is performed by the Engine upon submission through the Backend.

## 2.6 AgentConfiguration

Arbitrary agent-specific parameters provided at creation or configuration time.

```
AgentConfiguration: map<string, any>
```

The Backend passes this opaquely. The Agent Service interprets it. Well-known keys:

| Key | Agent Type | Meaning |
|---|---|---|
| `actions` | replay | Ordered list of Actions forming the replay sequence |

Future agent types may define additional keys. The Backend does not interpret or validate configuration content.

---

# 3. Control Plane

The Control Plane manages agent lifecycle. All operations are Backend-initiated. The Agent Service never initiates control operations.

## 3.1 Lifecycle State Machine

Each agent instance progresses through the following states:

```
CREATED → ACTIVE → STOPPED
                       ↓
                   DESTROYED (from any state)
```

- **CREATED**: Instance exists, not yet participating in gameplay
- **ACTIVE**: Instance is running; may receive GameState and produce Actions
- **STOPPED**: Instance paused; gameplay requests rejected
- **DESTROYED**: Instance released; no further operations may be called on it

## 3.2 ListTypes

Query the available agent types registered in the Agent Service.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input: _(none)_

Output:
```
agent_types: list<AgentTypeDescriptor>
```

**Semantics:**
- Returns the full registry of available agent types at the time of the call
- The registry is static at runtime (changes require service restart)
- The Backend uses this to populate available agent choices

---

## 3.3 CreateAgent

Instantiate a new agent of a given type, bound to a specific seat.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
agent_type:  string              -- Must match a registered type_id
binding:     SeatBinding         -- Room and seat the agent will occupy
config:      AgentConfiguration  -- Optional; may be empty
```

Output:
```
instance_id: AgentInstanceId
```

**Semantics:**
- Creates an agent instance in CREATED state
- The instance is bound to the given room/seat; this binding is immutable
- `config` is applied immediately if provided; it may also be applied later via ConfigureAgent
- Returns an opaque instance identifier the Backend uses in all subsequent operations

**Errors:**
- Unknown `agent_type`
- Seat already occupied in the given room

---

## 3.4 ConfigureAgent

Supply or update agent-specific parameters.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
instance_id:  AgentInstanceId
config:       AgentConfiguration
```

Output: _(acknowledgement)_

**Semantics:**
- MUST only be called when the instance is in CREATED state
- For Replay Agents: `config.actions` provides the full action sequence; MUST be set before the instance is started
- For other agent types: config is optional and agent-defined

**Errors:**
- Unknown `instance_id`
- Instance not in CREATED state

---

## 3.5 StartAgent

Activate an agent instance for gameplay.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
instance_id: AgentInstanceId
```

Output: _(acknowledgement)_

**Semantics:**
- Transitions the instance from CREATED → ACTIVE
- After this call, the instance is eligible to receive GameState and produce Actions
- Idempotent if instance is already ACTIVE

**Errors:**
- Unknown `instance_id`
- Instance not in a startable state (e.g., already DESTROYED)

---

## 3.6 StopAgent

Deactivate an agent instance.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
instance_id: AgentInstanceId
```

Output: _(acknowledgement)_

**Semantics:**
- Transitions ACTIVE → STOPPED
- After this call, Gameplay Plane requests will be rejected
- Idempotent if already STOPPED or CREATED

**Errors:**
- Unknown `instance_id`

---

## 3.7 DestroyAgent

Permanently release an agent instance.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
instance_id: AgentInstanceId
```

Output: _(acknowledgement)_

**Semantics:**
- Releases all resources for the instance
- Idempotent: destroying an unknown or already-destroyed instance is a no-op
- Any in-progress gameplay request is cancelled
- After destruction, no operations may be called on this `instance_id`

---

## 3.8 Backend Lifecycle Obligations

The Backend MUST:
- Call DestroyAgent for all instances when a game ends (including force-ended games)
- Not call StartAgent before ConfigureAgent for Replay Agents
- Not send gameplay requests to STOPPED or DESTROYED instances
- Tolerate DestroyAgent being called on already-destroyed instances

---

# 4. Gameplay Plane

The Gameplay Plane is the core decision loop. It executes during an active game whenever it is an agent seat's turn.

## 4.1 Execution Cycle (Normative)

The following is the authoritative execution cycle for each agent turn:

```
1. Backend determines it is an agent-controlled seat's turn
2. Backend calls RequestAction on the Agent Service
3. Agent Service routes the request to the bound agent instance
4. Agent instance produces an Action
5. Agent Service returns Action to Backend
6. Backend submits Action to Engine for validation
7a. Engine ACCEPTS → Backend broadcasts state_update; turn advances
7b. Engine REJECTS → Backend applies rejection policy (see Section 4.4)
```

The Agent Service is not notified of acceptance or rejection directly. The Backend is the sole mediator.

## 4.2 RequestAction

Request an action decision from an agent for a given turn.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
room_id:       string
seat:          1 | 2
game_state:    GameState         -- Current authoritative state
legal_actions: list<Action>      -- Legal pawn moves for the current player
```

Output:
```
action: Action
```

**Semantics:**
- `game_state` is the authoritative state at the time of the request; the agent receives it read-only
- `legal_actions` contains all legal pawn moves for `current_player`; it is provided as a convenience
  - The agent MAY choose from this list for pawn moves
  - The agent MAY attempt wall placements outside this list
  - Wall placement legality is only determined by the Engine upon submission
- The agent MUST return exactly one Action
- The Agent Service must route the request to the instance bound to `(room_id, seat)`
- The instance MUST be in ACTIVE state; requests to STOPPED or non-existent instances are errors

**Constraints on the Agent:**
- The agent MUST NOT call any external system (Engine, Backend, Frontend)
- The agent MUST NOT modify `game_state`
- The action's `player` field MUST equal `seat`

**Errors:**
- No active instance for `(room_id, seat)`
- Instance not in ACTIVE state
- Agent internal error during decision

---

## 4.3 AdvanceCursor (Replay Agent only)

Signal to a Replay Agent that its last-emitted action was accepted by the Engine.

**Caller:** Backend  
**Direction:** Backend → Agent Service

Input:
```
room_id: string
seat:    1 | 2
```

Output: _(acknowledgement)_

**Semantics:**
- Advances the Replay Agent's internal cursor to the next action
- MUST be called by the Backend after a successful Engine ACCEPT for a replay-managed seat
- MUST NOT be called on non-replay agent types (no-op for them, but callers should avoid it)
- If called before the first RequestAction, the cursor advances prematurely — caller error

**Rationale:**
The Replay Agent does not advance its cursor inside RequestAction. This separates the act of producing an action from confirming it was accepted. On Engine rejection, the Backend re-calls RequestAction without calling AdvanceCursor; the Replay Agent re-emits the same action.

---

## 4.4 Rejection Policy

When the Engine rejects an agent's action:

### Standard Agents (category: "ai" or "scripted")

A rejected action indicates a defect in the agent's decision logic. The Backend MUST force-end the game. No retry is performed.

### Replay Agents (category: "replay")

A rejection is expected behavior. The original recorded game may contain actions that were also rejected. The Backend MUST:

1. NOT call AdvanceCursor
2. Re-call RequestAction on the same instance with the same GameState
3. The Replay Agent re-emits the same action (cursor has not advanced)
4. Repeat until the Engine accepts or the maximum retry limit is reached
5. If the retry limit is exhausted, force-end the game

The Backend defines the maximum retry count. The Agent Service does not participate in retry counting.

---

## 4.5 Agent Service Obligations in Gameplay

The Agent Service MUST:
- Return an action within the Backend's timeout window
- Catch all internal agent errors and surface them as service-level errors (not unhandled crashes)
- Never return more than one action per RequestAction call
- Route RequestAction to the correct instance by `(room_id, seat)`

The Agent Service MUST NOT:
- Cache or retain GameState beyond the duration of a single RequestAction call
- Submit actions to the Backend directly
- Interact with the Engine
- Advance the Replay cursor without an explicit AdvanceCursor call

---

# 5. Error and Timeout Semantics

## 5.1 Agent Internal Error

If the agent raises an error during RequestAction:

- The Agent Service catches the error and returns a service-level error response
- The Backend treats this as a terminal failure and force-ends the game
- The Backend then calls DestroyAgent on the affected instance

## 5.2 Timeout

The Backend enforces a decision timeout on each RequestAction call.

- If the Agent Service does not respond within the timeout, the Backend force-ends the game
- The timeout policy (duration, retries) is defined by the Backend; the Agent Service does not define it

## 5.3 Game End

When a game ends (by any termination: goal, surrender, forced):

- The Backend stops calling RequestAction
- The Backend calls DestroyAgent (or the equivalent bulk-destroy for all room agents)
- The Agent Service releases instance resources

## 5.4 Crash Isolation

Agent crashes MUST NOT propagate to the Backend. The Agent Service process is responsible for isolating per-instance failures.

---

# 6. Responsibility Matrix

| Responsibility | Backend | Agent Service |
|---|---|---|
| Determine whose turn it is | ✓ | ✗ |
| Deliver GameState to agent | ✓ | ✗ |
| Provide legal pawn move list | ✓ | ✗ |
| Select action | ✗ | ✓ |
| Submit action to Engine | ✓ | ✗ |
| Validate action (rule authority) | Engine only | ✗ |
| Handle rejection (policy) | ✓ | ✗ |
| Advance Replay cursor | Backend triggers | Agent executes |
| Enforce decision timeout | ✓ | ✗ |
| Agent instance lifecycle | ✓ | ✗ (responds only) |
| Internal agent state management | ✗ | ✓ |

---

# 7. Extensibility Constraints

New agent types MUST satisfy:

1. **Interface conformance**: Implement the RequestAction response contract (receive GameState + legal_actions, return one Action)
2. **Isolation**: Produce actions without any external system calls
3. **Registration**: Declare a unique `type_id`, `display_name`, and `category`
4. **Reset support**: Expose a reset mechanism so instances can be reused for multiple games in the same binding (if the Backend ever requires it)
5. **Configuration**: Declare any required configuration keys in the agent's own documentation; the contract does not validate configuration content

New agent types DO NOT require:
- Changes to this contract
- Changes to the Backend
- Changes to the Frontend

---

# 8. Agent Category Behaviors

## 8.1 Scripted Agents (category: "scripted")

- No configuration required
- Stateless or trivially stateful (e.g., random seed)
- Decision is independent of game history beyond what is in GameState
- Examples: RandomAgent, DummyAgent

## 8.2 AI Agents (category: "ai")

- May maintain internal search state across turns (e.g., opening book, evaluation cache)
- Must reset internal state on the reset call
- Decision is derived from GameState analysis; no prior game knowledge is assumed
- Examples: GreedyAgent, future MCTS or ML agents

## 8.3 Replay Agents (category: "replay")

- Requires `config.actions` before start (list of Actions in sequence order)
- Maintains a cursor into the action sequence
- On RequestAction: return `actions[cursor]` without advancing
- On AdvanceCursor: advance cursor by one
- On reset: cursor returns to 0
- On exhaustion (cursor ≥ len(actions)): raise internal error

Replay Agents are deterministic given the same action sequence. They do not examine GameState for decision-making.

---

# Changelog

Version 1 (2026-04-07)
- Initial formal interface contract
- Defined Control Plane (ListTypes, Create, Configure, Start, Stop, Destroy)
- Defined Gameplay Plane (RequestAction, AdvanceCursor, rejection policy)
- Established shared data types (GameState, Action, AgentConfiguration)
- Captured per-category behavioral contracts (scripted, ai, replay)
- Defined responsibility matrix and extensibility constraints
