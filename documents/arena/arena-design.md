# Quoridor Arena Evaluation System — MVP Design

Author: Ji Hua  
Date: 2026-04-16  
Status: Draft  

---

## 1. Purpose

This document defines the MVP (Minimum Viable Product) design for a Quoridor evaluation system.

The goal is to build a **minimal, stable, and reproducible evaluation loop** that:

- Runs automated matches between baseline agents
- Records results in a lightweight database (SQLite)
- Produces a win-rate matrix (confusion matrix)
- Provides a minimal frontend for visualization

This system is intentionally scoped to avoid over-engineering and to establish a reliable foundation for future evaluation extensions.

---

## 2. System Scope

### Included

- 4 baseline agents:
  - Random
  - Greedy
  - Minimax (depth=2)
  - Minimax (depth=3)

- Automated agent vs agent matches
- Match orchestration via Arena Runner
- Full execution through Backend → Engine
- Game-level result storage in SQLite
- Win-rate aggregation
- Minimal frontend visualization (confusion matrix)

---

### Excluded (Non-Goals)

- ELO rating
- Training system (self-play, RL, etc.)
- Replay system
- Step-level or trajectory logging
- Real-time UI or live updates
- Complex metrics or analytics
- Distributed execution

---

## 3. System Architecture

### High-Level Structure

Arena system is a separate layer that orchestrates evaluation:

Arena Runner
    ↓
Backend (orchestrator)
    ↓
Engine (rule authority)

Agent definitions are shared system components:

Agent
    ↑
Arena materializes AgentInstance per game

---

### Design Principles

1. **Separation of Concerns**
    - Arena does not define agent algorithms
    - Arena does not modify game rules
    - Arena only selects Agents, instantiates AgentInstances, orchestrates matches, and collects results

2. **Single Source of Truth**
    - Engine is the only rule authority
    - All state transitions must go through Backend → Engine
    - Agent is the only agent-definition authority

3. **Determinism**
    - All matches must be reproducible via explicit instance seeds

4. **Minimalism**
    - Only essential data is stored
    - No premature feature expansion

---

## 4. Arena Runner Design

### Responsibilities

- Schedule matches between Agents
- Create AgentInstances for each game
- Execute multiple games per matchup
- Collect and aggregate results
- Persist game-level data
- Control per-instance seeds for reproducibility

---

### Core Interfaces

class ArenaRunner:
    def run_match(self, agent_a, agent_b, num_games) -> MatchResult
    def run_tournament(self, agents, num_games) -> TournamentResult

---

### Match Result

class MatchResult:
    wins_a: int
    wins_b: int
    draws: int

---

### Tournament Result

class TournamentResult:
    results: dict[(agent_a, agent_b)] -> MatchResult

---

## 5. AlgoType, Scorer, Policy, Agent, and AgentInstance

Arena does not operate on raw agent classes as its core abstraction.

The correct model is:

AlgoType → Scorer → Agent → AgentInstance

---

### 5.1 AlgoType

AlgoType defines the algorithm family:

- minimax
- greedy
- random

It defines capability but does not include parameters or behavior details.

---

### 5.2 Scorer

Scorer is:

AlgoType + parameters

Scorer is responsible for evaluating actions.

Input:
- state

Output:
- [(action, score)]

Scorer does NOT choose the final action.

---

### 5.3 Policy

Policy defines how to select a single action from scored candidates.

Input:
- [(action, score)], rng

Output:
- action

Policy is a first-class component and must be explicitly defined.

Examples:
- argmax (deterministic)
- top_k
- epsilon

---

### 5.4 Agent

Agent is the complete agent definition:

Agent = Scorer + Policy

At the conceptual level:

action = Policy(Scorer(state), rng)

Agent must have a stable identity:

- algo_type (e.g., minimax)
- parameters (e.g., depth=3)
- policy (e.g., top_k=3)

This identity must be serializable and comparable.

---

### 5.5 AgentInstance

AgentInstance is the runtime realization of an Agent.

AgentInstance:

- is created from an Agent for a particular game runtime
- owns seed / RNG state for that runtime
- is used in both arena evaluation and backend gameplay

The architectural difference between arena evaluation and backend gameplay is not a different kind of agent definition.

- Both must realize the same Agent
- The main runtime difference is seed choice and orchestration context
- Arena typically uses fixed seeds for reproducibility
- Backend gameplay may use dynamic seeds or runtime context

Constraints:

- Must not mutate shared state
- Must operate through provided state only
- May maintain runtime-local logic required by execution

---

## 6. Game Execution Flow

Single game execution must follow the full authoritative chain:

AgentInstance → Backend → Engine

---

### Pseudocode

def play_single_game(agent_a, agent_b, seed_a, seed_b):
    state = backend.new_game()
    instance_a = instantiate(agent_a, seed_a)
    instance_b = instantiate(agent_b, seed_b)

    while not state.is_game_over:
        if state.current_player == 1:
            agent = instance_a
        else:
            agent = instance_b

        action = request_action(agent, state)
        state = backend.apply_action(action)

    return state.winner

Arena must not bypass the agent definition layer by directly constructing arbitrary classes.

- Arena selects Agents
- Arena creates AgentInstances per game
- Arena controls seeds for reproducibility

---

## 7. Determinism

All experiments must be reproducible.

Requirements:

- Arena must assign explicit seeds when materializing AgentInstances
- Stochastic behavior must consume instance-owned RNG
- Backend and Engine must not introduce uncontrolled randomness

Example:

instantiate(agent=random_agent, seed=42)

---

## 8. Tournament Execution

Baseline Agents:

- Random
- Greedy
- Minimax (depth=2)
- Minimax (depth=3)

Execution:

for each pair (agent_A, agent_B):
    run N games
    record results

For each game, Arena creates fresh AgentInstances from the selected Agents and assigned seeds.

Suggested initial value:

N = 50

---

## 9. SQLite Storage Design

SQLite is used as a lightweight experiment record store.

---

### Table: games

games
- id (primary key)
- agent_a (string)
- agent_b (string)
- winner (string or int)
- num_steps (int)
- seed (int)
- created_at (timestamp)

Stored identity must refer to Agent (algo + params + policy), not runtime classes.

---

### Optional Table: runs

runs
- id
- config (json)
- created_at

---

### Design Constraints

- Only store game-level results
- Do NOT store:
  - per-step actions
  - full state traces

---

## 10. Aggregation Logic

Arena results must be aggregated into a win-rate matrix.

Aggregation compares results between Agents under controlled seeds.

---

### Output Format (Matrix)

        Random  Greedy  MM(d2)  MM(d3)
Random    —      0.23    0.10    0.05
Greedy   0.77     —      0.35    0.20
MM(d2)   0.90    0.65     —      0.45
MM(d3)   0.95    0.80    0.55     —

---

### Output Format (Text)

Random vs Greedy → 23% / 77%  
Greedy vs MM(d2) → 35% / 65%  
MM(d2) vs MM(d3) → 45% / 55%  

---

## 11. Frontend Design (Minimal Viewer)

### Purpose

The frontend is a **read-only result viewer**.

---

### Data Flow

SQLite → Backend API → Frontend → Render

---

### Backend API

GET /results

Returns:

- Aggregated matrix (JSON)

---

### Frontend Responsibilities

- Render confusion matrix
- Display basic statistics

---

### UI Scope

Single-page UI with:

- Table view
- Optional heatmap

---

### Explicit Non-Goals

- No real-time updates
- No WebSocket
- No replay
- No match control

---

## 12. Success Criteria

1. Arena runs automatically
2. Matches execute through Backend → Engine
3. Arena operates on Agent-defined logic and materializes AgentInstances
4. Results stored in SQLite
5. Win-rate matrix generated
6. Frontend displays correctly
7. Strength ordering is reasonable:

   Random < Greedy < Minimax(d2) ≤ Minimax(d3)

---

## 13. Future Extensions (Not in Scope)

- ELO rating
- Advanced metrics
- Replay system
- Experiment dashboard
- Distributed execution
- Training integration

---

## 14. Summary

This design defines a **minimal evaluation harness** that:

- Uses AlgoType → Scorer → Agent → AgentInstance abstraction
- Treats Agent as the complete, serializable agent definition
- Separates scoring (Scorer) from decision (Policy)
- Uses AgentInstance as the runtime execution unit
- Produces reliable and reproducible results
- Avoids premature complexity
