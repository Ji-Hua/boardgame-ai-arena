# Greedy Agent Design

Author: Ji Hua  
Created Date: 2026-04-07  
Last Modified: 2026-04-07  
Current Version: 1  

Document Type: Implementation  
Document Subtype: Agent Strategy  
Document Status: In Development  
Document Authority Scope: Agent module  

Document Purpose:  
This document defines a minimal Greedy Agent implementation that performs one-step lookahead using the Rust Rule Engine. It specifies how the agent evaluates actions based on shortest path distance without introducing multi-step search or learning components.

---

# 1. Overview

The Greedy Agent is a one-step search agent that selects an action by evaluating immediate outcomes using a local (shadow) Rust Rule Engine.

The agent:
- Receives GameState and legal_actions from Backend
- Uses a local Rule Engine to simulate actions
- Computes shortest path distances for evaluation
- Returns exactly one action

The agent does not:
- Access Backend or Engine remotely
- Modify authoritative GameState
- Perform multi-step planning or learning

---

# 2. Core Principle

The agent optimizes a local objective:

Prefer actions that:
- Decrease own shortest path length
- Increase opponent's shortest path length

This is implemented as a one-step heuristic evaluation.

---

# 3. Engine Usage

The Greedy Agent MUST reuse the same Rust Rule Engine as the Backend.

The Rule Engine is used locally as a shadow simulation tool:

- legal_actions(state) → list of all legal actions (pawn moves + wall placements)
- apply_action(state, action) → next_state
- shortest_path_len(state, player) → integer

Invariant:

For any (state, action):

AgentEngine.apply(state, action) ≡ BackendEngine.apply(state, action)

This guarantees consistency between prediction and actual execution.

---

# 4. Decision Algorithm

Given:

- state (authoritative GameState from Backend)
- legal_actions (pawn moves provided by Backend, used as fallback only)

The agent performs:

1. Generate full action space (pawn moves + wall placements) using local Rule Engine:

   candidates = engine.legal_actions(state)

2. Compute baseline distances:

   my_before  = shortest_path_len(state, me)  
   opp_before = shortest_path_len(state, opponent)

3. For each action in candidates:

   next_state = apply_action(state, action)

   my_after  = shortest_path_len(next_state, me)  
   opp_after = shortest_path_len(next_state, opponent)

   delta_my  = my_after  - my_before  
   delta_opp = opp_after - opp_before

   score = delta_opp - delta_my

4. Select action with maximum score

5. Return selected action (converted to wire format)

---

# 5. Properties

## 5.1 Statelessness

- The agent does not maintain authoritative state
- Each decision depends only on input state

## 5.2 Determinism

- Given the same state and legal_actions, the agent produces the same action (unless tie-breaking introduces randomness)

## 5.3 Local Simulation

- All evaluation is performed locally using the Rule Engine
- No external calls are made during decision

---

# 6. Constraints

The Greedy Agent MUST:

- Generate the full action space locally via the Rule Engine
- Treat GameState as read-only
- Return exactly one action
- Fall back to legal_actions from Backend if engine generation fails per request

The Greedy Agent MUST NOT:

- Call Backend or Engine remotely
- Modify GameState
- Assume previous actions were accepted

---

# 7. Limitations

- Only performs one-step lookahead
- Does not model opponent strategy beyond distance heuristic
- May make suboptimal decisions in complex wall interactions

---

# 8. Extensibility

This design serves as a baseline for more advanced agents:

- Multi-step search (MCTS)
- Heuristic tuning
- Learning-based evaluation

All future agents can reuse the same local Rule Engine interface.

---

# Changelog

Version 1 (2026-04-07)
- Initial Greedy Agent design
- Defined one-step evaluation using shortest path heuristic
- Established requirement to reuse Rust Rule Engine
