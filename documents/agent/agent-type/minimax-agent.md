# Minimax Agent

Author: Ji Hua  
Created Date: 2026-04-08  
Last Modified: 2026-10-20
Current Version: 2

Document Type: Implementation  
Document Subtype: Agent  
Document Status: In Development  
Document Authority Scope: Agent module  

Document Purpose:  
This document defines the implementation of a Minimax-based agent for Quoridor. It specifies the decision logic, evaluation function, alpha-beta pruning strategy, and integration with the agent interface. The agent is stateless and operates as a pure decision function over GameState.

---

# 1. Overview

The Minimax Agent is a deterministic search-based agent that selects actions by exploring the game tree up to a configurable depth. It assumes optimal play from both players and uses alpha-beta pruning to reduce the search space.

The agent does not maintain internal state across moves and relies entirely on the provided GameState.

---

# 2. Design Principles

- Stateless decision function
- Deterministic behavior
- Depth-limited search
- Alpha-beta pruning for efficiency
- Evaluation based on shortest path heuristic
- Fully compliant with Agent interface

---

# 3. Agent Interface

The agent must implement:

choose_action(state, context) -> action

Where:
- state: current GameState
- context: SearchContext (for metrics and control)

The agent must not:
- Mutate state
- Maintain persistent internal game state
- Interact with backend directly

---

# 4. Core Algorithm

The agent performs:

1. Enumerate legal actions
2. Apply each action
3. Evaluate resulting states using minimax with alpha-beta pruning
4. Select action with highest value

---

# 5. Alpha-Beta Search

function alphabeta(state, depth, alpha, beta, maximizing, context):

    context.nodes += 1

    if depth == 0 or is_terminal(state):
        return evaluate(state)

    if maximizing:

        value = -∞

        for action in get_legal_actions(state):

            next_state = apply_action(state, action)

            score = alphabeta(
                next_state,
                depth - 1,
                alpha,
                beta,
                False,
                context
            )

            value = max(value, score)

            if value >= beta:
                context.cutoffs += 1
                break

            alpha = max(alpha, value)

        return value

    else:

        value = +∞

        for action in get_legal_actions(state):

            next_state = apply_action(state, action)

            score = alphabeta(
                next_state,
                depth - 1,
                alpha,
                beta,
                True,
                context
            )

            value = min(value, score)

            if value <= alpha:
                context.cutoffs += 1
                break

            beta = min(beta, value)

        return value

---

# 6. Decision Function

function choose_action(state, context):

    best_value = -∞
    best_action = None

    alpha = -∞
    beta = +∞

    for action in get_legal_actions(state):

        next_state = apply_action(state, action)

        value = alphabeta(
            next_state,
            context.depth_limit - 1,
            alpha,
            beta,
            False,
            context
        )

        if value > best_value:
            best_value = value
            best_action = action

        alpha = max(alpha, best_value)

    return best_action

---

# 7. Evaluation Function

function evaluate(state):

    if is_terminal(state):

        if MAX player wins:
            return +∞

        if MIN player wins:
            return -∞

    dist_self = shortest_path_len(state, MAX)
    dist_opp  = shortest_path_len(state, MIN)

    return dist_opp - dist_self

---

# 8. Dependencies

The agent depends on the following engine-provided functions:

- get_legal_actions(state)
- apply_action(state, action)
- is_terminal(state)
- shortest_path_len(state, player)

---

# 9. SearchContext Requirements

SearchContext must provide:

- depth_limit
- nodes (counter)
- cutoffs (counter)

Optional extensions:

- time_limit
- node_limit
- should_stop()

---

# 10. Complexity Considerations

Without pruning:

O(b^d)

Where:
- b = branching factor (~100+ in Quoridor)
- d = depth

With alpha-beta pruning:

Effective complexity is reduced but still depends heavily on action ordering.

---

# 11. Limitations

- High branching factor limits practical depth
- Evaluation function is heuristic and not exact
- No action pruning or ordering in v1
- No transposition table

---

# 12. Future Improvements

- Action ordering based on heuristic
- Candidate action pruning
- Iterative deepening
- Transposition table
- Integration with MCTS framework

---

# 13. Decision Analysis

`minimax_agent.score_position(state, depth, observer=None)` is the serving root search,
also consumed by Decision Lab. Root candidates share alpha exactly as in serving.
It returns the original scores and order plus conservative `exact`, `upper`, or
`lower` classifications. A pruned inferior root candidate's returned score can
differ from its true depth-limited value. Sorting those returns does not produce an
exact ranking of all actions.

`minimax_agent.inspect_branch(state, action, depth, observer=None)` searches one
candidate from a fresh `[-∞, +∞]` window, yielding its exact value at the chosen depth.
This remains a heuristic search horizon, not a game-theoretic solution of Quoridor.

`SearchContext.observer` is optional. Its `root_action`, `enter`, `edge`, and `exit`
events expose the actual recursion, original alpha/beta windows, leaf values, and
unvisited direct children after a cutoff. With no observer, no trace is created.
Decision Lab derives the principal variation from these events. The serving policy
and engine action ordering remain unchanged.

Depth counts half-moves: depth 2 contains the agent move and one opponent reply.
All values use the root player's perspective. Non-terminal leaves use
`distance(opponent) - distance(self)`; terminal leaves use `+∞` or `-∞`.
Depth 1 and Greedy have equivalent ordering among non-terminal results, but differ
in terminal treatment and in the numeric baseline.

The analysis component caps search resources and refuses to emit an incomplete
recommendation. Its separate trace-storage cap only limits what is displayed.
The trace distinguishes pruned branches from visited nodes omitted from storage.
The legacy runtime `SearchContext.nodes` includes an extra root-candidate increment;
Decision Lab's `stats.nodes` counts actual recursive `enter` events.

See [analysis contract](../analysis/decision-analysis-interface.md).

---

# Changelog

Version 2 (2026-10-20)
- Added shared scoring and full-window branch-inspection entries with opt-in search events.
- Defined exact values versus fail-soft bounds, trace limits, and depth/terminal semantics.

Version 1 (2026-04-08)
- Initial Minimax Agent implementation document
