# Minimax Agent

Author: Ji Hua  
Created Date: 2026-04-08  
Last Modified: 2026-04-08  
Current Version: 1  

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

# Changelog

Version 1 (2026-04-08)
- Initial Minimax Agent implementation document