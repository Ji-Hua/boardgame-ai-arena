"""FFI layer — thin wrapper over the Rust Rule Engine.

No logic. No validation. No state management.
All calls delegate directly to the Rust Rule Engine.

The `quoridor_engine` module is the Python binding of the Rust
`quoridor-engine` crate. When PyO3 bindings are built, they will
expose the types used here (RuleEngine, RawState, Action, Player).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Lazy import: resolved on first use, not at module load time.
# This allows the module to be imported (and mocked in tests)
# before the Rust PyO3 bindings are built.
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import quoridor_engine
        _engine = quoridor_engine
    return _engine


def create_rule_engine() -> Any:
    """Create a standard Rule Engine instance."""
    return _get_engine().RuleEngine.standard()


def initial_state(engine: Any) -> Any:
    """Get the initial game state from the Rule Engine."""
    return engine.initial_state()


def apply_action(engine: Any, state: Any, action: Any) -> Any:
    """Apply an action to the state. Raises on invalid action."""
    return engine.apply_action(state, action)


def legal_actions(engine: Any, state: Any) -> list:
    """Return all legal actions for the current player."""
    return engine.legal_actions(state)


def is_game_over(engine: Any, state: Any) -> bool:
    """Check if the game is over."""
    return engine.is_game_over(state)


def winner(engine: Any, state: Any) -> Any:
    """Return the winner, or None if no winner."""
    return engine.winner(state)


def remaining_walls(state: Any, player: Any) -> int:
    """Return number of walls remaining for a player."""
    return state.walls_remaining(player)


def goal_cells(engine: Any, player: Any) -> set[tuple[int, int]]:
    """Return the set of goal cells for a player."""
    n = engine.topology.n()
    goal_y = engine.topology.goal_y(player)
    return {(x, goal_y) for x in range(n)}


def path_exists(engine: Any, state: Any, player: Any) -> bool:
    """Check if a path exists from player's position to goal."""
    return engine.path_exists(state, player)


def shortest_path_len(engine: Any, state: Any, player: Any) -> int | None:
    """Return shortest path length to goal, or None if no path.

    Delegates to calculation.shortest_path_len in the Rust crate, which
    requires (state, player, topology) — the engine's topology is used.
    """
    return _get_engine().calculation.shortest_path_len(state, player, engine.topology)
