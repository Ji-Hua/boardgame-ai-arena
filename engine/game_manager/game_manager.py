"""GameManager — stateful orchestration over the Rust Rule Engine.

Manages a single game instance: lifecycle, state, history, and
action submission. All rule semantics are delegated to the Rule Engine
via the FFI layer. No rule logic lives here.

Lifecycle: UNINITIALIZED → RUNNING → TERMINAL
"""

from __future__ import annotations

from engine.game_manager import ffi
from engine.game_manager.types import ActionResult, State, Action, Player


class GameManager:
    """Deterministic state wrapper over the Rust Rule Engine.

    Owns current state, action history, and state history.
    Delegates all rule semantics to the Rule Engine via FFI.
    """

    def __init__(self) -> None:
        self._engine = ffi.create_rule_engine()
        self._initial_state: State | None = None
        self._current_state: State | None = None
        self._actions: list[Action] = []
        self._states: list[State] = []
        self._initialized: bool = False
        self._terminal: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the game. Creates initial state and clears history.

        Must fail if already initialized.
        """
        if self._initialized:
            raise RuntimeError("GameManager is already initialized")

        state = ffi.initial_state(self._engine)
        self._initial_state = state
        self._current_state = state
        self._actions = []
        self._states = []
        self._initialized = True
        self._terminal = False

    def is_initialized(self) -> bool:
        """Return whether the game has been initialized."""
        return self._initialized

    def terminate(self) -> None:
        """Mark the game as terminal. Idempotent."""
        self._terminal = True

    def is_terminal(self) -> bool:
        """Return whether the game is in terminal state."""
        return self._terminal

    # ------------------------------------------------------------------
    # Core Game Control
    # ------------------------------------------------------------------

    def submit_action(self, action: Action) -> ActionResult:
        """Submit an action to the game.

        Rejects if not initialized or terminal.
        Delegates validation and application to Rule Engine via FFI.
        On success: updates current_state, appends to actions and states.
        On failure: no state modification.
        """
        if not self._initialized:
            return ActionResult(success=False, error="GameManager is not initialized")
        if self._terminal:
            return ActionResult(success=False, error="GameManager is terminal")

        try:
            new_state = ffi.apply_action(self._engine, self._current_state, action)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

        self._actions.append(action)
        self._states.append(new_state)
        self._current_state = new_state

        return ActionResult(success=True, state=new_state)

    def undo(self) -> bool:
        """Revert the last action.

        Rejects if not initialized, terminal, or no actions to undo.
        On success: pops action and state, restores previous state.
        """
        if not self._initialized:
            return False
        if self._terminal:
            return False
        if not self._actions:
            return False

        self._actions.pop()
        self._states.pop()
        self._current_state = self._states[-1] if self._states else self._initial_state

        return True

    # ------------------------------------------------------------------
    # State Query
    # ------------------------------------------------------------------

    def current_state(self) -> State:
        """Return the current game state."""
        return self._current_state

    def initial_state(self) -> State:
        """Return the initial game state (immutable)."""
        return self._initial_state

    def legal_actions(self) -> list[Action]:
        """Return legal actions for the current state via Rule Engine."""
        return ffi.legal_actions(self._engine, self._current_state)

    # ------------------------------------------------------------------
    # Rule Semantic Query (Pass-through)
    # ------------------------------------------------------------------

    def is_game_over(self) -> bool:
        """Check if the game is over. Delegates to Rule Engine."""
        return ffi.is_game_over(self._engine, self._current_state)

    def winner(self) -> Player | None:
        """Return the winner, or None. Delegates to Rule Engine."""
        return ffi.winner(self._engine, self._current_state)

    def remaining_walls(self, player: Player) -> int:
        """Return remaining walls for a player. Delegates to Rule Engine."""
        return ffi.remaining_walls(self._current_state, player)

    def goal_cells(self, player: Player) -> set[tuple[int, int]]:
        """Return goal cells for a player. Delegates to Rule Engine."""
        return ffi.goal_cells(self._engine, player)

    def path_exists(self, player: Player) -> bool:
        """Check if a path exists for a player. Delegates to Rule Engine."""
        return ffi.path_exists(self._engine, self._current_state, player)

    def shortest_path_len(self, player: Player) -> int | None:
        """Return shortest path length for a player. Delegates to Rule Engine."""
        return ffi.shortest_path_len(self._engine, self._current_state, player)

    # ------------------------------------------------------------------
    # History Query
    # ------------------------------------------------------------------

    def actions(self) -> list[Action]:
        """Return all actions taken."""
        return list(self._actions)

    def states(self) -> list[State]:
        """Return all non-initial states."""
        return list(self._states)

    def step_count(self) -> int:
        """Return the number of actions taken."""
        return len(self._actions)

    def get_state_at(self, step: int) -> State:
        """Return state at a given step.

        step=0 returns initial_state.
        step>0 returns states[step-1].
        Raises IndexError if out of range.
        """
        if step < 0 or step > len(self._actions):
            raise IndexError(f"Step {step} out of range [0, {len(self._actions)}]")
        if step == 0:
            return self._initial_state
        return self._states[step - 1]

    # ------------------------------------------------------------------
    # Debug / Consistency
    # ------------------------------------------------------------------

    def replay(self) -> State:
        """Recompute final state by replaying all actions from initial state.

        Used for validation only.
        """
        state = self._initial_state
        for action in self._actions:
            state = ffi.apply_action(self._engine, state, action)
        return state
