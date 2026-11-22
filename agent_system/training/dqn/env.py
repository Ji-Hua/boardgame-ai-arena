"""Quoridor RL environment for DQN training.

Wraps the existing Engine directly. No backend/runtime coupling.

Usage:
    engine = RuleEngine.standard()
    env = QuoridorEnv(engine)
    obs = env.reset()                         # list[float], length 292
    mask = env.legal_action_mask()             # list[bool], length 209
    ids = env.legal_action_ids()               # list[int]
    obs, reward, done, info = env.step(action_id)
    raw = env.raw_state()                      # quoridor_engine.RawState

The environment is single-player in the sense that it always returns
from the perspective of the *learner*. The calling trainer is responsible
for choosing which seat is the learner and for executing opponent turns.

Observation (Phase 3 / dqn_obs_v1):
    reset() and step() return a flat list[float] of length
    OBSERVATION_SIZE (292) produced by encode_observation().
    Use raw_state() for the underlying RawState when needed.

Reward:
    Win  (learner is winner at terminal):    +1.0
    Loss (opponent is winner at terminal):   -1.0
    Non-terminal step:                        0.0
    If learner seat is not set (None), reward is always 0.0 at terminal.
"""

from __future__ import annotations

from typing import Any

from agent_system.training.dqn.action_space import (
    ACTION_SPACE_VERSION,
    decode_action_id,
    is_valid_action_id,
    legal_action_ids as _legal_action_ids,
    legal_action_mask as _legal_action_mask,
)
from agent_system.training.dqn.observation import (
    OBSERVATION_VERSION,
    encode_observation,
)


class QuoridorEnv:
    """Minimal Quoridor RL environment backed by the Rust Rule Engine.

    All state transitions are delegated to the Engine.
    No rule logic lives here.

    Parameters
    ----------
    engine:
        A ``quoridor_engine.RuleEngine`` instance (e.g. ``RuleEngine.standard()``).
    learner_player:
        Optional ``quoridor_engine.Player`` indicating which player the learner
        controls. Used only for reward computation. If None, reward is always 0.0
        at terminal states.
    """

    def __init__(self, engine: Any, learner_player: Any = None) -> None:
        self._engine = engine
        self._learner_player = learner_player
        self._state: Any = None
        self._done: bool = False
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> list[float]:
        """Reset to the standard initial game state.

        Returns an encoded observation (list[float] of length OBSERVATION_SIZE).
        Use raw_state() to access the underlying RawState.
        """
        self._state = self._engine.initial_state()
        self._done = False
        self._step_count = 0
        return encode_observation(self._state)

    def step(self, action_id: int) -> tuple[Any, float, bool, dict]:
        """Apply action_id to the environment.

        Parameters
        ----------
        action_id:
            An integer in [0, 209). Must be a legal action for the current state.

        Returns
        -------
        next_obs:   Encoded observation (list[float], length OBSERVATION_SIZE).
        reward:     Float reward from the learner's perspective.
        done:       True if the game has reached a terminal state.
        info:       Dict with debug information.

        Raises
        ------
        RuntimeError:
            If the environment has not been reset, is already done, or if
            action_id is out of the valid action space bounds.
        ValueError:
            If action_id is not a legal action in the current state (engine
            rejects it).
        """
        if self._state is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")
        if self._done:
            raise RuntimeError("Environment is done. Call reset() to start a new episode.")
        if not is_valid_action_id(action_id):
            raise RuntimeError(
                f"action_id {action_id!r} is outside the valid action space [0, 209)."
            )

        current_player = self._state.current_player
        action = decode_action_id(action_id, current_player)

        # Delegate to engine — raises ValueError if action is illegal.
        try:
            next_state = self._engine.apply_action(self._state, action)
        except Exception as exc:
            raise ValueError(
                f"Engine rejected action_id={action_id} "
                f"(decoded: kind={action.kind}, x={action.target_x}, y={action.target_y}): {exc}"
            ) from exc

        self._state = next_state
        self._step_count += 1

        done = self._engine.is_game_over(next_state)
        self._done = done

        reward = self._compute_reward(done, next_state)
        obs = encode_observation(next_state)

        info = {
            "action_id": action_id,
            "acting_player": current_player,
            "step_count": self._step_count,
            "done": done,
            "winner": self._engine.winner(next_state) if done else None,
            "legal_action_count": (
                len(self._engine.legal_actions(next_state)) if not done else 0
            ),
            "action_space_version": ACTION_SPACE_VERSION,
            "observation_version": OBSERVATION_VERSION,
        }

        return obs, reward, done, info

    def legal_action_mask(self) -> list[bool]:
        """Return the 209-element boolean mask for the current state.

        True at index i means action_id i is legal for the current player.

        Raises RuntimeError if not reset.
        """
        self._require_active()
        return _legal_action_mask(self._engine, self._state)

    def legal_action_ids(self) -> list[int]:
        """Return the list of legal action IDs for the current state.

        Raises RuntimeError if not reset.
        """
        self._require_active()
        return _legal_action_ids(self._engine, self._state)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def current_state(self) -> Any:
        """Return the current RawState. Raises RuntimeError if not reset.

        Deprecated alias for raw_state(); kept for backward compatibility.
        """
        self._require_active()
        return self._state

    def raw_state(self) -> Any:
        """Return the current RawState. Raises RuntimeError if not reset.

        Use this when the underlying engine state is needed (e.g. for
        observation debugging, wall-placement tests, or engine queries).
        """
        self._require_active()
        return self._state

    @property
    def is_done(self) -> bool:
        """True if the current episode has ended."""
        return self._done

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_active(self) -> None:
        if self._state is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

    def _compute_reward(self, done: bool, state: Any) -> float:
        """Compute learner-perspective reward.

        If learner_player is not set, return 0.0 for all transitions.
        """
        if not done:
            return 0.0
        if self._learner_player is None:
            return 0.0
        winner = self._engine.winner(state)
        if winner is None:
            return 0.0
        return 1.0 if winner == self._learner_player else -1.0
