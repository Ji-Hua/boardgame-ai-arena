"""DQN checkpoint agent and simple vs-random evaluator.

DQNCheckpointAgent:
    Wraps a loaded DQNCheckpoint (or a QNetwork directly) and exposes a
    single `select_action(observation, legal_action_mask) -> action_id`
    interface.  Always uses deterministic greedy masked selection (epsilon=0).

    This is the minimal executable checkpoint agent needed for:
    - evaluation tests
    - future Arena integration (DQNCheckpointScorer)
    - future runtime registration

EvalResult:
    Dataclass with evaluation metrics for a fixed number of games.

evaluate_vs_random:
    Run a loaded DQN checkpoint against a uniform-random legal opponent
    for N games, collecting win/loss/draw counts, step counts, and
    illegal action diagnostics.

    Player perspective:
        The DQN agent plays as P1 in the first game, P2 in the second, etc.
        (alternating by game index).  Rewards and win/loss accounting are
        DQN-centric: the DQN wins if the engine declares it as winner.

    Opponent:
        Random: selects uniformly from engine legal actions at each turn.

    Draw / max-step handling:
        If a game exceeds max_steps without a terminal state, it is counted
        as a draw.  This is a safety guard only; well-trained agents rarely
        hit the limit.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from agent_system.training.dqn.action_space import (
    ACTION_COUNT,
    legal_action_ids as _legal_action_ids,
    legal_action_mask as _legal_action_mask,
)
from agent_system.training.dqn.checkpoint import DQNCheckpoint, load_checkpoint
from agent_system.training.dqn.model import QNetwork, select_greedy_action
from agent_system.training.dqn.observation import encode_observation


# ---------------------------------------------------------------------------
# DQNCheckpointAgent
# ---------------------------------------------------------------------------

class DQNCheckpointAgent:
    """Executable inference agent backed by a DQN checkpoint.

    Supports both MLP (``QNetwork``) and CNN (``CNNQNetwork``) networks.
    The network type is determined by the loaded checkpoint's ``model_arch``
    metadata and is handled transparently.

    Parameters
    ----------
    network:
        A fully-loaded QNetwork or CNNQNetwork (already in eval mode).
        Typically obtained via ``load_checkpoint(path).network``.
    checkpoint_id:
        Identifier string used for logging / diagnostics.
    """

    def __init__(
        self,
        network: nn.Module,
        checkpoint_id: str = "dqn_checkpoint",
    ) -> None:
        self._network = network
        self._network.eval()
        self._checkpoint_id = checkpoint_id

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint: DQNCheckpoint) -> "DQNCheckpointAgent":
        """Construct from a loaded :class:`DQNCheckpoint`."""
        return cls(
            network=checkpoint.network,
            checkpoint_id=checkpoint.checkpoint_id,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        expected_obs_version: str | None = None,
    ) -> "DQNCheckpointAgent":
        """Load checkpoint from *path* and construct agent.

        Parameters
        ----------
        expected_obs_version:
            Passed to :func:`load_checkpoint` for version validation.
            Defaults to the current ``OBSERVATION_VERSION`` (v1) if None.
        """
        ckpt = load_checkpoint(path, expected_obs_version=expected_obs_version)
        return cls.from_checkpoint(ckpt)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        observation: list[float],
        legal_action_mask: list[bool],
    ) -> int:
        """Select a legal action using greedy masked Q-value selection.

        Parameters
        ----------
        observation:
            Encoded observation of length OBSERVATION_SIZE (list[float]).
        legal_action_mask:
            Boolean mask of length ACTION_COUNT. True = legal.

        Returns
        -------
        int action_id in [0, ACTION_COUNT).

        The selected action is guaranteed to be in the legal mask provided
        that at least one action is legal (otherwise ValueError is raised).
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            q_values = self._network(obs_tensor)
        return select_greedy_action(q_values, legal_action_mask)

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    @property
    def network(self) -> nn.Module:
        """The underlying network module (QNetwork or CNNQNetwork)."""
        return self._network


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Metrics from a checkpoint evaluation run.

    Attributes
    ----------
    checkpoint_id:
        Identifier of the evaluated checkpoint.
    opponent_id:
        Identifier of the opponent policy (e.g. "random").
    num_games:
        Total games requested.
    wins:
        Games where the DQN agent was declared winner.
    losses:
        Games where the opponent was declared winner.
    draws:
        Games that hit max_steps without a terminal.
    win_rate:
        wins / num_games.
    avg_game_length:
        Mean step count per game.
    illegal_action_count:
        Total illegal actions selected by the DQN agent across all games.
        Zero is the expected value for a correctly-masked agent.
    game_lengths:
        Per-game step counts.
    """

    checkpoint_id: str
    opponent_id: str
    num_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_game_length: float
    illegal_action_count: int
    game_lengths: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_vs_random(
    agent: DQNCheckpointAgent,
    engine: Any,
    num_games: int = 20,
    max_steps: int = 3000,
    rng: _random.Random | None = None,
    opponent_id: str = "random",
    encoder: Callable[[Any], list[float]] | None = None,
) -> EvalResult:
    """Evaluate *agent* against a uniform-random legal opponent.

    The DQN agent plays as P1 in even-indexed games (0, 2, …) and as P2 in
    odd-indexed games (1, 3, …).  Win/loss accounting is DQN-centric.

    Parameters
    ----------
    agent:
        A :class:`DQNCheckpointAgent` to evaluate.
    engine:
        ``quoridor_engine.RuleEngine`` instance.
    num_games:
        Number of games to play.
    max_steps:
        Maximum steps per game before declaring a draw.
    rng:
        Optional ``random.Random`` for reproducible evaluation.
    opponent_id:
        Label for the opponent (used in the returned metrics).
    encoder:
        Observation encoder callable ``(RawState) -> list[float]``.
        Defaults to ``encode_observation`` (v1).  Pass
        ``encode_observation_v2`` when evaluating a v2-trained checkpoint.

    Returns
    -------
    EvalResult with aggregate statistics.
    """
    _rng = rng if rng is not None else _random.Random()
    _encoder = encoder if encoder is not None else encode_observation

    wins = 0
    losses = 0
    draws = 0
    illegal_count = 0
    game_lengths: list[int] = []

    # Lazy import to avoid circular dependency at module level
    from quoridor_engine import Player

    for game_idx in range(num_games):
        # Alternate DQN seat: even games = P1, odd games = P2
        dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2

        state = engine.initial_state()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Determine whose turn it is
            current_player = state.current_player
            mask = _legal_action_mask(engine, state)
            legal_ids = [i for i, v in enumerate(mask) if v]

            if current_player == dqn_player:
                # DQN turn: greedy masked selection
                obs = _encoder(state)
                action_id = agent.select_action(obs, mask)
                if not mask[action_id]:
                    illegal_count += 1
            else:
                # Opponent turn: uniform random legal
                action_id = _rng.choice(legal_ids)

            # Apply action
            from agent_system.training.dqn.action_space import decode_action_id
            engine_action = decode_action_id(action_id, current_player)
            state = engine.apply_action(state, engine_action)
            steps += 1
            done = engine.is_game_over(state)

        game_lengths.append(steps)

        if not done:
            # Hit max_steps — draw
            draws += 1
        else:
            winner = engine.winner(state)
            if winner == dqn_player:
                wins += 1
            else:
                losses += 1

    win_rate = wins / num_games if num_games > 0 else 0.0
    avg_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0

    return EvalResult(
        checkpoint_id=agent.checkpoint_id,
        opponent_id=opponent_id,
        num_games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        avg_game_length=avg_len,
        illegal_action_count=illegal_count,
        game_lengths=game_lengths,
    )


def evaluate_vs_opponent(
    agent: DQNCheckpointAgent,
    engine: Any,
    opponent: Any,
    num_games: int = 20,
    max_steps: int = 3000,
    rng: _random.Random | None = None,
    opponent_id: str = "opponent",
    encoder: Callable[[Any], list[float]] | None = None,
) -> EvalResult:
    """Evaluate *agent* against any :class:`TrainingOpponent`.

    Identical to :func:`evaluate_vs_random` except the opponent turn uses
    ``opponent.select_action_id(engine, state, legal_ids, rng)`` instead of
    uniform-random selection.

    Parameters
    ----------
    agent:
        A :class:`DQNCheckpointAgent` to evaluate.
    engine:
        ``quoridor_engine.RuleEngine`` instance.
    opponent:
        Any object with a ``select_action_id(engine, state, legal_ids, rng)``
        method (i.e. a :class:`TrainingOpponent`).
    num_games:
        Number of games to play.
    max_steps:
        Maximum steps per game before declaring a draw.
    rng:
        Optional ``random.Random`` for reproducible evaluation.
    opponent_id:
        Label for the opponent (used in the returned metrics).
    encoder:
        Observation encoder callable.  Defaults to ``encode_observation``.

    Returns
    -------
    EvalResult with aggregate statistics.
    """
    _rng = rng if rng is not None else _random.Random()
    _encoder = encoder if encoder is not None else encode_observation

    wins = 0
    losses = 0
    draws = 0
    illegal_count = 0
    game_lengths: list[int] = []

    from quoridor_engine import Player

    for game_idx in range(num_games):
        dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2

        state = engine.initial_state()
        done = False
        steps = 0

        while not done and steps < max_steps:
            current_player = state.current_player
            mask = _legal_action_mask(engine, state)
            legal_ids = [i for i, v in enumerate(mask) if v]

            if current_player == dqn_player:
                obs = _encoder(state)
                action_id = agent.select_action(obs, mask)
                if not mask[action_id]:
                    illegal_count += 1
            else:
                action_id = opponent.select_action_id(engine, state, legal_ids, _rng)

            from agent_system.training.dqn.action_space import decode_action_id
            engine_action = decode_action_id(action_id, current_player)
            state = engine.apply_action(state, engine_action)
            steps += 1
            done = engine.is_game_over(state)

        game_lengths.append(steps)

        if not done:
            draws += 1
        else:
            winner = engine.winner(state)
            if winner == dqn_player:
                wins += 1
            else:
                losses += 1

    win_rate = wins / num_games if num_games > 0 else 0.0
    avg_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0

    return EvalResult(
        checkpoint_id=agent.checkpoint_id,
        opponent_id=opponent_id,
        num_games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        avg_game_length=avg_len,
        illegal_action_count=illegal_count,
        game_lengths=game_lengths,
    )


def evaluate_vs_opponent_with_replays(
    agent: DQNCheckpointAgent,
    engine: Any,
    opponent: Any,
    num_games: int = 20,
    max_steps: int = 3000,
    rng: _random.Random | None = None,
    opponent_id: str = "opponent",
    encoder: Callable[[Any], list[float]] | None = None,
    sample_indices: set[int] | None = None,
    replay_kwargs: dict | None = None,
) -> tuple[EvalResult, list]:
    """Evaluate *agent* against *opponent* and optionally record full replays.

    Identical to :func:`evaluate_vs_opponent` except that for game indices
    in *sample_indices* the full move+state trace is recorded and returned as
    a list of :class:`~agent_system.training.dqn.replay_writer.GameReplay`.

    Parameters
    ----------
    sample_indices:
        Set of game indices (0-based) for which to record replays.
        If ``None`` or empty, no replays are recorded and the second return
        value is an empty list.
    replay_kwargs:
        Extra keyword arguments forwarded to
        :func:`~agent_system.training.dqn.replay_writer.record_game`
        (e.g. ``run_id``, ``agent_id``, ``episode``, ``checkpoint``,
        ``opp_cfg``).

    Returns
    -------
    (EvalResult, list[GameReplay])
        The aggregate evaluation metrics and the list of recorded replays.
    """
    from agent_system.training.dqn.action_space import decode_action_id
    from agent_system.training.dqn.replay_writer import record_game, GameReplay

    _rng = rng if rng is not None else _random.Random()
    _encoder = encoder if encoder is not None else encode_observation
    _sample = sample_indices or set()
    _replay_kwargs: dict = replay_kwargs or {}

    wins = 0
    losses = 0
    draws = 0
    illegal_count = 0
    game_lengths: list[int] = []
    replays: list[GameReplay] = []

    from quoridor_engine import Player

    for game_idx in range(num_games):
        dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2

        if game_idx in _sample:
            # Record the full trace for this game
            replay = record_game(
                engine=engine,
                agent=agent,
                opponent=opponent,
                dqn_player=dqn_player,
                game_index=game_idx,
                max_steps=max_steps,
                rng=_rng,
                encoder=_encoder,
                **_replay_kwargs,
            )
            replays.append(replay)
            game_lengths.append(replay.game_length)
            illegal_count += replay.illegal_acts
            result = replay.result_from_dqn_perspective
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            else:
                draws += 1
        else:
            # Fast path: no recording
            state = engine.initial_state()
            done = False
            steps = 0

            while not done and steps < max_steps:
                current_player = state.current_player
                mask = _legal_action_mask(engine, state)
                legal_ids = [i for i, v in enumerate(mask) if v]

                if current_player == dqn_player:
                    obs = _encoder(state)
                    action_id = agent.select_action(obs, mask)
                    if not mask[action_id]:
                        illegal_count += 1
                else:
                    action_id = opponent.select_action_id(engine, state, legal_ids, _rng)

                engine_action = decode_action_id(action_id, current_player)
                state = engine.apply_action(state, engine_action)
                steps += 1
                done = engine.is_game_over(state)

            game_lengths.append(steps)

            if not done:
                draws += 1
            else:
                winner = engine.winner(state)
                if winner == dqn_player:
                    wins += 1
                else:
                    losses += 1

    win_rate = wins / num_games if num_games > 0 else 0.0
    avg_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0

    result_obj = EvalResult(
        checkpoint_id=agent.checkpoint_id,
        opponent_id=opponent_id,
        num_games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        avg_game_length=avg_len,
        illegal_action_count=illegal_count,
        game_lengths=game_lengths,
    )
    return result_obj, replays
