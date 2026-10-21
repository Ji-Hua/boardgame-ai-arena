"""Head-to-head evaluation between two DQN checkpoints.

Loads two DQN checkpoints (A and B) and plays N games between them
using the Quoridor rule engine.  Seats are alternated fairly so that
neither checkpoint has a structural advantage from always playing as P1.

Usage
-----
    python scripts/evaluate_dqn_head_to_head.py \\
        --checkpoint-a PATH_A \\
        --checkpoint-b PATH_B \\
        --label-a terminal \\
        --label-b distance_delta \\
        --games 100 \\
        --seed 42 \\
        --device auto

Output
------
Prints a text summary and a JSON summary to stdout.  The JSON is
prefixed with ``--- JSON SUMMARY ---`` for easy extraction.

Limitations
-----------
- Both checkpoints must use the same obs_version (dqn_obs_v1).
- Legal masking is mandatory; illegal actions terminate the game immediately
  and award the win to the non-offending agent.
- This script does NOT use training reward — outcome only.
- Single-threaded; ~0.5–2 s per game depending on game length.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agent_system.training.dqn.model import select_greedy_action


# ---------------------------------------------------------------------------
# Device-aware agent wrapper
# ---------------------------------------------------------------------------

class _DeviceAgent:
    """Minimal greedy inference wrapper with device-aware tensor placement.

    ``load_checkpoint`` always loads networks to CPU.  This wrapper moves the
    network to the requested *device* so that GPU-accelerated inference is
    supported when requested.

    Parameters
    ----------
    network:
        A fully-loaded QNetwork (already in eval mode on CPU).
    device:
        Target torch.device for both the network and inference tensors.
    checkpoint_id:
        Identifier string used for logging / diagnostics.
    """

    def __init__(
        self,
        network: Any,
        device: torch.device,
        checkpoint_id: str = "dqn",
    ) -> None:
        self._network = network.to(device)
        self._network.eval()
        self._device = device
        self._checkpoint_id = checkpoint_id

    def select_action(
        self,
        observation: list[float],
        legal_action_mask: list[bool],
    ) -> int:
        """Select the greedy legal action.  Observation tensor is placed on
        the same device as the network before inference."""
        obs_tensor = torch.tensor(
            observation, dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            q_values = self._network(obs_tensor)
        return select_greedy_action(q_values, legal_action_mask)

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def _check_compatibility(ckpt_a: Any, ckpt_b: Any) -> None:
    """Raise ``ValueError`` if the two checkpoints are not head-to-head
    compatible (different obs_version, obs_size, action_count, or
    hidden_layers architecture).

    ``load_checkpoint`` already validates each checkpoint against current
    runtime constants.  This function performs cross-checkpoint validation
    so that A and B are guaranteed to observe the same state representation
    and produce Q-values over the same action space.

    For architecture comparison, ``hidden_layers`` is preferred when present
    in both checkpoints.  For old checkpoints that only store ``hidden_size``,
    the legacy ``hidden_size`` field is used as a fallback so that pre-16A
    checkpoints can still be compared against each other.
    """
    mismatches: list[str] = []

    scalar_fields = [
        ("observation_version", ckpt_a.observation_version, ckpt_b.observation_version),
        ("observation_size", ckpt_a.observation_size, ckpt_b.observation_size),
        ("action_count", ckpt_a.action_count, ckpt_b.action_count),
    ]
    for name, va, vb in scalar_fields:
        if va != vb:
            mismatches.append(f"  {name}: A={va!r}  B={vb!r}")

    # Architecture comparison: prefer hidden_layers (Phase 16A+); fall back
    # to hidden_size for old checkpoints that lack hidden_layers.
    hl_a = ckpt_a.hidden_layers
    hl_b = ckpt_b.hidden_layers
    if hl_a != hl_b:
        mismatches.append(f"  hidden_layers: A={hl_a!r}  B={hl_b!r}")

    if mismatches:
        raise ValueError(
            "Checkpoint compatibility check failed — checkpoints cannot be "
            "compared in head-to-head evaluation:\n" + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class H2HResult:
    """Aggregated results from a head-to-head evaluation run.

    Seat-split fields track wins/draws broken down by which seat each agent
    occupied in each game.  When A plays as P1 (even game index), a win for
    A is counted in ``a_wins_as_p1``; B's win is counted in ``b_wins_as_p2``
    (since B was P2).
    """

    label_a: str
    label_b: str
    num_games: int
    a_wins: int
    b_wins: int
    draws: int
    a_win_rate: float
    b_win_rate: float
    draw_rate: float
    avg_game_length: float
    a_illegal_actions: int
    b_illegal_actions: int
    game_lengths: list[int] = field(default_factory=list)
    a_wins_as_p1: int = 0
    a_wins_as_p2: int = 0
    b_wins_as_p1: int = 0
    b_wins_as_p2: int = 0
    draws_when_a_p1: int = 0
    draws_when_a_p2: int = 0


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_head_to_head(
    agent_a: _DeviceAgent,
    agent_b: _DeviceAgent,
    engine: Any,
    label_a: str,
    label_b: str,
    num_games: int = 100,
    max_steps: int = 3000,
) -> H2HResult:
    """Play *num_games* games between *agent_a* and *agent_b*.

    Seat alternation:
        Even game index (0, 2, …): A=P1, B=P2.
        Odd game index  (1, 3, …): B=P1, A=P2.

    Illegal action handling:
        If an agent selects an action that is not in the legal mask, the game
        terminates immediately.  The offending agent's illegal counter is
        incremented and the other agent is awarded the win.  The illegal
        action is **never** passed to ``engine.apply_action``.

    Parameters
    ----------
    agent_a, agent_b:
        :class:`_DeviceAgent` instances (greedy, device-aware).
    engine:
        ``quoridor_engine.RuleEngine``.
    label_a, label_b:
        Human-readable labels used only in the returned ``H2HResult``.
    num_games:
        Total games.  Should be even for a balanced seat split; if odd, A
        gets one extra P1 game.
    max_steps:
        Steps per game before declaring a draw.

    Returns
    -------
    H2HResult
    """
    from quoridor_engine import Player
    from agent_system.training.dqn.action_space import (
        legal_action_mask as _mask,
        decode_action_id,
    )
    from agent_system.training.dqn.observation import encode_observation

    a_wins = 0
    b_wins = 0
    draws = 0
    a_illegal = 0
    b_illegal = 0
    a_wins_as_p1 = 0
    a_wins_as_p2 = 0
    b_wins_as_p1 = 0
    b_wins_as_p2 = 0
    draws_when_a_p1 = 0
    draws_when_a_p2 = 0
    game_lengths: list[int] = []

    for game_idx in range(num_games):
        # Seat assignment
        a_is_p1 = (game_idx % 2 == 0)
        a_player = Player.P1 if a_is_p1 else Player.P2
        b_player = Player.P2 if a_is_p1 else Player.P1

        state = engine.initial_state()
        done = False
        steps = 0
        illegal_termination = False
        illegal_winner_is_a: bool = False  # only meaningful when illegal_termination

        while not done and steps < max_steps:
            current_player = state.current_player
            mask = _mask(engine, state)
            is_a_turn = (current_player == a_player)
            acting_agent = agent_a if is_a_turn else agent_b

            obs = encode_observation(state)
            action_id = acting_agent.select_action(obs, mask)

            # --- Illegal action: terminate immediately, do not apply ---
            if not mask[action_id]:
                if is_a_turn:
                    a_illegal += 1
                    illegal_winner_is_a = False  # B wins
                else:
                    b_illegal += 1
                    illegal_winner_is_a = True  # A wins
                illegal_termination = True
                steps += 1  # count the illegal step in game length
                break

            engine_action = decode_action_id(action_id, current_player)
            state = engine.apply_action(state, engine_action)
            steps += 1
            done = engine.is_game_over(state)

        game_lengths.append(steps)

        if illegal_termination:
            if illegal_winner_is_a:
                a_wins += 1
                if a_is_p1:
                    a_wins_as_p1 += 1
                else:
                    a_wins_as_p2 += 1
            else:
                b_wins += 1
                if a_is_p1:
                    b_wins_as_p2 += 1  # B was P2 when A was P1
                else:
                    b_wins_as_p1 += 1  # B was P1 when A was P2

        elif not done:
            # Max-steps draw
            draws += 1
            if a_is_p1:
                draws_when_a_p1 += 1
            else:
                draws_when_a_p2 += 1

        else:
            winner = engine.winner(state)
            if winner == a_player:
                a_wins += 1
                if a_is_p1:
                    a_wins_as_p1 += 1
                else:
                    a_wins_as_p2 += 1
            elif winner == b_player:
                b_wins += 1
                if a_is_p1:
                    b_wins_as_p2 += 1  # B was P2
                else:
                    b_wins_as_p1 += 1  # B was P1
            else:
                draws += 1
                if a_is_p1:
                    draws_when_a_p1 += 1
                else:
                    draws_when_a_p2 += 1

    avg_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0

    return H2HResult(
        label_a=label_a,
        label_b=label_b,
        num_games=num_games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_win_rate=a_wins / num_games,
        b_win_rate=b_wins / num_games,
        draw_rate=draws / num_games,
        avg_game_length=avg_len,
        a_illegal_actions=a_illegal,
        b_illegal_actions=b_illegal,
        game_lengths=game_lengths,
        a_wins_as_p1=a_wins_as_p1,
        a_wins_as_p2=a_wins_as_p2,
        b_wins_as_p1=b_wins_as_p1,
        b_wins_as_p2=b_wins_as_p2,
        draws_when_a_p1=draws_when_a_p1,
        draws_when_a_p2=draws_when_a_p2,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Head-to-head evaluation between two DQN checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-a", required=True,
        help="Path to checkpoint A (.pt file)",
    )
    p.add_argument(
        "--checkpoint-b", required=True,
        help="Path to checkpoint B (.pt file)",
    )
    p.add_argument("--label-a", default="A", help="Label for checkpoint A")
    p.add_argument("--label-b", default="B", help="Label for checkpoint B")
    p.add_argument("--games", type=int, default=100, help="Number of games to play")
    p.add_argument("--seed", type=int, default=42, help="Random seed (reserved for future use)")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Torch device for network inference.  "
            "'auto' selects cuda if available, else cpu.  "
            "'cuda' fails hard if CUDA is not available."
        ),
    )
    return p.parse_args(argv)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device.

    Raises
    ------
    RuntimeError:
        If ``device_str == "cuda"`` and CUDA is not available.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' was explicitly requested but "
            "torch.cuda.is_available() is False.  Check your CUDA installation."
        )
    return dev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        _run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _run(args: argparse.Namespace) -> None:
    from agent_system.training.dqn.checkpoint import load_checkpoint
    import quoridor_engine as qe

    device = _resolve_device(args.device)

    print("Head-to-Head DQN Evaluation")
    print(f"  A: {args.label_a}  →  {args.checkpoint_a}")
    print(f"  B: {args.label_b}  →  {args.checkpoint_b}")
    print(f"  games={args.games}  seed={args.seed}  device={device}")
    print()

    # --- Load and validate checkpoints ---
    path_a = Path(args.checkpoint_a)
    path_b = Path(args.checkpoint_b)
    if not path_a.exists():
        raise FileNotFoundError(f"Checkpoint A not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Checkpoint B not found: {path_b}")

    ckpt_a = load_checkpoint(path_a)
    ckpt_b = load_checkpoint(path_b)

    # Cross-checkpoint compatibility (obs_version, obs_size, action_count, hidden_size)
    _check_compatibility(ckpt_a, ckpt_b)

    print(f"  A checkpoint_id : {ckpt_a.checkpoint_id}")
    print(f"  B checkpoint_id : {ckpt_b.checkpoint_id}")
    print(f"  obs_version     : {ckpt_a.observation_version}")
    print(f"  obs_size        : {ckpt_a.observation_size}")
    print(f"  action_count    : {ckpt_a.action_count}")
    print(f"  hidden_layers   : {ckpt_a.hidden_layers}")
    print()

    # --- Build device-aware agents ---
    agent_a = _DeviceAgent(
        ckpt_a.network,
        device=device,
        checkpoint_id=f"{args.label_a}:{ckpt_a.checkpoint_id}",
    )
    agent_b = _DeviceAgent(
        ckpt_b.network,
        device=device,
        checkpoint_id=f"{args.label_b}:{ckpt_b.checkpoint_id}",
    )

    # --- Build engine ---
    engine = qe.RuleEngine.standard()

    # --- Run ---
    result = run_head_to_head(
        agent_a=agent_a,
        agent_b=agent_b,
        engine=engine,
        label_a=args.label_a,
        label_b=args.label_b,
        num_games=args.games,
    )

    # --- Text summary ---
    la, lb = args.label_a, args.label_b
    print("Results")
    print(f"  {'games:':<32} {result.num_games}")
    print(f"  {la + ' wins:':<32} {result.a_wins}  (win_rate={result.a_win_rate:.3f})")
    print(f"  {lb + ' wins:':<32} {result.b_wins}  (win_rate={result.b_win_rate:.3f})")
    print(f"  {'draws:':<32} {result.draws}  (draw_rate={result.draw_rate:.3f})")
    print(f"  {'avg_game_length:':<32} {result.avg_game_length:.1f}")
    print(f"  {la + ' illegal_actions:':<32} {result.a_illegal_actions}")
    print(f"  {lb + ' illegal_actions:':<32} {result.b_illegal_actions}")
    print()
    print("  Seat split:")
    print(f"    {la + ' wins as P1:':<30} {result.a_wins_as_p1}")
    print(f"    {la + ' wins as P2:':<30} {result.a_wins_as_p2}")
    print(f"    {lb + ' wins as P1:':<30} {result.b_wins_as_p1}")
    print(f"    {lb + ' wins as P2:':<30} {result.b_wins_as_p2}")
    print(f"    {'draws (A was P1):':<30} {result.draws_when_a_p1}")
    print(f"    {'draws (A was P2):':<30} {result.draws_when_a_p2}")
    print()

    # --- JSON summary ---
    summary = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "device": str(device),
        "num_games": result.num_games,
        "seed": args.seed,
        "a_wins": result.a_wins,
        "b_wins": result.b_wins,
        "draws": result.draws,
        "a_win_rate": result.a_win_rate,
        "b_win_rate": result.b_win_rate,
        "draw_rate": result.draw_rate,
        "avg_game_length": result.avg_game_length,
        "a_illegal_actions": result.a_illegal_actions,
        "b_illegal_actions": result.b_illegal_actions,
        "a_wins_as_p1": result.a_wins_as_p1,
        "a_wins_as_p2": result.a_wins_as_p2,
        "b_wins_as_p1": result.b_wins_as_p1,
        "b_wins_as_p2": result.b_wins_as_p2,
        "draws_when_a_p1": result.draws_when_a_p1,
        "draws_when_a_p2": result.draws_when_a_p2,
        "ckpt_a_obs_version": ckpt_a.observation_version,
        "ckpt_a_obs_size": ckpt_a.observation_size,
        "ckpt_a_action_count": ckpt_a.action_count,
        "ckpt_a_hidden_layers": ckpt_a.hidden_layers,
        "ckpt_b_obs_version": ckpt_b.observation_version,
        "ckpt_b_obs_size": ckpt_b.observation_size,
        "ckpt_b_action_count": ckpt_b.action_count,
        "ckpt_b_hidden_layers": ckpt_b.hidden_layers,
    }

    print("--- JSON SUMMARY ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
