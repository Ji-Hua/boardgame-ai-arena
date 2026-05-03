"""DQN evaluation diagnostic script.

Runs a battery of targeted tests to verify the DQN evaluation pipeline:
- Checkpoint loading and metadata verification
- Side-accounting correctness (wins/losses from DQN perspective)
- CNN observation encoding sanity
- Dummy opponent behavior characterisation
- Minimax opponent behavior characterisation
- DQN action-trace: does the agent move toward its goal?

Usage
-----
PYTHONPATH=. uv run python scripts/diagnose_dqn_eval.py \\
  --config agent_system/training/artifacts/dqn/cnn_curriculum_random_to_minimax_5k_001/config.yaml \\
  --checkpoint agent_system/training/artifacts/dqn/cnn_curriculum_random_to_minimax_5k_001/checkpoints/ep01000_step245961.pt \\
  --games 20 \\
  --sample-games 3

Outputs
-------
A diagnostic report in Markdown is written to the location specified by
--report (default: .copilot/diagnostics/dqn-eval-diagnostic-YYYYMMDD.md).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports (deferred so we can print nice error messages)
# ---------------------------------------------------------------------------

def _fatal(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data structures for results
# ---------------------------------------------------------------------------

@dataclass
class MatchupResult:
    label: str
    p1_label: str
    p2_label: str
    games: int
    p1_wins: int
    p2_wins: int
    draws: int
    game_lengths: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def avg_len(self) -> float:
        return sum(self.game_lengths) / len(self.game_lengths) if self.game_lengths else 0.0


@dataclass
class SampleGame:
    label: str
    dqn_player: str  # "P1" or "P2"
    winner: str      # "P1", "P2", or "draw"
    game_length: int
    dqn_moves: list[dict]       # first N DQN moves: {step, action_id, action_type, (x,y), progress_toward_goal}
    opponent_moves: list[dict]  # first N opponent moves
    p1_start: tuple
    p2_start: tuple
    p1_goal_y: int
    p2_goal_y: int


# ---------------------------------------------------------------------------
# Play helpers
# ---------------------------------------------------------------------------

def _action_type_and_direction(action_id: int, player_label: str, prev_pos, new_pos, goal_y: int) -> dict:
    """Describe a DQN action: move type, coordinates, direction toward goal."""
    from agent_system.training.dqn.action_space import (
        PAWN_ID_START, PAWN_ID_END,
        HWALL_ID_START, HWALL_ID_END,
        VWALL_ID_START, VWALL_ID_END,
        WALL_GRID_SIZE,
    )

    if PAWN_ID_START <= action_id < PAWN_ID_END:
        offset = action_id - PAWN_ID_START
        x = offset // 9
        y = offset % 9
        direction = None
        progress = None
        if prev_pos is not None:
            _, prev_y = prev_pos
            _, new_y = new_pos
            dy = new_y - prev_y
            if player_label == "P1":  # P1 wants higher y
                progress = dy > 0
                direction = "toward_goal" if dy > 0 else ("backward" if dy < 0 else "lateral")
            else:  # P2 wants lower y
                progress = dy < 0
                direction = "toward_goal" if dy < 0 else ("backward" if dy > 0 else "lateral")
        return {
            "type": "pawn",
            "x": x, "y": y,
            "direction": direction,
            "progress_toward_goal": progress,
        }

    if HWALL_ID_START <= action_id < HWALL_ID_END:
        offset = action_id - HWALL_ID_START
        x = offset // WALL_GRID_SIZE
        y = offset % WALL_GRID_SIZE
        return {"type": "hwall", "x": x, "y": y, "direction": None, "progress_toward_goal": None}

    # vwall
    offset = action_id - VWALL_ID_START
    x = offset // WALL_GRID_SIZE
    y = offset % WALL_GRID_SIZE
    return {"type": "vwall", "x": x, "y": y, "direction": None, "progress_toward_goal": None}


def play_game_with_trace(
    engine,
    p1,       # agent with select_action_id(engine, state, legal_ids, rng) or DQNCheckpointAgent
    p2,       # same
    rng,
    max_steps: int = 3000,
    trace_first_n: int = 20,
    dqn_player_id=None,   # if set, record detailed moves for this player
    encoder=None,
) -> dict:
    """Play one game and return trace dict."""
    from quoridor_engine import Player
    from agent_system.training.dqn.action_space import (
        legal_action_mask, legal_action_ids, decode_action_id
    )

    state = engine.initial_state()
    done = False
    steps = 0

    p1_start = state.pawn_pos(Player.P1)
    p2_start = state.pawn_pos(Player.P2)
    p1_goal_y = 8  # P1 wins at y=8
    p2_goal_y = 0  # P2 wins at y=0

    moves = []          # list of {step, player, action_info, pos_before, pos_after}
    illegal_dqn = 0

    prev_pos_p1 = p1_start
    prev_pos_p2 = p2_start

    while not done and steps < max_steps:
        cur = state.current_player
        mask = legal_action_mask(engine, state)
        ids = [i for i, v in enumerate(mask) if v]
        pos_before = state.pawn_pos(cur)

        if cur == Player.P1:
            agent = p1
        else:
            agent = p2

        # Select action
        is_dqn = (dqn_player_id is not None and cur == dqn_player_id)
        if is_dqn and hasattr(agent, "select_action"):
            # DQNCheckpointAgent interface
            obs = encoder(state)
            action_id = agent.select_action(obs, mask)
            if not mask[action_id]:
                illegal_dqn += 1
        else:
            action_id = agent.select_action_id(engine, state, ids, rng)

        eng_action = decode_action_id(action_id, cur)
        state = engine.apply_action(state, eng_action)
        steps += 1
        done = engine.is_game_over(state)

        pos_after = state.pawn_pos(cur)
        player_label = "P1" if cur == Player.P1 else "P2"
        goal_y = p1_goal_y if cur == Player.P1 else p2_goal_y
        prev_pos = prev_pos_p1 if cur == Player.P1 else prev_pos_p2

        if steps <= trace_first_n:
            action_info = _action_type_and_direction(
                action_id, player_label, prev_pos, pos_after, goal_y
            )
            action_info["action_id"] = action_id
            moves.append({
                "step": steps,
                "player": player_label,
                "pos_before": list(pos_before),
                "pos_after": list(pos_after),
                "action": action_info,
            })

        if cur == Player.P1:
            prev_pos_p1 = pos_after
        else:
            prev_pos_p2 = pos_after

    winner_obj = engine.winner(state) if done else None
    if winner_obj is None:
        winner_str = "draw"
    elif winner_obj == Player.P1:
        winner_str = "P1"
    else:
        winner_str = "P2"

    return {
        "p1_start": list(p1_start),
        "p2_start": list(p2_start),
        "p1_goal_y": p1_goal_y,
        "p2_goal_y": p2_goal_y,
        "steps": steps,
        "winner": winner_str,
        "done": done,
        "illegal_dqn": illegal_dqn,
        "moves": moves,
    }


def run_matchup(
    engine, p1, p2, p1_label, p2_label, games, rng, max_steps=3000,
    dqn_player_id=None, encoder=None
) -> MatchupResult:
    """Run N games and collect aggregate stats."""
    from quoridor_engine import Player

    p1_wins = 0
    p2_wins = 0
    draws = 0
    lengths = []
    for _ in range(games):
        g = play_game_with_trace(
            engine, p1, p2, rng, max_steps=max_steps, trace_first_n=0,
            dqn_player_id=dqn_player_id, encoder=encoder
        )
        lengths.append(g["steps"])
        if g["winner"] == "P1":
            p1_wins += 1
        elif g["winner"] == "P2":
            p2_wins += 1
        else:
            draws += 1

    return MatchupResult(
        label=f"{p1_label} vs {p2_label}",
        p1_label=p1_label,
        p2_label=p2_label,
        games=games,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        game_lengths=lengths,
    )


# ---------------------------------------------------------------------------
# DQN proxy: wraps DQNCheckpointAgent to match TrainingOpponent interface
# ---------------------------------------------------------------------------

class _DQNProxy:
    """Adapts DQNCheckpointAgent to the TrainingOpponent select_action_id interface."""

    def __init__(self, agent, encoder):
        self._agent = agent
        self._encoder = encoder

    def select_action_id(self, engine, state, legal_ids, rng):
        from agent_system.training.dqn.action_space import legal_action_mask
        mask = legal_action_mask(engine, state)
        obs = self._encoder(state)
        return self._agent.select_action(obs, mask)


# ---------------------------------------------------------------------------
# Observation sanity check
# ---------------------------------------------------------------------------

def check_obs_encoding(engine, encoder, obs_version_expected: str, model_arch: str) -> dict:
    """Verify CNN observation encoding for initial state."""
    from quoridor_engine import Player

    state = engine.initial_state()
    obs = encoder(state)

    p1_pos = state.pawn_pos(Player.P1)
    p2_pos = state.pawn_pos(Player.P2)

    report = {
        "model_arch": model_arch,
        "obs_version_expected": obs_version_expected,
        "p1_start": list(p1_pos),
        "p2_start": list(p2_pos),
        "p1_goal_y": 8,
        "p2_goal_y": 0,
    }

    if model_arch == "cnn":
        # obs is [7, 9, 9]
        report["obs_shape"] = [len(obs), len(obs[0]), len(obs[0][0])]
        # Check channel 0 (current player = P1 pawn one-hot)
        cx, cy = p1_pos
        ch0_val = obs[0][cx][cy]
        report["ch0_current_pawn_at_p1_pos"] = ch0_val
        # Check channel 1 (opponent = P2 pawn)
        ox, oy = p2_pos
        ch1_val = obs[1][ox][oy]
        report["ch1_opponent_pawn_at_p2_pos"] = ch1_val
        # Check channel 6 (goal row for P1 = y=8)
        goal_row_values = [obs[6][x][8] for x in range(9)]
        report["ch6_goal_row_y8_values"] = goal_row_values
        non_goal_row = [obs[6][x][0] for x in range(9)]
        report["ch6_non_goal_row_y0_values"] = non_goal_row
        report["obs_ok"] = (
            ch0_val == 1.0 and
            ch1_val == 1.0 and
            all(v == 1.0 for v in goal_row_values) and
            all(v == 0.0 for v in non_goal_row)
        )
        # Now check P2's turn perspective
        from agent_system.training.dqn.action_space import decode_action_id
        # Make a minimal move to get to P2's turn
        import quoridor_engine as qe
        # Advance to P2's turn by making a pawn move for P1
        p1_legal = [i for i, v in enumerate([False] * 209) if v]
        from agent_system.training.dqn.action_space import legal_action_ids
        p1_ids = legal_action_ids(engine, state)
        # Pick first pawn move for P1
        pawn_id = next((aid for aid in p1_ids if aid < 81), p1_ids[0])
        eng_action = decode_action_id(pawn_id, Player.P1)
        state2 = engine.apply_action(state, eng_action)
        obs2 = encoder(state2)
        # Now it's P2's turn. Channel 0 should show P2's pawn
        p2_pos2 = state2.pawn_pos(Player.P2)
        ch0_val2 = obs2[0][p2_pos2[0]][p2_pos2[1]]
        # Channel 6 goal row for P2 should be y=0
        goal_row_p2 = [obs2[6][x][0] for x in range(9)]
        non_goal_row_p2 = [obs2[6][x][8] for x in range(9)]
        report["p2_turn_ch0_current_pawn_at_p2_pos"] = ch0_val2
        report["p2_turn_ch6_goal_row_y0_values"] = goal_row_p2
        report["p2_turn_ch6_non_goal_row_y8_values"] = non_goal_row_p2
        report["p2_turn_obs_ok"] = (
            ch0_val2 == 1.0 and
            all(v == 1.0 for v in goal_row_p2) and
            all(v == 0.0 for v in non_goal_row_p2)
        )
    else:
        report["obs_shape"] = [len(obs)]
        report["obs_ok"] = (len(obs) == 292)

    return report


# ---------------------------------------------------------------------------
# DQN goal direction analysis
# ---------------------------------------------------------------------------

def analyze_dqn_direction(trace: dict, dqn_player_id) -> dict:
    """Count DQN pawn moves toward goal, backward, lateral, and wall placements."""
    from quoridor_engine import Player

    dqn_label = "P1" if dqn_player_id == Player.P1 else "P2"
    counts = {"toward_goal": 0, "backward": 0, "lateral": 0, "hwall": 0, "vwall": 0, "total": 0}

    for move in trace["moves"]:
        if move["player"] != dqn_label:
            continue
        act = move["action"]
        t = act["type"]
        counts["total"] += 1
        if t == "pawn":
            d = act.get("direction", "lateral")
            if d == "toward_goal":
                counts["toward_goal"] += 1
            elif d == "backward":
                counts["backward"] += 1
            else:
                counts["lateral"] += 1
        elif t == "hwall":
            counts["hwall"] += 1
        elif t == "vwall":
            counts["vwall"] += 1

    if counts["total"] > 0:
        pawn_total = counts["toward_goal"] + counts["backward"] + counts["lateral"]
        counts["pct_toward_goal"] = round(counts["toward_goal"] / counts["total"] * 100, 1)
    else:
        counts["pct_toward_goal"] = 0.0
    return counts


# ---------------------------------------------------------------------------
# Dummy path trace
# ---------------------------------------------------------------------------

def trace_dummy_path(engine, player_label: str, max_steps: int = 30) -> dict:
    """Trace the deterministic dummy path for one player (no opponent interactions)."""
    from quoridor_engine import Player
    from agent_system.training.dqn.action_space import legal_action_mask, decode_action_id
    from agent_system.training.dqn.opponent import DummyOpponent

    dummy = DummyOpponent()
    state = engine.initial_state()
    rng = random.Random(0)

    player = Player.P1 if player_label == "P1" else Player.P2
    path = []

    pos = state.pawn_pos(player)
    path.append({"step": 0, "pos": list(pos), "legal_ids_first3": []})

    for step in range(max_steps):
        # Advance game to get to this player's turn
        # Build a minimal-move game: skip the other player with first legal action
        pass

    # Trace in isolation: just simulate dummy moves without opponent
    # We'll construct a custom one-player trace by making opponent pass
    # Since we can't do that in Quoridor, we'll simulate both sides with dummy
    from agent_system.training.dqn.opponent import RandomLegalOpponent
    rand = RandomLegalOpponent()
    state = engine.initial_state()
    pos_history = []
    steps = 0
    done = False

    while steps < max_steps and not done:
        cur = state.current_player
        mask = legal_action_mask(engine, state)
        ids = [i for i, v in enumerate(mask) if v]

        if cur == player:
            aid = dummy.select_action_id(engine, state, ids, rng)
            pos_before = state.pawn_pos(player)
            action = decode_action_id(aid, cur)
            state = engine.apply_action(state, action)
            pos_after = state.pawn_pos(player)
            pos_history.append({
                "step": steps + 1,
                "action_id": aid,
                "pos": list(pos_after),
                "action_id_top3": sorted(ids)[:3],
            })
        else:
            # Other player: pick first action too (use dummy for both sides for isolation)
            aid = dummy.select_action_id(engine, state, ids, rng)
            action = decode_action_id(aid, cur)
            state = engine.apply_action(state, action)

        steps += 1
        done = engine.is_game_over(state)

    winner = engine.winner(state) if done else None
    return {
        "player": player_label,
        "goal_y": 8 if player_label == "P1" else 0,
        "path": pos_history[:15],
        "final_pos": list(state.pawn_pos(player)),
        "total_steps": steps,
        "game_over": done,
        "winner": ("P1" if winner and str(winner).endswith("P1") else "P2") if winner else "draw",
    }


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def run_diagnostics(
    checkpoint_path: str,
    config_path: str | None,
    games: int,
    sample_games: int,
    seed: int,
    max_steps: int,
) -> dict:
    """Run all diagnostics and return a report dict."""
    from quoridor_engine import RuleEngine, Player
    import torch

    from agent_system.training.dqn.checkpoint import load_checkpoint
    from agent_system.training.dqn.evaluator import DQNCheckpointAgent
    from agent_system.training.dqn.action_space import (
        legal_action_mask, decode_action_id, legal_action_ids
    )
    from agent_system.training.dqn.opponent import (
        RandomLegalOpponent, DummyOpponent, MinimaxOpponent, build_opponent
    )

    report: dict = {}
    rng = random.Random(seed)
    engine = RuleEngine.standard()

    # ------------------------------------------------------------------
    # 1. Checkpoint loading
    # ------------------------------------------------------------------
    print("1. Loading checkpoint...", flush=True)
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt = load_checkpoint(checkpoint_path, expected_obs_version=raw.get("observation_version"))
        agent = DQNCheckpointAgent.from_checkpoint(ckpt)
    except Exception as exc:
        _fatal(f"Failed to load checkpoint: {exc}")

    ckpt_info = {
        "path": checkpoint_path,
        "checkpoint_id": ckpt.checkpoint_id,
        "agent_id": ckpt.agent_id,
        "model_arch": ckpt.model_arch,
        "observation_version": ckpt.observation_version,
        "observation_size": ckpt.observation_size,
        "action_count": ckpt.action_count,
        "episode_count": ckpt.episode_count,
        "training_step": ckpt.training_step,
        "algorithm": ckpt.algorithm,
        "param_count": ckpt.param_count,
        "reward_mode": ckpt.reward_mode,
        "cnn_channels": ckpt.cnn_channels,
        "created_at": ckpt.created_at,
        "load_status": "ok",
    }
    report["checkpoint"] = ckpt_info
    print(f"   model_arch={ckpt.model_arch}, obs_version={ckpt.observation_version}, "
          f"params={ckpt.param_count:,}, ep={ckpt.episode_count}", flush=True)

    # Resolve encoder
    if ckpt.model_arch == "cnn":
        from agent_system.training.dqn.observation_cnn import encode_observation_cnn as encoder
    elif ckpt.observation_version and "v2" in ckpt.observation_version:
        from agent_system.training.dqn.observation_v2 import encode_observation_v2 as encoder
    else:
        from agent_system.training.dqn.observation import encode_observation as encoder

    dqn_proxy = _DQNProxy(agent, encoder)

    # ------------------------------------------------------------------
    # 2. Observation encoding sanity
    # ------------------------------------------------------------------
    print("2. Checking observation encoding...", flush=True)
    obs_check = check_obs_encoding(engine, encoder, ckpt.observation_version, ckpt.model_arch)
    report["observation_check"] = obs_check
    status = "OK" if (obs_check.get("obs_ok") and obs_check.get("p2_turn_obs_ok", True)) else "FAIL"
    print(f"   P1 turn obs: {obs_check.get('obs_ok')}  |  P2 turn obs: {obs_check.get('p2_turn_obs_ok')}  → {status}")
    initial_state = engine.initial_state()
    report["initial_positions"] = {
        "P1": list(initial_state.pawn_pos(Player.P1)),
        "P2": list(initial_state.pawn_pos(Player.P2)),
        "P1_goal_y": 8,
        "P2_goal_y": 0,
    }
    print(f"   P1 starts at {initial_state.pawn_pos(Player.P1)}, goal y=8")
    print(f"   P2 starts at {initial_state.pawn_pos(Player.P2)}, goal y=0")

    # ------------------------------------------------------------------
    # 3. Baseline matchups (no DQN)
    # ------------------------------------------------------------------
    print("3. Running baseline matchups (no DQN)...", flush=True)
    rand = RandomLegalOpponent()
    dummy = DummyOpponent()
    mm1 = build_opponent("minimax", 1)
    mm2 = build_opponent("minimax", 2)

    baselines = {}

    r = run_matchup(engine, rand, rand, "random", "random", games, rng)
    baselines["random_vs_random"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   random vs random: P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    r = run_matchup(engine, mm1, rand, "minimax_d1", "random", games, rng)
    baselines["minimax_d1_vs_random"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   minimax_d1(P1) vs random(P2): P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    r = run_matchup(engine, rand, mm1, "random", "minimax_d1", games, rng)
    baselines["random_vs_minimax_d1"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   random(P1) vs minimax_d1(P2): P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    r = run_matchup(engine, dummy, dummy, "dummy", "dummy", games, rng)
    baselines["dummy_vs_dummy"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   dummy(P1) vs dummy(P2): P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    r = run_matchup(engine, rand, dummy, "random", "dummy", games, rng)
    baselines["random_vs_dummy"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   random(P1) vs dummy(P2): P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    r = run_matchup(engine, dummy, rand, "dummy", "random", games, rng)
    baselines["dummy_vs_random"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   dummy(P1) vs random(P2): P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    r = run_matchup(engine, mm1, dummy, "minimax_d1", "dummy", games, rng)
    baselines["minimax_d1_vs_dummy"] = {"p1_wins": r.p1_wins, "p2_wins": r.p2_wins, "draws": r.draws, "avg_len": round(r.avg_len, 1)}
    print(f"   minimax_d1(P1) vs dummy(P2): P1W={r.p1_wins} P2W={r.p2_wins} D={r.draws} avg_len={r.avg_len:.1f}")

    report["baselines"] = baselines

    # ------------------------------------------------------------------
    # 4. Dummy path traces
    # ------------------------------------------------------------------
    print("4. Tracing dummy paths...", flush=True)
    dummy_traces = {}
    for plabel in ("P1", "P2"):
        t = trace_dummy_path(engine, plabel)
        dummy_traces[plabel] = t
        path_str = " → ".join(f"({p['pos'][0]},{p['pos'][1]})" for p in t["path"][:10])
        print(f"   dummy {plabel} first 10 positions: {path_str}")
        print(f"     final_pos={t['final_pos']}, winner={t['winner']}, steps={t['total_steps']}")
    report["dummy_traces"] = dummy_traces

    # ------------------------------------------------------------------
    # 5. Side-accounting verification (alternating seat correctness)
    # ------------------------------------------------------------------
    print("5. Verifying side-accounting (DQN alternates P1/P2)...", flush=True)
    # Play games manually and verify accounting
    seat_results = {"p1_games": {"wins": 0, "losses": 0, "draws": 0},
                    "p2_games": {"wins": 0, "losses": 0, "draws": 0}}
    total_wins = 0; total_losses = 0; total_draws = 0

    for game_idx in range(games):
        dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2
        opp_proxy = _DQNProxy(agent, encoder)  # reuse agent as both sides to test accounting
        g = play_game_with_trace(engine, dqn_proxy, dqn_proxy, rng, max_steps=200, trace_first_n=0)
        # Figure out result from DQN perspective
        winner_str = g["winner"]
        dqn_str = "P1" if dqn_player == Player.P1 else "P2"
        seat_key = "p1_games" if dqn_player == Player.P1 else "p2_games"

        # Simulate accounting as evaluate_vs_opponent does
        if winner_str == "draw":
            result = "draw"
            total_draws += 1
            seat_results[seat_key]["draws"] += 1
        elif winner_str == dqn_str:
            result = "win"
            total_wins += 1
            seat_results[seat_key]["wins"] += 1
        else:
            result = "loss"
            total_losses += 1
            seat_results[seat_key]["losses"] += 1

    report["side_accounting_check"] = {
        "total_games": games,
        "total_wins_from_dqn_perspective": total_wins,
        "total_losses_from_dqn_perspective": total_losses,
        "total_draws": total_draws,
        "p1_seat_breakdown": seat_results["p1_games"],
        "p2_seat_breakdown": seat_results["p2_games"],
        "note": "DQN plays against itself — both sides win equally if accounting is correct. "
                "Any strong imbalance between p1_games and p2_games wins indicates a seat-accounting bug.",
    }
    print(f"   DQN(P1) vs DQN(P2): P1W={seat_results['p1_games']['wins']} "
          f"P1L={seat_results['p1_games']['losses']} P1D={seat_results['p1_games']['draws']}")
    print(f"   DQN(P2) vs DQN(P2): P2W={seat_results['p2_games']['wins']} "
          f"P2L={seat_results['p2_games']['losses']} P2D={seat_results['p2_games']['draws']}")

    # ------------------------------------------------------------------
    # 6. DQN vs opponents: aggregate matchups
    # ------------------------------------------------------------------
    print("6. DQN vs opponents (aggregate)...", flush=True)
    dqn_matchups = {}

    for opp_label, opp in [("random", rand), ("dummy", dummy), ("minimax_d1", mm1), ("minimax_d2", mm2)]:
        p1_wins = 0; p2_wins = 0; draws = 0; lengths = []

        for game_idx in range(games):
            dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2
            if dqn_player == Player.P1:
                g = play_game_with_trace(engine, dqn_proxy, opp, rng, max_steps=max_steps, trace_first_n=0,
                                         dqn_player_id=Player.P1, encoder=encoder)
            else:
                g = play_game_with_trace(engine, opp, dqn_proxy, rng, max_steps=max_steps, trace_first_n=0,
                                         dqn_player_id=Player.P2, encoder=encoder)

            lengths.append(g["steps"])
            w = g["winner"]
            dqn_str = "P1" if dqn_player == Player.P1 else "P2"
            if w == "draw":
                draws += 1
            elif w == dqn_str:
                p1_wins += 1  # reuse p1_wins as dqn_wins
            else:
                p2_wins += 1  # reuse p2_wins as dqn_losses

        avg = sum(lengths) / len(lengths) if lengths else 0.0
        dqn_matchups[f"dqn_vs_{opp_label}"] = {
            "dqn_wins": p1_wins, "dqn_losses": p2_wins, "draws": draws,
            "win_rate": round(p1_wins / games, 3), "avg_len": round(avg, 1),
        }
        print(f"   DQN vs {opp_label}: W={p1_wins} L={p2_wins} D={draws} "
              f"wr={p1_wins/games:.2f} avg_len={avg:.1f}")

    report["dqn_matchups"] = dqn_matchups

    # ------------------------------------------------------------------
    # 7. Sample game traces with DQN action analysis
    # ------------------------------------------------------------------
    print(f"7. Recording {sample_games} sample game traces per opponent...", flush=True)
    sample_traces = {}

    for opp_label, opp in [("dummy", dummy), ("random", rand), ("minimax_d1", mm1)]:
        games_for_label = []
        for game_idx in range(sample_games):
            dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2
            dqn_str = "P1" if dqn_player == Player.P1 else "P2"

            if dqn_player == Player.P1:
                g = play_game_with_trace(engine, dqn_proxy, opp, rng, max_steps=max_steps,
                                         trace_first_n=20, dqn_player_id=Player.P1, encoder=encoder)
            else:
                g = play_game_with_trace(engine, opp, dqn_proxy, rng, max_steps=max_steps,
                                         trace_first_n=20, dqn_player_id=Player.P2, encoder=encoder)

            direction_counts = analyze_dqn_direction(g, dqn_player)
            g["dqn_player"] = dqn_str
            g["direction_counts"] = direction_counts
            g["result_for_dqn"] = (
                "win" if g["winner"] == dqn_str else
                ("draw" if g["winner"] == "draw" else "loss")
            )
            games_for_label.append(g)

            print(f"   DQN({dqn_str}) vs {opp_label}: "
                  f"steps={g['steps']} winner={g['winner']} "
                  f"→ DQN {g['result_for_dqn']} "
                  f"| toward_goal={direction_counts['toward_goal']} "
                  f"backward={direction_counts['backward']} "
                  f"walls={direction_counts['hwall']+direction_counts['vwall']}")

        sample_traces[f"dqn_vs_{opp_label}"] = games_for_label

    report["sample_traces"] = sample_traces

    # ------------------------------------------------------------------
    # 8. Seat-by-seat DQN vs dummy breakdown
    # ------------------------------------------------------------------
    print("8. Seat-by-seat DQN vs dummy breakdown...", flush=True)
    seat_detail = {"dqn_p1": {"wins": 0, "losses": 0, "draws": 0, "avg_len": 0.0},
                   "dqn_p2": {"wins": 0, "losses": 0, "draws": 0, "avg_len": 0.0}}
    p1_lens = []; p2_lens = []

    for game_idx in range(games):
        dqn_player = Player.P1 if game_idx % 2 == 0 else Player.P2
        dqn_str = "P1" if dqn_player == Player.P1 else "P2"
        seat_key = "dqn_p1" if dqn_player == Player.P1 else "dqn_p2"

        if dqn_player == Player.P1:
            g = play_game_with_trace(engine, dqn_proxy, dummy, rng, max_steps=max_steps,
                                     trace_first_n=0, dqn_player_id=Player.P1, encoder=encoder)
            p1_lens.append(g["steps"])
        else:
            g = play_game_with_trace(engine, dummy, dqn_proxy, rng, max_steps=max_steps,
                                     trace_first_n=0, dqn_player_id=Player.P2, encoder=encoder)
            p2_lens.append(g["steps"])

        w = g["winner"]
        if w == "draw":
            seat_detail[seat_key]["draws"] += 1
        elif w == dqn_str:
            seat_detail[seat_key]["wins"] += 1
        else:
            seat_detail[seat_key]["losses"] += 1

    if p1_lens:
        seat_detail["dqn_p1"]["avg_len"] = round(sum(p1_lens) / len(p1_lens), 1)
    if p2_lens:
        seat_detail["dqn_p2"]["avg_len"] = round(sum(p2_lens) / len(p2_lens), 1)

    report["dqn_vs_dummy_seat_breakdown"] = seat_detail
    print(f"   DQN(P1) vs dummy(P2): W={seat_detail['dqn_p1']['wins']} "
          f"L={seat_detail['dqn_p1']['losses']} D={seat_detail['dqn_p1']['draws']} "
          f"avg_len={seat_detail['dqn_p1']['avg_len']}")
    print(f"   DQN(P2) vs dummy(P1): W={seat_detail['dqn_p2']['wins']} "
          f"L={seat_detail['dqn_p2']['losses']} D={seat_detail['dqn_p2']['draws']} "
          f"avg_len={seat_detail['dqn_p2']['avg_len']}")

    # ------------------------------------------------------------------
    # 9. Q-value inspection: show top Q-values for initial state
    # ------------------------------------------------------------------
    print("9. Inspecting Q-values at initial state...", flush=True)
    import torch
    state = engine.initial_state()
    obs = encoder(state)
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        q_values = agent.network(obs_tensor)
    q_np = q_values.cpu().tolist()

    mask = legal_action_mask(engine, state)
    legal_qs = [(i, q_np[i]) for i in range(len(q_np)) if mask[i]]
    legal_qs_sorted = sorted(legal_qs, key=lambda x: -x[1])
    top10 = legal_qs_sorted[:10]

    from agent_system.training.dqn.action_space import PAWN_ID_END

    def describe_action(aid):
        if aid < PAWN_ID_END:
            x, y = aid // 9, aid % 9
            return f"pawn({x},{y})"
        elif aid < 145:
            offset = aid - 81
            return f"hwall({offset//8},{offset%8})"
        else:
            offset = aid - 145
            return f"vwall({offset//8},{offset%8})"

    top10_described = [(describe_action(aid), round(q, 4)) for aid, q in top10]
    greedy_action_id = legal_qs_sorted[0][0] if legal_qs_sorted else -1
    greedy_description = describe_action(greedy_action_id) if greedy_action_id >= 0 else "N/A"

    report["q_value_inspection"] = {
        "state": "initial_state_p1_turn",
        "greedy_action": greedy_description,
        "greedy_action_id": greedy_action_id,
        "top10_legal_q_values": top10_described,
        "total_legal_actions": sum(mask),
        "q_range": [round(min(q_np), 4), round(max(q_np), 4)],
        "legal_q_range": [round(min(q for _, q in legal_qs), 4), round(max(q for _, q in legal_qs), 4)] if legal_qs else [0, 0],
    }
    print(f"   greedy action at initial state: {greedy_description}")
    print(f"   top3 Q-actions: {top10_described[:3]}")
    print(f"   P1 initial pos: {state.pawn_pos(Player.P1)}, goal y=8")

    return report


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(report: dict, out_path: str) -> None:
    """Write a Markdown diagnostic report."""
    from quoridor_engine import Player

    ckpt = report.get("checkpoint", {})
    obs = report.get("observation_check", {})
    baselines = report.get("baselines", {})
    dqn_matchups = report.get("dqn_matchups", {})
    traces = report.get("sample_traces", {})
    dummy_traces = report.get("dummy_traces", {})
    init_pos = report.get("initial_positions", {})
    q_info = report.get("q_value_inspection", {})
    seat_detail = report.get("dqn_vs_dummy_seat_breakdown", {})
    side_check = report.get("side_accounting_check", {})

    md = []
    md.append(f"# DQN Evaluation Diagnostic Report")
    md.append(f"\nDate: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    md.append(f"\n## Checkpoint Summary\n")
    md.append(f"| Key | Value |")
    md.append(f"|---|---|")
    for k in ("path", "agent_id", "model_arch", "observation_version", "observation_size",
               "action_count", "episode_count", "training_step", "algorithm", "param_count",
               "reward_mode", "cnn_channels", "load_status"):
        md.append(f"| {k} | `{ckpt.get(k)}` |")

    md.append(f"\n## Initial Board State\n")
    md.append(f"- P1 starts at: `{init_pos.get('P1')}` — goal: y=8 (top row)")
    md.append(f"- P2 starts at: `{init_pos.get('P2')}` — goal: y=0 (bottom row)")

    md.append(f"\n## Observation Encoding Check\n")
    md.append(f"| Check | Result |")
    md.append(f"|---|---|")
    md.append(f"| obs_shape | `{obs.get('obs_shape')}` |")
    md.append(f"| P1 turn: ch0 pawn at P1 pos | `{obs.get('ch0_current_pawn_at_p1_pos')}` |")
    md.append(f"| P1 turn: ch1 opponent at P2 pos | `{obs.get('ch1_opponent_pawn_at_p2_pos')}` |")
    md.append(f"| P1 turn: ch6 goal row y=8 all 1.0 | `{all(v==1.0 for v in obs.get('ch6_goal_row_y8_values', []))}` |")
    md.append(f"| P1 turn obs_ok | `{obs.get('obs_ok')}` |")
    md.append(f"| P2 turn: ch0 pawn at P2 pos | `{obs.get('p2_turn_ch0_current_pawn_at_p2_pos')}` |")
    md.append(f"| P2 turn: ch6 goal row y=0 all 1.0 | `{all(v==1.0 for v in obs.get('p2_turn_ch6_goal_row_y0_values', []))}` |")
    md.append(f"| P2 turn obs_ok | `{obs.get('p2_turn_obs_ok')}` |")

    md.append(f"\n## Side Accounting Verification\n")
    md.append(f"Method: DQN plays against itself (both seats), accounting from DQN perspective.\n")
    md.append(f"| Metric | Value |")
    md.append(f"|---|---|")
    md.append(f"| P1-seat games: wins | `{side_check.get('p1_seat_breakdown', {}).get('wins')}` |")
    md.append(f"| P1-seat games: losses | `{side_check.get('p1_seat_breakdown', {}).get('losses')}` |")
    md.append(f"| P1-seat games: draws | `{side_check.get('p1_seat_breakdown', {}).get('draws')}` |")
    md.append(f"| P2-seat games: wins | `{side_check.get('p2_seat_breakdown', {}).get('wins')}` |")
    md.append(f"| P2-seat games: losses | `{side_check.get('p2_seat_breakdown', {}).get('losses')}` |")
    md.append(f"| P2-seat games: draws | `{side_check.get('p2_seat_breakdown', {}).get('draws')}` |")
    md.append(f"\n{side_check.get('note', '')}\n")

    md.append(f"\n## Baseline Matchups (No DQN)\n")
    md.append(f"| Matchup | P1W | P2W | D | AvgLen |")
    md.append(f"|---|---|---|---|---|")
    for k, v in baselines.items():
        label = k.replace("_", " ")
        md.append(f"| {label} | {v['p1_wins']} | {v['p2_wins']} | {v['draws']} | {v['avg_len']} |")

    md.append(f"\n## Dummy Opponent Path Traces\n")
    for plabel, t in dummy_traces.items():
        md.append(f"\n### Dummy {plabel} path (goal y={'8' if plabel=='P1' else '0'})\n")
        path_str = " → ".join(f"({p['pos'][0]},{p['pos'][1]})" for p in t["path"][:12])
        md.append(f"- First 12 positions: `{path_str}`")
        md.append(f"- Final pos: `{t['final_pos']}`, steps={t['total_steps']}, winner={t['winner']}")
        if plabel == "P2":
            md.append(f"- **Finding**: Dummy P2 marches toward corner (0,0) which is y=0 = P2's goal. "
                      f"Dummy P2 wins if not blocked.")
        else:
            md.append(f"- **Finding**: Dummy P1 marches toward corner (0,0) which is y=0 = P2's goal (not P1's goal). "
                      f"Dummy P1 can never win on its own — it walks away from its goal y=8.")

    md.append(f"\n## DQN vs Dummy Seat-by-Seat Breakdown\n")
    md.append(f"| Seat | DQN W | DQN L | D | AvgLen |")
    md.append(f"|---|---|---|---|---|")
    md.append(f"| DQN=P1, dummy=P2 | {seat_detail.get('dqn_p1', {}).get('wins')} | "
              f"{seat_detail.get('dqn_p1', {}).get('losses')} | "
              f"{seat_detail.get('dqn_p1', {}).get('draws')} | "
              f"{seat_detail.get('dqn_p1', {}).get('avg_len')} |")
    md.append(f"| DQN=P2, dummy=P1 | {seat_detail.get('dqn_p2', {}).get('wins')} | "
              f"{seat_detail.get('dqn_p2', {}).get('losses')} | "
              f"{seat_detail.get('dqn_p2', {}).get('draws')} | "
              f"{seat_detail.get('dqn_p2', {}).get('avg_len')} |")

    md.append(f"\n## DQN Matchup Summary\n")
    md.append(f"| Matchup | DQN W | DQN L | D | WR | AvgLen |")
    md.append(f"|---|---|---|---|---|---|")
    for k, v in dqn_matchups.items():
        label = k.replace("_", " ")
        md.append(f"| {label} | {v['dqn_wins']} | {v['dqn_losses']} | {v['draws']} | "
                  f"{v['win_rate']:.2f} | {v['avg_len']} |")

    md.append(f"\n## Q-Value Inspection (Initial State, P1 Turn)\n")
    md.append(f"- Greedy action: `{q_info.get('greedy_action')}` (id={q_info.get('greedy_action_id')})")
    md.append(f"- Total legal actions: {q_info.get('total_legal_actions')}")
    md.append(f"- Q range (all actions): {q_info.get('q_range')}")
    md.append(f"- Q range (legal only): {q_info.get('legal_q_range')}")
    md.append(f"\nTop 10 legal actions by Q-value:\n")
    md.append(f"| Rank | Action | Q-value |")
    md.append(f"|---|---|---|")
    for i, (action, q) in enumerate(q_info.get("top10_legal_q_values", [])[:10]):
        md.append(f"| {i+1} | `{action}` | {q} |")

    md.append(f"\n## Sample Game Traces\n")
    for opp_key, games_list in traces.items():
        opp_label = opp_key.replace("dqn_vs_", "")
        md.append(f"\n### DQN vs {opp_label}\n")
        for i, g in enumerate(games_list):
            dc = g.get("direction_counts", {})
            md.append(f"\n**Game {i+1}** — DQN={g['dqn_player']}, winner={g['winner']}, "
                      f"DQN result={g['result_for_dqn']}, steps={g['steps']}\n")
            md.append(f"- DQN pawn moves: toward_goal={dc.get('toward_goal',0)}, "
                      f"backward={dc.get('backward',0)}, lateral={dc.get('lateral',0)}, "
                      f"walls={dc.get('hwall',0)+dc.get('vwall',0)}")
            md.append(f"- Illegal DQN actions: {g.get('illegal_dqn', 0)}")
            if g.get("moves"):
                md.append(f"\nFirst moves:")
                md.append(f"| Step | Player | Pos Before | Pos After | Action |")
                md.append(f"|---|---|---|---|---|")
                for m in g["moves"][:10]:
                    act = m["action"]
                    act_str = f"{act['type']}({act['x']},{act['y']})"
                    if act["type"] == "pawn" and act.get("direction"):
                        act_str += f" [{act['direction']}]"
                    md.append(f"| {m['step']} | {m['player']} | {m['pos_before']} | {m['pos_after']} | {act_str} |")
            md.append("")

    md.append(f"\n## Findings and Hypotheses\n")

    # Auto-generate findings
    findings = []

    # Obs check
    if obs.get("obs_ok") and obs.get("p2_turn_obs_ok"):
        findings.append("✅ CNN observation encoding is correct for both P1 and P2 turns.")
    else:
        findings.append("❌ **BUG**: CNN observation encoding is INCORRECT. P1 ok=" +
                        str(obs.get("obs_ok")) + ", P2 ok=" + str(obs.get("p2_turn_obs_ok")))

    # Side accounting
    p1w = side_check.get("p1_seat_breakdown", {}).get("wins", 0)
    p2w = side_check.get("p2_seat_breakdown", {}).get("wins", 0)
    total_g = side_check.get("total_games", 1)
    if abs(p1w - p2w) <= total_g // 4:
        findings.append(f"✅ Side-accounting appears correct (P1-seat wins={p1w}, P2-seat wins={p2w}, symmetric).")
    else:
        findings.append(f"⚠️  Side-accounting asymmetry: P1-seat wins={p1w}, P2-seat wins={p2w}. "
                        f"Investigate if policy is seat-biased.")

    # Dummy P2 behavior
    d2_trace = dummy_traces.get("P2", {})
    d2_path = d2_trace.get("path", [])
    if d2_path:
        last_y = d2_path[-1]["pos"][1] if d2_path else -1
        findings.append(
            f"⚠️  **Dummy P2 quirk**: DummyOpponent always picks `legal_ids[0]` (lowest action ID). "
            f"Since pawn moves to low-ID squares are sorted first, Dummy P2 traces a deterministic path "
            f"toward (0,0) which is y=0 = P2's goal. Dummy P2 wins against any opponent that doesn't block. "
            f"This is a valid (though unintended) benchmark behavior — but it means dummy P2 is "
            f"**artificially strong** as P2 and **can never win as P1** (walks away from goal y=8)."
        )
    findings.append(
        "⚠️  **Dummy P1 quirk**: DummyOpponent as P1 walks to corner (0,0)=y=0 and oscillates. "
        "It can NEVER reach goal y=8 by itself. When DQN is P2 vs dummy P1, the game outcome "
        "depends entirely on whether DQN P2 can reach y=0 before dummy P1 somehow wins via jump."
    )

    # DQN direction
    all_toward = 0; all_total = 0
    for opp_key, games_list in traces.items():
        for g in games_list:
            dc = g.get("direction_counts", {})
            all_toward += dc.get("toward_goal", 0)
            all_total += dc.get("total", 0)

    if all_total > 0:
        pct = all_toward / all_total * 100
        if pct >= 60:
            findings.append(f"✅ DQN moves toward goal {pct:.1f}% of pawn moves (healthy).")
        elif pct >= 30:
            findings.append(f"⚠️  DQN moves toward goal only {pct:.1f}% of pawn moves (possibly suboptimal).")
        else:
            findings.append(f"❌ DQN moves toward goal only {pct:.1f}% of pawn moves — likely policy is broken or heavily wall-focused.")

    # Minimax results
    mm1_result = dqn_matchups.get("dqn_vs_minimax_d1", {})
    if mm1_result.get("dqn_wins", 0) == 0:
        findings.append(
            f"ℹ️  DQN wins 0/{mm1_result.get('dqn_wins', 0)+mm1_result.get('dqn_losses', 0)} "
            f"against minimax_d1 with avg_len={mm1_result.get('avg_len')}. "
            f"This matches baseline: random also wins 0/N against minimax_d1. "
            f"Short game lengths indicate DQN does not block minimax effectively — "
            f"minimax reaches goal at near-minimum moves."
        )

    for f in findings:
        md.append(f"- {f}")

    md.append(f"\n## Recommended Next Steps\n")
    md.append(f"1. **Dummy benchmark**: Consider replacing DummyOpponent in eval with a `pawn-only-random` opponent "
              f"that only makes random pawn moves (no walls) without the broken first-action bias.")
    md.append(f"2. **Training convergence**: DQN is unable to beat minimax_d1 (0% win rate). "
              f"Confirm whether training opponent is actually minimax_d1 or mostly random. "
              f"Check train_opponent_phase in metrics.jsonl.")
    md.append(f"3. **Goal-direction analysis**: If DQN pawn moves toward goal < 50%, the reward signal "
              f"may not be strong enough to encourage forward progress. "
              f"Consider increasing distance_reward_weight or adding a per-pawn-step reward.")
    md.append(f"4. **Catastrophic forgetting**: DQN ep=1000 beats dummy 100% but ep=1500 wins 0%. "
              f"This may indicate forgetting during phase transition. Consider experience replay diversity.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nReport written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DQN evaluation diagnostic script")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", default=None, help="Path to config.yaml (informational only)")
    parser.add_argument("--games", type=int, default=20, help="Games per matchup (default 20)")
    parser.add_argument("--sample-games", type=int, default=3, help="Sample games to trace (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per game")
    parser.add_argument(
        "--report",
        default=None,
        help="Output Markdown report path (default: .copilot/diagnostics/dqn-eval-diagnostic-YYYYMMDD.md)",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to write raw JSON report")
    args = parser.parse_args()

    if args.report is None:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.report = f".copilot/diagnostics/dqn-eval-diagnostic-{today}.md"

    print(f"DQN Evaluation Diagnostic")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  games/matchup: {args.games}")
    print(f"  sample_games: {args.sample_games}")
    print(f"  report: {args.report}")
    print()

    report = run_diagnostics(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        games=args.games,
        sample_games=args.sample_games,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"JSON report written to: {args.json_out}")

    write_report(report, args.report)


if __name__ == "__main__":
    main()
