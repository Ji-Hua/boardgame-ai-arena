"""Evaluation replay recording utilities.

Records full game traces during DQN evaluation and writes them as offline
replay JSON files. The replay format is self-contained — the dashboard viewer
can replay without any live backend or engine calls.

Replay JSON schema (schema_version="eval_replay_v1"):
-----------------------------------------------------
{
  "schema_version": "eval_replay_v1",
  "run_id": "...",
  "agent_id": "...",
  "episode": 1000,
  "checkpoint": "ep01000_step245961.pt",
  "eval_opponent": {
    "name": "random_legal",
    "type": "random_legal",
    "depth": null
  },
  "game_index": 0,
  "dqn_player_id": "P1",
  "opponent_player_id": "P2",
  "winner": "P1",
  "result_from_dqn_perspective": "win",
  "game_length": 42,
  "illegal_acts": 0,
  "termination_reason": "terminal",
  "created_at": "2026-05-02T00:00:00Z",
  "actions": [
    {
      "turn_index": 0,
      "player_id": "P1",
      "action_id": 37,
      "action_type": "pawn",
      "x": 4,
      "y": 1,
      "is_dqn_action": true,
      "actor_name": "dqn_agent_id",
      "actor_type": "dqn"
    },
    ...
  ],
  "states": [
    {
      "p1": [4, 0],
      "p2": [4, 8],
      "h_walls": [],
      "v_walls": [],
      "walls_remaining_p1": 10,
      "walls_remaining_p2": 10
    },
    ...
  ]
}

State encoding notes:
- `p1` and `p2` are [x, y] engine coordinates (x=0 left, y=0 bottom)
- `h_walls` and `v_walls` are lists of [x, y] in action_space wall-head coordinates
  (x in [0,7], y in [0,7], where (0,0) is bottom-left corner of wall grid)
- `states[0]` is the initial state before any action
- `states[i]` is the state AFTER actions[i-1] has been applied
- len(states) == len(actions) + 1

Wall coordinate mapping for rendering:
  H wall head (wx, wy): horizontal wall between engine rows wy and wy+1,
                        spanning engine cols wx and wx+1
  V wall head (wx, wy): vertical wall between engine cols wx and wx+1,
                        spanning engine rows wy and wy+1
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_system.training.dqn.action_space import (
    PAWN_ID_START, PAWN_ID_END,
    HWALL_ID_START, HWALL_ID_END,
    VWALL_ID_START, VWALL_ID_END,
    WALL_GRID_SIZE, BOARD_SIZE,
    decode_action_id,
)

REPLAY_SCHEMA_VERSION = "eval_replay_v1"
_WALL_BITS = WALL_GRID_SIZE * WALL_GRID_SIZE  # 64


# ---------------------------------------------------------------------------
# State snapshot helpers
# ---------------------------------------------------------------------------

def _decode_wall_bitmask(bitmask: int) -> list[list[int]]:
    """Return list of [wx, wy] for each set bit in a wall-head bitmask."""
    walls: list[list[int]] = []
    for bit_idx in range(_WALL_BITS):
        if (bitmask >> bit_idx) & 1:
            wx = bit_idx // WALL_GRID_SIZE
            wy = bit_idx % WALL_GRID_SIZE
            walls.append([wx, wy])
    return walls


def encode_state_snapshot(state: object) -> dict[str, Any]:
    """Encode a RawState into a minimal snapshot dict for replay storage.

    Returns
    -------
    dict with keys:
        p1: [x, y] — P1 pawn position (engine coords)
        p2: [x, y] — P2 pawn position (engine coords)
        h_walls: [[wx, wy], ...] — horizontal wall head coords
        v_walls: [[wx, wy], ...] — vertical wall head coords
        walls_remaining_p1: int
        walls_remaining_p2: int
    """
    from quoridor_engine import Player

    p1 = state.pawn_pos(Player.P1)
    p2 = state.pawn_pos(Player.P2)
    return {
        "p1": [int(p1[0]), int(p1[1])],
        "p2": [int(p2[0]), int(p2[1])],
        "h_walls": _decode_wall_bitmask(state.horizontal_heads),
        "v_walls": _decode_wall_bitmask(state.vertical_heads),
        "walls_remaining_p1": int(state.walls_remaining(Player.P1)),
        "walls_remaining_p2": int(state.walls_remaining(Player.P2)),
    }


def decode_action_info(action_id: int) -> dict[str, Any]:
    """Decode an action_id into a human-readable action info dict."""
    if PAWN_ID_START <= action_id < PAWN_ID_END:
        offset = action_id - PAWN_ID_START
        x = offset // BOARD_SIZE
        y = offset % BOARD_SIZE
        return {"action_type": "pawn", "x": x, "y": y}
    if HWALL_ID_START <= action_id < HWALL_ID_END:
        offset = action_id - HWALL_ID_START
        x = offset // WALL_GRID_SIZE
        y = offset % WALL_GRID_SIZE
        return {"action_type": "hwall", "x": x, "y": y}
    # vwall
    offset = action_id - VWALL_ID_START
    x = offset // WALL_GRID_SIZE
    y = offset % WALL_GRID_SIZE
    return {"action_type": "vwall", "x": x, "y": y}


# ---------------------------------------------------------------------------
# Single-game replay recorder
# ---------------------------------------------------------------------------

@dataclass
class GameReplay:
    """Recorded game data ready to serialise to JSON."""

    schema_version: str = REPLAY_SCHEMA_VERSION
    run_id: str = ""
    agent_id: str = ""
    episode: int = 0
    checkpoint: str = ""
    eval_opponent: dict[str, Any] = field(default_factory=dict)
    game_index: int = 0
    dqn_player_id: str = "P1"
    opponent_player_id: str = "P2"
    winner: str | None = None   # "P1" | "P2" | null (draw)
    result_from_dqn_perspective: str = "draw"  # "win" | "loss" | "draw"
    game_length: int = 0
    illegal_acts: int = 0
    termination_reason: str = "terminal"   # "terminal" | "timeout"
    created_at: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    states: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "episode": self.episode,
            "checkpoint": self.checkpoint,
            "eval_opponent": self.eval_opponent,
            "game_index": self.game_index,
            "dqn_player_id": self.dqn_player_id,
            "opponent_player_id": self.opponent_player_id,
            "winner": self.winner,
            "result_from_dqn_perspective": self.result_from_dqn_perspective,
            "game_length": self.game_length,
            "illegal_acts": self.illegal_acts,
            "termination_reason": self.termination_reason,
            "created_at": self.created_at,
            "actions": self.actions,
            "states": self.states,
        }


def record_game(
    engine: object,
    agent: object,
    opponent: object,
    dqn_player: object,  # quoridor_engine.Player
    game_index: int,
    max_steps: int,
    rng: object,
    encoder: object,
    # Metadata
    run_id: str = "",
    agent_id: str = "",
    episode: int = 0,
    checkpoint: str = "",
    opp_cfg: dict[str, Any] | None = None,
) -> GameReplay:
    """Play one game and record the full trace as a GameReplay.

    Parameters
    ----------
    engine:
        RuleEngine instance.
    agent:
        DQNCheckpointAgent with select_action(obs, mask) -> int.
    opponent:
        TrainingOpponent with select_action_id(engine, state, legal_ids, rng) -> int.
    dqn_player:
        Player.P1 or Player.P2.
    game_index:
        Index within the evaluation batch (for identification).
    max_steps:
        Maximum steps before draw.
    rng:
        random.Random instance.
    encoder:
        Observation encoder callable (state) -> list[float].
    """
    from quoridor_engine import Player
    from agent_system.training.dqn.action_space import legal_action_mask

    player_id_str = "P1" if dqn_player == Player.P1 else "P2"
    opp_player = Player.P2 if dqn_player == Player.P1 else Player.P1
    opp_player_id_str = "P2" if dqn_player == Player.P1 else "P1"

    opp_cfg = opp_cfg or {}
    opp_name = opp_cfg.get("name") or opp_cfg.get("type", "opponent")
    opp_type = opp_cfg.get("type", "unknown")
    opp_depth = opp_cfg.get("depth")

    state = engine.initial_state()
    done = False
    steps = 0
    illegal_acts = 0
    actions_list: list[dict[str, Any]] = []
    states_list: list[dict[str, Any]] = []

    # Record initial state
    states_list.append(encode_state_snapshot(state))

    while not done and steps < max_steps:
        current_player = state.current_player
        mask = legal_action_mask(engine, state)
        legal_ids = [i for i, v in enumerate(mask) if v]

        if current_player == dqn_player:
            obs = encoder(state)
            action_id = agent.select_action(obs, mask)
            if not mask[action_id]:
                illegal_acts += 1
            actor_name = agent_id or "dqn"
            actor_type = "dqn"
            is_dqn = True
        else:
            action_id = opponent.select_action_id(engine, state, legal_ids, rng)
            actor_name = opp_name
            actor_type = opp_type
            is_dqn = False

        action_info = decode_action_info(action_id)
        turn_player_str = "P1" if current_player == Player.P1 else "P2"

        actions_list.append({
            "turn_index": steps,
            "player_id": turn_player_str,
            "action_id": action_id,
            "action_type": action_info["action_type"],
            "x": action_info["x"],
            "y": action_info["y"],
            "is_dqn_action": is_dqn,
            "actor_name": actor_name,
            "actor_type": actor_type,
        })

        eng_action = decode_action_id(action_id, current_player)
        state = engine.apply_action(state, eng_action)
        steps += 1
        done = engine.is_game_over(state)

        # Record state after action
        states_list.append(encode_state_snapshot(state))

    # Determine outcome
    termination = "terminal" if done else "timeout"
    winner_obj = engine.winner(state) if done else None

    if winner_obj is None:
        winner_str = None
        result = "draw"
    elif winner_obj == Player.P1:
        winner_str = "P1"
        result = "win" if dqn_player == Player.P1 else "loss"
    else:
        winner_str = "P2"
        result = "win" if dqn_player == Player.P2 else "loss"

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return GameReplay(
        schema_version=REPLAY_SCHEMA_VERSION,
        run_id=run_id,
        agent_id=agent_id,
        episode=episode,
        checkpoint=checkpoint,
        eval_opponent={"name": opp_name, "type": opp_type, "depth": opp_depth},
        game_index=game_index,
        dqn_player_id=player_id_str,
        opponent_player_id=opp_player_id_str,
        winner=winner_str,
        result_from_dqn_perspective=result,
        game_length=steps,
        illegal_acts=illegal_acts,
        termination_reason=termination,
        created_at=ts,
        actions=actions_list,
        states=states_list,
    )


def write_replay(replay: GameReplay, path: Path) -> None:
    """Write a GameReplay to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(replay.to_dict(), fh, separators=(",", ":"))


def replay_filename(episode: int, opponent_name: str, game_index: int) -> str:
    """Return the canonical filename for a replay file."""
    # Sanitise opponent_name: keep alphanumeric, _ and -
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in opponent_name)
    return f"eval_ep{episode:05d}_{safe_name}_game{game_index:04d}.json"


def replay_index_entry(
    replay: GameReplay,
    replay_path_rel: str,
) -> dict[str, Any]:
    """Return a single-line dict for appending to eval_replays.jsonl."""
    return {
        "type": "eval_replay",
        "episode": replay.episode,
        "checkpoint": replay.checkpoint,
        "opponent_name": replay.eval_opponent.get("name"),
        "opponent_type": replay.eval_opponent.get("type"),
        "opponent_depth": replay.eval_opponent.get("depth"),
        "game_index": replay.game_index,
        "dqn_player_id": replay.dqn_player_id,
        "winner": replay.winner,
        "result_from_dqn_perspective": replay.result_from_dqn_perspective,
        "game_length": replay.game_length,
        "illegal_acts": replay.illegal_acts,
        "replay_path": replay_path_rel,
        "created_at": replay.created_at,
    }
