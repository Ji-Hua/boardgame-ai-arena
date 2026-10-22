"""Read-only Arena evidence conversion; no agents or model weights are loaded.

Runtime needs the Rust ``quoridor_engine`` wheel and repository sources for
engine/game_manager, the backend engine adapter, arena scorers/core, and DQN
replay_writer/action_space. None of these imports requires PyTorch. The adapter
is loaded by filename because both dashboard and game server are named backend;
putting backend-server first on PYTHONPATH would shadow the dashboard package.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def _stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=1)
def _dependencies():
    # Reuse the real wire adapter without modifying backend.__path__ or loading
    # the application, agent loader, matrix worker, or neural inference stack.
    root = Path(os.environ.get("TRAINING_DASHBOARD_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
    path = root / "backend-server/backend/adapters/engine_adapter.py"
    spec = importlib.util.spec_from_file_location("_dashboard_replay_engine_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    from quoridor_engine import Player, RuleEngine
    from agent_system.evaluation.arena.agents.scorers import _build_state
    from agent_system.training.dqn.replay_writer import encode_state_snapshot
    return adapter, Player, RuleEngine, _build_state, encode_state_snapshot


def arena_replay(manifest, block, game_index, agent_id, run_id, checkpoint):
    """Validate immutable evidence and reconstruct every state with RuleEngine.

    Invalid or incomplete evidence raises ValueError, never a partial replay.
    Opening actions are checked separately and are not evaluated game actions.
    Outcome metadata remains authoritative for non-goal terminations (forfeits,
    cycles, ply limits); these do not require an engine goal winner.
    """
    try:
        return _arena_replay(manifest, block, game_index, agent_id, run_id, checkpoint)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Malformed Arena replay evidence: {exc}") from exc


def _arena_replay(manifest, block, game_index, agent_id, run_id, checkpoint):
    digest = manifest.get("manifest_sha256")
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Missing manifest SHA256")
    _require(digest == block.get("manifest_sha256"), "Block manifest SHA mismatch")
    _require(_stable_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == digest,
             "Manifest SHA mismatch")
    roster = manifest["roster"]["agents"]
    ids = [a["id"] for a in roster]
    _require(len(ids) == len(set(ids)), "Duplicate roster agent identity")
    specs = {a["id"]: a for a in roster}
    _require(agent_id in specs, "Agent identity absent from manifest roster")
    sha = checkpoint.get("sha256")
    _require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), "Missing checkpoint SHA256")
    _require(specs[agent_id]["algo"]["params"].get("sha256") == sha,
             "Checkpoint SHA does not match exact roster agent identity")
    pair = [block["a"], block["b"]]
    _require(pair[0] != pair[1] and all(i in specs for i in pair) and agent_id in pair,
             "Invalid block agent identities")
    _require(type(game_index) is int and 0 <= game_index < len(block["games"]), "Invalid game index")
    game = block["games"][game_index]
    seats = [game["p1"], game["p2"]]
    _require(sorted(seats) == sorted(pair), "Game seats do not match block identities")
    actions = game["actions"]
    _require(isinstance(actions, list) and type(game["steps"]) is int and len(actions) == game["steps"],
             "Missing or incomplete recorded actions/steps")

    adapter, Player, RuleEngine, build_state, snapshot = _dependencies()
    engine = RuleEngine.standard()

    def apply(state, action):
        _require(not engine.is_game_over(state), "Recorded action after engine goal")
        seat = 1 if state.current_player == Player.P1 else 2
        _require(type(action["player"]) is int and action["player"] == seat,
                 "Recorded action player does not match engine current player")
        target = action["target"]
        _require(isinstance(target, list) and len(target) == 2 and all(type(x) is int for x in target),
                 "Invalid recorded action target")
        return engine.apply_action(state, adapter._action_to_engine(action))

    opening_id = block["opening_id"]
    _require(type(opening_id) is int and type(block["control"]) is bool, "Invalid opening identity")
    if block["control"]:
        _require(opening_id == -1, "Control must use standard opening")
        state = engine.initial_state()
        opening_hash = _stable_hash(adapter._serialize_state(state))
    else:
        _require(opening_id >= 0, "Non-control opening must be in book")
        openings = [o for o in manifest["book"] if o["id"] == opening_id]
        _require(len(openings) == 1, "Opening identity missing or ambiguous in manifest book")
        opening = openings[0]
        opening_hash = _stable_hash(opening["state"])
        _require(opening_hash == opening["state_hash"], "Opening state hash mismatch")
        state = build_state(opening["state"])
        _require(adapter._serialize_state(state) == opening["state"], "Opening state wire mismatch")
        # If recorded, verify the prefix too, but never prepend it to the game.
        if "actions" in opening:
            _require(isinstance(opening["actions"], list), "Invalid opening actions")
            prefix = engine.initial_state()
            for action in opening["actions"]:
                prefix = apply(prefix, action)
            _require(adapter._serialize_state(prefix) == opening["state"], "Opening actions/state mismatch")
    _require(opening_hash == block["opening_hash"], "Block opening hash mismatch")
    _require(_stable_hash([opening_hash, actions]) == game["trajectory_hash"], "Trajectory hash mismatch")

    def encode(state):
        return {**snapshot(state), "current_player": 1 if state.current_player == Player.P1 else 2}

    states = [encode(state)]
    wire_actions = []
    selected_seat = seats.index(agent_id) + 1
    for index, action in enumerate(actions):
        state = apply(state, action)
        seat = action["player"]
        actor = seats[seat - 1]
        wire = dict(turn_index=index, player_id=f"P{seat}", is_dqn_action=seat == selected_seat,
                    actor_name=actor, actor_type=specs[actor]["algo"]["type"],
                    action_type={"pawn": "pawn", "horizontal": "hwall", "vertical": "vwall"}[action["type"]],
                    x=action["target"][0], y=action["target"][1])
        if "action_id" in action:
            wire["action_id"] = action["action_id"]
        wire_actions.append(wire)
        states.append(encode(state))

    final = adapter._serialize_state(state)
    winner = engine.winner(state)
    final["game_over"] = engine.is_game_over(state)
    final["winner"] = None if winner is None else (1 if winner == Player.P1 else 2)
    recorded_final = game["final_state"]
    for key in ("current_player", "pawns", "walls_remaining", "wall_state"):
        _require(recorded_final[key] == final[key], f"Final state mismatch: {key}")
    for key in ("game_over", "winner"):
        if key in recorded_final:
            _require(recorded_final[key] == final[key], f"Final state mismatch: {key}")
    outcome = game["winner"]
    _require(outcome is None or outcome in seats, "Recorded winner is not a game participant")
    termination = game["termination"]
    _require(isinstance(termination, str) and bool(termination), "Missing termination reason")
    if winner is not None or termination == "goal":
        _require(winner is not None and outcome == seats[final["winner"] - 1] and termination == "goal",
                 "Recorded goal winner disagrees with engine")
    policy_progress = checkpoint.get("progress_type") in ("gradient_updates", "gradient_update", "update", "updates")
    return dict(
        schema_version="eval_replay_v1", run_id=run_id, agent_id=agent_id,
        episode=None if policy_progress else checkpoint.get("episode"), checkpoint=checkpoint["filename"],
        step=checkpoint.get("step"), progress_type=checkpoint.get("progress_type"),
        eval_opponent={"name": seats[2 - selected_seat], "type": specs[seats[2 - selected_seat]]["algo"]["type"],
                       "depth": specs[seats[2 - selected_seat]]["algo"].get("params", {}).get("depth")},
        game_index=game_index, dqn_player_id=f"P{selected_seat}", opponent_player_id=f"P{3 - selected_seat}",
        winner=None if outcome is None else f"P{seats.index(outcome) + 1}",
        result_from_dqn_perspective="draw" if outcome is None else ("win" if outcome == agent_id else "loss"),
        game_length=game["steps"], illegal_acts=int(termination == "illegal_action"),
        termination_reason=termination, created_at=manifest.get("created_at"),
        actions=wire_actions, states=states,
        source={"type": "arena_recorded_actions", "manifest_sha256": digest,
                "checkpoint_sha256": sha, "artifact_ref": checkpoint.get("artifact_ref"),
                "model_type": checkpoint.get("model_type"), "block_id": block.get("block_id"),
                "opening_id": opening_id, "opening_hash": opening_hash, "control": block["control"],
                "trajectory_hash": game["trajectory_hash"], "game_index": game_index,
                "agent_id": agent_id, "source_sha256": manifest.get("source_sha256"),
                "git_head": manifest.get("git_head"), "offender": game.get("offender"), "error": game.get("error")})