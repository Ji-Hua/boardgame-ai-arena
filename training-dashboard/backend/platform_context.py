"""Read-only, evidence-addressed Decision Lab contexts.

Ply is a state offset: states[ply] is BEFORE actions[ply]. Replay paths are
opaque provider keys, never local paths here. Rust alone verifies transitions.
No checkpoint is deserialized and no analysis/search is performed.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypedDict
from urllib.parse import quote, urlencode


class AnalysisContext(TypedDict):
    schema_version: str
    position: dict[str, Any]
    position_hash: str
    source: dict[str, Any]
    actual_action: dict[str, Any] | None
    recorded_model: dict[str, Any] | None
    analysis_model: dict[str, Any] | None
    return_route: str
    warnings: list[str]


def canonical_hash(value: Any) -> str:
    """SHA-256 of UTF-8 JSON, sorted keys, compact separators, no NaN."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _seat(value: Any) -> int:
    if type(value) is int and value in (1, 2):
        return value
    if isinstance(value, str) and value in ("P1", "P2"):
        return int(value[1])
    raise ValueError("Missing or invalid explicit replay turn")


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Model identity must be a full lowercase SHA-256")
    return value


def _model(index, digest: str) -> dict:
    digest = _sha(digest)
    model = index.detail(digest, verify_sha=True)
    if model.get("sha256") != digest or model.get("id") != digest:
        raise ValueError("Model index returned a different exact identity")
    if not model.get("analyzable") or not model.get("sha_verified"):
        raise ValueError("Selected model is unavailable, unsupported, conflicting, or SHA-unverified")
    # Static eligibility is not a claim that the isolated Lab loaded the model.
    # A navigation envelope carries identity/capability, not catalog paths or
    # the entire evidence index. Large provenance remains at the Models API.
    return {key: model[key] for key in (
        "id", "sha256", "aliases", "architecture", "output_semantics",
        "sha_verified", "analyzable", "actual_load_verified",
    ) if key in model}


def evaluations(service, run_id: str) -> dict:
    provider = service.provider_for(run_id)
    benchmarks = provider.benchmarks(run_id)
    replays = provider.replays(run_id)
    if benchmarks["run_id"] != replays["run_id"]:
        raise ValueError("Evaluation provider run identity mismatch")
    return dict(run_id=benchmarks["run_id"], benchmarks=benchmarks["benchmarks"],
                replays=replays["replays"], warnings=list(dict.fromkeys(
                    benchmarks.get("warnings", []) + replays.get("warnings", []))))


def _trajectory(replay: dict, ply: int):
    # Lazy import: listing models/evaluations must not require neural or Rust imports.
    from agent_system.analysis.state import ENGINE, Move, Position, build_state, to_position

    states, actions = replay.get("states"), replay.get("actions")
    if (not isinstance(states, list) or not states or not isinstance(actions, list)
            or len(states) != len(actions) + 1
            or type(replay.get("game_length")) is not int or replay["game_length"] != len(actions)):
        raise ValueError("Incomplete replay trajectory")
    if type(ply) is not int or not 0 <= ply < len(states):
        raise ValueError("Invalid ply: expected a state offset from 0 through game_length")
    initial = states[0].get("current_player", replay.get("initial_turn"))
    if initial is None and actions:
        # An actual first action's actor is explicit turn evidence, unlike parity.
        initial = actions[0].get("player_id")
    initial = _seat(initial)

    def position(snapshot, turn):
        if "current_player" in snapshot and _seat(snapshot["current_player"]) != turn:
            raise ValueError("Replay current_player disagrees with Rust transition")
        walls = [dict(x=x, y=y, orientation=orientation)
                 for key, orientation in (("h_walls", "horizontal"), ("v_walls", "vertical"))
                 for x, y in snapshot[key]]
        walls.sort(key=lambda wall: (wall["orientation"], wall["x"], wall["y"]))
        return Position(pawns=(snapshot["p1"], snapshot["p2"]),
                        walls_remaining=(snapshot["walls_remaining_p1"], snapshot["walls_remaining_p2"]),
                        walls=walls, turn=turn).model_dump(mode="json")

    first = position(states[0], initial)
    state = build_state(Position.model_validate(first))
    canonical_states, wire_actions = [first], []
    for offset, action in enumerate(actions):
        kind = {"pawn": "pawn", "hwall": "horizontal", "vwall": "vertical"}.get(action.get("action_type"))
        move = Move(type=kind, player=_seat(action.get("player_id")), target=(action["x"], action["y"]))
        wire = move.model_dump(mode="json")
        if ENGINE.is_game_over(state):
            raise ValueError("Recorded action after engine goal")
        state = ENGINE.apply_action(state, move.engine_action())
        actual = to_position(state)
        actual["walls"].sort(key=lambda wall: (wall["orientation"], wall["x"], wall["y"]))
        if actual != position(states[offset + 1], actual["turn"]):
            raise ValueError("Replay snapshot disagrees with Rust transition")
        canonical_states.append(actual)
        wire_actions.append(wire)
    return canonical_states, wire_actions


def analysis_context(service, index, *, run_id: str | None = None,
                     replay_path: str | None = None, ply: int = 0,
                     model_sha: str | None = None) -> AnalysisContext:
    """Resolve exact model and original replay evidence, failing closed on ambiguity."""
    selected = _model(index, model_sha) if model_sha is not None else None
    if replay_path is None:
        if run_id is not None or ply != 0:
            raise ValueError("A run context requires an explicit replay_path; ply requires a replay")
        from agent_system.analysis.state import ENGINE, to_position
        position = to_position(ENGINE.initial_state())
        return AnalysisContext(schema_version="analysis_context_v1", position=position,
            position_hash=canonical_hash(position), source={"type": "standard_initial", "ply": 0},
            actual_action=None, recorded_model=None, analysis_model=selected,
            return_route=f"/models/{model_sha}" if model_sha else "/models", warnings=[])
    if not run_id:
        raise ValueError("replay_path requires run_id")
    provider = service.provider_for(run_id)
    # Always use the existing provider path checks and original detail reader.
    replay = provider.replay(run_id, replay_path)
    listing = provider.replays(run_id)
    matches = [row for row in listing["replays"] if row["replay_path"] == replay_path]
    if not matches:
        raise ValueError("Replay has no exact indexed identity")
    # Duplicate identical entries are benign; contradictory evidence is not.
    identity_keys = ("checkpoint_sha", "evaluation_id", "game_id", "game_index", "checkpoint", "episode", "step")
    identities = {canonical_hash({key: row.get(key) for key in identity_keys}) for row in matches}
    if len(identities) != 1:
        raise ValueError("Ambiguous duplicate replay index identity")
    meta = matches[0]
    for key in ("game_index", "checkpoint", "episode", "step"):
        if meta.get(key) is not None and replay.get(key) is not None and meta[key] != replay[key]:
            raise ValueError(f"Replay/index identity mismatch: {key}")
    canonical_run = listing["run_id"]
    source_type = getattr(provider, "source", None)
    if source_type not in ("legacy", "mlflow"):
        raise ValueError("Unsupported run provider source")
    states, actions = _trajectory(replay, ply)
    evidence = replay.get("source", {})
    if not isinstance(evidence, dict):
        raise ValueError("Malformed replay source evidence")
    if evidence.get("game_index") is not None and evidence["game_index"] != replay.get("game_index"):
        raise ValueError("Replay/source identity mismatch: game_index")
    recorded_shas = [value for value in (evidence.get("checkpoint_sha256"),
        replay.get("checkpoint_sha256"), replay.get("checkpoint_sha"), meta.get("checkpoint_sha")) if value is not None]
    for value in recorded_shas:
        _sha(value)
    if len(set(recorded_shas)) > 1:
        raise ValueError("Recorded checkpoint SHA evidence mismatch")
    recorded_sha = recorded_shas[0] if recorded_shas else None
    recorded = {"sha256": recorded_sha} if recorded_sha else None
    if selected is None and recorded_sha:
        selected = _model(index, recorded_sha)
    warnings = list(dict.fromkeys(listing.get("warnings", []) + replay.get("warnings", [])))
    if not recorded:
        warnings.append("Recorded checkpoint SHA unavailable; never inferred from checkpoint names or paths.")
    run_ref = dict(source=source_type, run_id=canonical_run)
    manifest_sha = evidence.get("manifest_sha256")
    if manifest_sha is not None:
        _sha(manifest_sha)
    explicit_eval = meta.get("evaluation_id")
    evaluation_ref = None
    if manifest_sha or explicit_eval is not None:
        eval_evidence = ({"manifest_sha256": manifest_sha} if manifest_sha else
                         {"run": run_ref, "evaluation_id": explicit_eval})
        evaluation_ref = dict(id=canonical_hash(eval_evidence), evaluation_id=explicit_eval,
                              manifest_sha256=manifest_sha)
    else:
        warnings.append("Evaluation identity unavailable; no filename/name association was guessed.")
    # Content identity, not a filename, agent display name, or provider game_id slug.
    game_evidence = dict(states=states, actions=actions)
    game_ref = dict(id=canonical_hash(game_evidence), game_index=replay.get("game_index"),
                    trajectory_hash=evidence.get("trajectory_hash"), opening_hash=evidence.get("opening_hash"))
    position = states[ply]
    source = dict(type="recorded_replay", run=run_ref, evaluation=evaluation_ref, game=game_ref,
                  replay_path=replay_path, ply=ply, evidence=evidence)
    query = urlencode({"replay": replay_path, "ply": ply, **({"model_sha": model_sha} if model_sha else {})})
    return AnalysisContext(schema_version="analysis_context_v1", position=position,
        position_hash=canonical_hash(position), source=source,
        actual_action=actions[ply] if ply < len(actions) else None,
        recorded_model=recorded, analysis_model=selected,
        return_route=f"/replays/{quote(canonical_run, safe='')}?{query}", warnings=warnings)