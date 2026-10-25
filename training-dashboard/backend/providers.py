"""Read-only, provider-neutral training views. Identity is never inferred from names.

Only configured local roots are readable. MLflow artifact references are provenance,
not download instructions; historical host paths are translated into mounted roots.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import discovery, mlflow
from .config import get_artifact_roots
from .models import (BenchmarkResult, DiagnosisItem, DiagnosisReport, EvalReplayMeta,
                     MetricPoint, MetricSeries, RunSummary)


def plain(value: Any) -> Any:
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric_metadata(key: str, algorithm: str, step_type: str | None) -> dict:
    algo = algorithm.lower()
    known = {
        "policy_cross_entropy": ("Policy cross-entropy", "nats", "policy_cross_entropy", "training_batch"),
        "training_batch_accuracy": ("Training batch accuracy", "fraction", "action_agreement", "training_batch"),
        "strict_action_agreement": ("Strict action agreement", "fraction", "action_agreement", "heldout"),
        "equivalent_optimal_action_rate": ("Equivalent optimal action agreement", "fraction", "equivalent_optimal_action_agreement", "heldout"),
        "cross_entropy": ("Cross-entropy", "nats", "policy_cross_entropy", "training_batch"),
        "max_abs_logit": ("Maximum absolute logit", "logit", "max_abs_logit", "training_batch"),
        "epsilon": ("Exploration epsilon", "fraction", "exploration_probability", "training"),
        "avg_reward": ("Average reward", "reward", "episode_reward", "training"),
        "avg_q_max": ("Average maximum Q-value", "Q-value", "action_value", "training"),
        "buffer_size": ("Replay buffer size", "transitions", "replay_buffer_size", "training"),
        "opt_steps": ("Optimizer steps", "updates", "optimizer_steps", "training"),
    }
    if key in ("loss", "avg_loss", "final_avg_loss") and "dqn" in algo:
        label, unit, semantic, scope = "TD loss", "loss", "td_loss", "training"
    elif key in known:
        label, unit, semantic, scope = known[key]
    else:
        label, unit, semantic, scope = f"{algorithm}: {key}", None, f"{algo}:{key}", "unknown"
    return dict(label=label, unit=unit, semantic=semantic, scope=scope, step_type=step_type)


def annotate(series: MetricSeries, algorithm: str, step_type: str | None = None,
             reward_mode: str | None = None) -> MetricSeries:
    for point in series.points:
        point.values = {key: value if finite(value) else None for key, value in point.values.items()}
        updates = point.values.get("opt_steps")
        if point.gradient_updates is None and finite(updates) and updates >= 0 and int(updates) == updates:
            point.gradient_updates = int(updates)
    axes = [("episode", "Episode"), ("step", "Step"), ("update", "Gradient updates"),
            ("gradient_updates", "Gradient updates"), ("timestamp", "Timestamp")]
    series.axes = [{"key": key, "label": label} for key, label in axes
                   if not (key == "update" and any(p.gradient_updates is not None for p in series.points))
                   if any(getattr(p, key) is not None for p in series.points)]
    series.metadata = {key: metric_metadata(key, algorithm, step_type) for key in series.columns}
    for metadata in series.metadata.values():
        if metadata["semantic"] == "episode_reward":
            metadata["reward_mode"] = reward_mode
            metadata["semantic"] = (f"episode_reward:{reward_mode}" if reward_mode else
                                    f"episode_reward:unknown:{algorithm.lower()}:{series.run_id}")
    for snapshot in series.snapshots:
        snapshot["metadata"] = {key: {**metric_metadata(key, algorithm, None), "scope": snapshot.get("scope", "unknown")}
                                for key, value in snapshot.get("values", {}).items() if finite(value)}
    return series


class LocalEvidence:
    def __init__(self, roots: list[str] | None = None):
        self.data_root = Path(os.environ.get("QUORIDOR_DATA_ROOT", "/data/quoridor")).resolve()
        self.roots = [self.data_root, *(Path(p).resolve() for p in (roots or get_artifact_roots()))]

    def path(self, raw: str | Path) -> Path:
        text = str(raw)
        if "://" in text or text.startswith(("file:", "mlflow-artifacts:")):
            raise ValueError("Remote/URI evidence is not readable")
        path = Path(text)
        # Translate historical absolute paths before checking realpath containment.
        for prefix in ("/home/jihua/quoridor-data", "/quoridor-data", "/data/quoridor"):
            if text == prefix or text.startswith(prefix + "/"):
                path = self.data_root / text[len(prefix):].lstrip("/")
                break
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in self.roots):
            raise ValueError("Evidence outside configured local roots")
        return resolved

    def json(self, raw: str | Path) -> Any:
        return json.loads(self.path(raw).read_text())

    def optional_json(self, raw: str | Path) -> dict:
        try:
            value = self.json(raw)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}


class LegacyAdapter:
    source = "legacy"

    def __init__(self, roots: list[str] | None = None):
        self.roots = roots if roots is not None else get_artifact_roots()

    @staticmethod
    def _safe_tree(path: Path) -> bool:
        # Parsers use several fixed files and evaluation subdirectories. Reject
        # outward symlinks before any of those legacy readers can follow them.
        try:
            return all(not p.is_symlink() or p.resolve().is_relative_to(path) for p in path.rglob("*"))
        except (OSError, ValueError):
            return False

    def find(self, run_id: str) -> Path | None:
        if not run_id or run_id in (".", "..") or Path(run_id).name != run_id or "\\" in run_id:
            return None
        for raw in self.roots:
            root = Path(raw).resolve()
            candidate = (root / run_id).resolve()
            if candidate.is_relative_to(root) and candidate.is_dir() and self._safe_tree(candidate):
                return candidate
        return None

    def require(self, run_id: str) -> Path:
        found = self.find(run_id)
        if found is None:
            raise KeyError(run_id)
        return found

    def listing(self) -> tuple[list[dict], list[str]]:
        runs, warnings, seen = [], [], set()
        for raw in self.roots:
            root = Path(raw).resolve()
            if not root.is_dir():
                continue
            for candidate in sorted(root.iterdir()):
                path = self.find(candidate.name)
                if path is None or path in seen:
                    continue
                seen.add(path)
                if not ((path / "train.log").is_file() or (path / "checkpoints").is_dir()):
                    continue
                try:
                    run = discovery.discover_run(path)
                    if run:
                        runs.append(run)
                except (OSError, ValueError, TypeError) as exc:
                    warnings.append(f"Legacy run {path.name} unavailable: {exc}")
        if not runs:
            warnings.append("No legacy training runs discovered in configured artifact roots.")
        result = []
        for run in runs:
            path = self.find(run.run_id)
            if path is None:
                warnings.append(f"Skipped unsafe legacy run {run.run_id}")
                continue
            run.run_name = run.run_id
            run.provenance = {"source_path": str(path)}
            result.append(plain(run))
        return result, warnings

    def detail(self, run_id: str) -> dict:
        path = self.require(run_id)
        detail = discovery.load_run_detail(path)
        if detail is None:
            raise KeyError(run_id)
        detail.run_name = run_id
        detail.provenance = {"source_path": str(path)}
        for ck in detail.checkpoints:
            ck.model_type = detail.model_arch
            ck.artifact_ref = ck.path
            ck.progress_type = "env_steps" if ck.step is not None else "episode"
        return plain(detail)

    def metrics(self, run_id: str) -> dict:
        path = self.require(run_id)
        detail = discovery.load_run_detail(path)
        # detail.reward_mode may have been guessed from the run name. Only recorded
        # configuration/summary evidence permits comparison of shaped rewards.
        reward_mode = (detail.config.get("reward_mode") or detail.summary_stats.get("reward_mode")) if detail else None
        return plain(annotate(discovery.load_run_metrics(path), detail.algorithm if detail else "DQN", "env_steps", reward_mode))

    def benchmarks(self, run_id: str) -> dict:
        path = self.require(run_id)
        rows = discovery.load_run_benchmarks(path)
        replays = self.replays(run_id)["replays"]
        checkpoints = discovery._parse_checkpoints(path / "checkpoints")
        for row in rows:
            if row.checkpoint is None and row.checkpoint_episode is not None and row.checkpoint_step is not None:
                # Both progress coordinates must agree; episode alone is not identity.
                names = {ck.filename for ck in checkpoints if ck.episode == row.checkpoint_episode
                         and ck.step == row.checkpoint_step}
                names.update(r["checkpoint"] for r in replays if r["checkpoint"]
                             and r["episode"] == row.checkpoint_episode and r["step"] == row.checkpoint_step)
                if len(names) == 1:
                    row.checkpoint = names.pop()
            row.evaluation_id = row.extra.get("evaluation_id")
            row.protocol = row.extra.get("protocol")
            row.checkpoint_sha = row.extra.get("checkpoint_sha256") or row.extra.get("checkpoint_sha")
            row.score_rate = row.extra.get("score_rate", row.extra.get("score"))
            if row.games and row.wins is not None:
                row.win_rate = row.wins / row.games
            row.progress_type = "env_steps" if row.checkpoint_step is not None else "episode"
            row.replay_available = any(r["replay_available"] and
                row.checkpoint is not None and row.checkpoint == r["checkpoint"]
                and (row.checkpoint_episode is None or row.checkpoint_episode == r["episode"])
                and (row.checkpoint_step is None or row.checkpoint_step == r["step"])
                and (row.opponent is None or row.opponent == r["opponent_name"]) for r in replays)
        return {"run_id": run_id, "benchmarks": [plain(r) for r in rows], "count": len(rows)}

    def diagnosis(self, run_id: str) -> dict:
        return plain(discovery.diagnose_run(self.require(run_id)))

    def replays(self, run_id: str) -> dict:
        path = self.require(run_id)
        rows = discovery.load_eval_replays(path)
        warnings = []
        for row in rows:
            match = discovery._CKPT_RE.fullmatch(Path(row.checkpoint).name) if isinstance(row.checkpoint, str) else None
            if match:
                episode, step = map(int, match.groups())
                if row.episode in (None, episode) and row.step in (None, step):
                    row.checkpoint = Path(row.checkpoint).name
                    row.episode, row.step, row.progress_type = episode, step, "env_steps"
            try:
                replay = self._load_replay(path, row.replay_path, row)
                warnings.extend(replay.get("warnings", []))
                # Only an identity-checked, complete payload can replace stale
                # display metadata. Never rewrite the original index or JSON.
                for key in ("game_length", "winner", "result_from_dqn_perspective",
                            "dqn_player_id", "illegal_acts", "created_at"):
                    if key in replay:
                        setattr(row, key, replay[key])
                row.replay_available = True
                row.replay_unavailable_reason = None
            except (OSError, ValueError, TypeError) as exc:
                row.replay_available = False
                row.replay_unavailable_reason = str(exc)
        return {"run_id": run_id, "replays": [plain(r) for r in rows], "count": len(rows),
                "warnings": warnings}

    @staticmethod
    def _load_replay(path: Path, replay_path: str, indexed: EvalReplayMeta) -> dict:
        candidate = (path / replay_path).resolve()
        if not candidate.is_relative_to(path.resolve()):
            raise ValueError("Invalid replay path")
        if not candidate.is_file():
            raise ValueError("Replay file is missing")
        try:
            replay = json.loads(candidate.read_text())
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Malformed replay JSON") from exc
        if not isinstance(replay, dict):
            raise ValueError("Malformed replay object")
        # Old fixtures may omit identity, but any recorded identity must agree.
        # In particular, an overwritten payload is never reassociated to another
        # checkpoint merely because it contains a complete trajectory.
        if replay.get("run_id") is not None and replay["run_id"] != path.name:
            # discovery uses the folder name, but scripts/train_dqn.py passes
            # cfg.agent_id as BOTH run_id and agent_id to the replay writer.
            # Authorize only the historical dqn_<folder> spelling, backed by
            # this run's recorded DQN config and the payload's matching agent.
            # A prefix alone (or an arbitrary configured agent) is not identity.
            config = {}
            try:
                import yaml
                recorded = yaml.safe_load((path / "config.yaml").read_text())
                if isinstance(recorded, dict):
                    config = recorded
            except (ImportError, OSError, UnicodeError, ValueError):
                pass
            except yaml.YAMLError:
                pass
            canonical = f"dqn_{path.name}"
            if not (replay["run_id"] == canonical
                    and replay.get("agent_id") == canonical
                    and config.get("agent_id") == canonical
                    and str(config.get("algorithm", "")).lower() in ("dqn", "double_dqn", "doubledqn")):
                raise ValueError("Replay identity mismatch: run_id")
        opponent = replay.get("eval_opponent")
        if opponent is not None and not isinstance(opponent, dict):
            raise ValueError("Malformed replay opponent identity")
        identities = {key: replay.get(key) for key in ("checkpoint", "episode", "game_index")}
        identities.update({f"opponent_{key}": (opponent or {}).get(key)
                           for key in ("name", "type", "depth")})
        for key, value in identities.items():
            expected = getattr(indexed, key)
            if value is not None and expected is not None and value != expected:
                raise ValueError(f"Replay identity mismatch: {key}")
        states, actions, length = replay.get("states"), replay.get("actions"), replay.get("game_length")
        if (type(length) is not int or length < 0 or not isinstance(states, list)
                or not isinstance(actions, list) or len(actions) != length or len(states) != length + 1):
            raise ValueError("Incomplete replay trajectory: states/actions must match game_length")
        for state in states:
            if not isinstance(state, dict) or not all(key in state for key in
                    ("p1", "p2", "h_walls", "v_walls", "walls_remaining_p1", "walls_remaining_p2")):
                raise ValueError("Malformed replay state snapshot")
            positions = [state["p1"], state["p2"]]
            for key in ("h_walls", "v_walls"):
                if not isinstance(state[key], list):
                    raise ValueError("Malformed replay wall snapshots")
                positions.extend(state[key])
            if (not all(isinstance(pos, list) and len(pos) == 2 and all(type(v) is int for v in pos)
                        for pos in positions)
                    or not all(type(state[key]) is int for key in ("walls_remaining_p1", "walls_remaining_p2"))):
                raise ValueError("Malformed replay state coordinates or wall counts")
            # Preserve actual recorded turns. Old DQN snapshots omit the field;
            # leave that documented standard-opening fallback to the viewer.
            current = state.get("current_player")
            if type(current) is int and current in (1, 2):
                state["current_player"] = f"P{current}"
            elif "current_player" in state and current not in ("P1", "P2"):
                raise ValueError("Malformed replay current_player")
        if not all(isinstance(action, dict) and action.get("player_id") in ("P1", "P2")
                   and action.get("action_type") in ("pawn", "hwall", "vwall")
                   and type(action.get("x")) is int and type(action.get("y")) is int for action in actions):
            raise ValueError("Malformed replay action")
        warnings = list(replay.get("warnings", []))
        for key in ("game_length", "winner", "result_from_dqn_perspective",
                    "dqn_player_id", "illegal_acts", "created_at"):
            if key in replay and getattr(indexed, key) is not None and getattr(indexed, key) != replay[key]:
                warnings.append(f"Stale replay index metadata ({replay_path}): {key} "
                                f"index={getattr(indexed, key)!r}, payload={replay[key]!r}; "
                                "using authoritative payload; originals unchanged")
        if warnings:
            replay["warnings"] = warnings
        return replay

    def replay(self, run_id: str, replay_path: str) -> dict:
        path = self.require(run_id)
        candidate = (path / replay_path).resolve()
        if not candidate.is_relative_to(path.resolve()):
            raise ValueError("Invalid replay path")
        # A run's arbitrary config/JSON files are not replay endpoints.
        indexed = {row.replay_path: row for row in discovery.load_eval_replays(path)}
        if replay_path not in indexed:
            raise KeyError(replay_path)
        return self._load_replay(path, replay_path, indexed[replay_path])


def _iso(milliseconds: Any) -> str | None:
    if milliseconds is None:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def _sha(row: dict) -> str | None:
    values = {**row.get("params", {}), **row.get("tags", {})}
    value = values.get("checkpoint_sha256") or values.get("checkpoint_sha") or values.get("sha256")
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) else None


def _values(row: dict) -> dict:
    """Some historical loggers JSON-encoded scalar params; tags stay authoritative."""
    result = dict(row.get("params", {}))
    for key, value in list(result.items()):
        if isinstance(value, str):
            try:
                result[key] = json.loads(value)
            except ValueError:
                pass
    result.update(row.get("tags", {}))
    return result


class MLflowAdapter:
    source = "mlflow"

    def __init__(self, roots: list[str] | None = None, ttl: float = 15):
        self.local = LocalEvidence(roots)
        self.ttl = ttl
        self._expires = 0.0
        self._lock = threading.RLock()
        self._groups: dict[str, list[dict]] = {}
        self._aliases: dict[str, str] = {}
        self._warnings: list[str] = []
        self._evidence_cache: dict[str, dict] = {}
        self._arena_cache: dict[str, list[dict]] = {}
        self._replay_cache: dict[str, tuple[list[dict], dict]] = {}
        self._catalogs: list[tuple[Path, dict]] = []

    @staticmethod
    def _parent(row: dict) -> str | None:
        return row.get("tags", {}).get("mlflow.parentRunId") or row.get("tags", {}).get("parent")

    @staticmethod
    def _training(row: dict) -> bool:
        return _values(row).get("run_type") in ("training", "train", "trainer")

    def _checkpoint_child(self, row: dict) -> bool:
        if self._training(row):
            return False
        values = _values(row)
        raw = values.get("source_path") or values.get("source") or values.get("checkpoint")
        return bool(self._parent(row) and (_sha(row) or
                    (isinstance(raw, str) and Path(raw).suffix in (".pt", ".pth")) or
                    values.get("run_type") in ("checkpoint", "registration")))

    @staticmethod
    def _evaluation(row: dict) -> bool:
        values = {**row.get("params", {}), **row.get("tags", {})}
        return bool(row.get("_evaluation") or values.get("benchmark_run_id") or values.get("protocol") or values.get("holdout_sha256") or
                    values.get("run_type") in ("evaluation", "benchmark", "arena"))

    def _explicit_source(self, values: dict) -> Path | None:
        raw = values.get("source_path") or values.get("training_experiment") or values.get("source_dir") or values.get("source") or values.get("checkpoint_dir")
        if not raw:
            return None
        try:
            path = self.local.path(raw)
            if path.name == "checkpoints" and values.get("checkpoint_dir"):
                return path.parent
            return path.parent.parent if path.suffix in (".pt", ".pth") and path.parent.name == "checkpoints" else (path.parent if path.suffix in (".pt", ".pth") else path)
        except ValueError:
            return None

    def _source(self, row: dict) -> Path | None:
        values = _values(row)
        explicit = self._explicit_source(values)
        if any(values.get(k) for k in ("source_path", "training_experiment", "source_dir", "source", "checkpoint_dir")):
            return explicit
        sha = _sha(row)
        sources = set()
        for _, model in self._catalogs:
            if sha and str(model.get("sha256", "")).lower() == sha:
                source = self._catalog_source(model)
                if source:
                    sources.add(source)
        return next(iter(sources)) if len(sources) == 1 else None

    def _catalog_source(self, model: dict) -> Path | None:
        if model.get("training_experiment"):
            return self._explicit_source({"training_experiment": model["training_experiment"]})
        return self._explicit_source({"source_path": model.get("source_path") or model.get("checkpoint")})

    def _refresh(self) -> None:
        with self._lock:
            if time.monotonic() < self._expires:
                return
            rows, warnings = [], []
            try:
                experiments = mlflow.experiments()
            except Exception as exc:
                experiments = []
                warnings.append(f"MLflow unavailable: {exc}")
            for exp in experiments:
                try:
                    for row in mlflow.runs(exp["id"]):
                        rows.append({**{k: v for k, v in row.items() if k != "_evaluation"},
                                     "experiment_id": exp["id"], "experiment_name": exp["name"]})
                except Exception as exc:
                    warnings.append(f"MLflow experiment {exp['id']} unavailable: {exc}")
            by_id = {r["run_id"]: r for r in rows}
            self._catalogs = []
            for root in dict.fromkeys(self.local.roots):
                for path in sorted(root.rglob("*catalog*.json")):
                    for model in self.local.optional_json(path).get("models", []):
                        if isinstance(model, dict):
                            self._catalogs.append((path, model))
            # Classify the explicit parent graph before source-based grouping.
            def is_evaluation(row, visiting):
                if self._training(row) or row["run_id"] in visiting:
                    return False
                if self._evaluation(row):
                    return True
                parent = by_id.get(self._parent(row))
                return bool(parent and is_evaluation(parent, visiting | {row["run_id"]}))

            classifications = {r["run_id"]: is_evaluation(r, set()) for r in rows}
            for row in rows:
                row["_evaluation"] = classifications[row["run_id"]]
            parents = {self._parent(r) for r in rows}
            groups: dict[str, list[dict]] = {}
            for row in rows:
                if self._evaluation(row):
                    continue
                source = self._source(row)
                # Benchmark containers and empty orchestration parents are not training runs.
                if not source and not row.get("metrics") and row["run_id"] in parents:
                    continue
                values = {**row.get("tags", {}), **row.get("params", {})}
                if not source and not row.get("metrics") and not _sha(row) and not values.get("algorithm"):
                    continue
                parent = row.get("tags", {}).get("mlflow.parentRunId") or row.get("tags", {}).get("parent")
                parent_row = by_id.get(parent)
                if parent_row and not self._evaluation(parent_row) and (parent_row.get("metrics") or self._source(parent_row)):
                    key = f"parent:{parent}"
                elif source:
                    key = f"source:{source}"
                else:
                    key = f"run:{row['run_id']}"
                groups.setdefault(key, []).append(row)
            # Coalesce a metric-bearing parent with its checkpoint children explicitly.
            for key in list(groups):
                if key.startswith("parent:"):
                    parent_id = key.split(":", 1)[1]
                    parent_row = by_id[parent_id]
                    for other in list(groups):
                        if other != key and any(r["run_id"] == parent_id for r in groups[other]):
                            groups[key].extend(groups.pop(other))
                    if all(r["run_id"] != parent_id for r in groups[key]):
                        groups[key].insert(0, parent_row)
            self._groups, self._aliases = {}, {}
            # A registration without parent and children with a parent can still have
            # the same explicit training directory. Merge those identities too.
            changed = True
            while changed:
                changed = False
                owners = {}
                for key in list(groups):
                    if key not in groups:
                        continue
                    sources = {str(self._source(r)) for r in groups[key] if self._source(r)}
                    other = next((owners[s] for s in sources if s in owners and owners[s] != key), None)
                    if other:
                        groups[other].extend(groups.pop(key))
                        changed = True
                        break
                    for source in sources:
                        owners[source] = key
            for group in groups.values():
                group.sort(key=lambda r: (r["run_id"] not in parents, r.get("start_time") or 0, r["run_id"]))
                canonical = group[0]["run_id"]
                self._groups[canonical] = group
                for row in group:
                    self._aliases[row["run_id"]] = canonical
            # Explicit checkpoint identity links evaluation children, never fuzzy display names.
            for group in self._groups.values():
                hashes = {_sha(r) for r in group} - {None}
                ids = {r["run_id"] for r in group}
                related = [r["run_id"] for r in rows if self._evaluation(r) and
                           (_sha(r) in hashes or (r.get("tags", {}).get("mlflow.parentRunId") or r.get("tags", {}).get("parent")) in ids
                            or (self._source(r) is not None and self._source(r) in {self._source(g) for g in group}))]
                group[0]["related_evaluation_runs"] = related
            self._warnings = warnings
            self._evidence_cache.clear()
            self._arena_cache.clear()
            self._replay_cache.clear()
            self._expires = time.monotonic() + self.ttl

    def resolve(self, run_id: str) -> tuple[str, list[dict]]:
        self._refresh()
        canonical = self._aliases.get(run_id)
        if canonical is None:
            raise KeyError(run_id)
        return canonical, self._groups[canonical]

    def _evidence(self, run_id: str) -> dict:
        canonical, group = self.resolve(run_id)
        if canonical in self._evidence_cache:
            return self._evidence_cache[canonical]
        primary = group[0]
        warnings = list(self._warnings)
        source = next((self._source(r) for r in group if self._source(r)), None)
        catalogs: list[tuple[Path, dict]] = list(self._catalogs)
        dirs = [self.local.data_root]
        if source:
            dirs += [p for p in [source, *source.parents] if p.is_relative_to(self.local.data_root)]
        seen = {path for path, _ in catalogs}
        for directory in dirs:
            for path in directory.glob("*catalog*.json"):
                if path in seen:
                    continue
                seen.add(path)
                for model in self.local.optional_json(path).get("models", []):
                    catalogs.append((path, model))
        checkpoints, matches = [], []
        for row in group:
            values = _values(row)
            raw = values.get("source_path")
            sha = _sha(row)
            matched = None
            sha_sources = {self._catalog_source(m)
                           for _, m in catalogs if sha and str(m.get("sha256", "")).lower() == sha} - {None}
            if len(sha_sources) > 1:
                warnings.append(f"Ambiguous catalog source for checkpoint SHA {sha}; no source inferred for {row['run_id']}.")
            for catalog_path, model in catalogs:
                try:
                    same_path = raw and model.get("source_path") and self.local.path(raw) == self.local.path(model["source_path"])
                except ValueError:
                    same_path = False
                catalog_sha = str(model.get("sha256", "")).lower()
                if same_path and sha and catalog_sha and sha != catalog_sha:
                    warnings.append(f"Checkpoint SHA conflict for {row['run_id']}: source path catalog SHA disagrees with MLflow tag.")
                same_sha = sha is not None and sha == catalog_sha
                if same_sha:
                    catalog_source = self._catalog_source(model)
                    explicit_source = self._explicit_source(values)
                    if explicit_source and catalog_source and explicit_source != catalog_source:
                        warnings.append(f"Source provenance conflict for {row['run_id']}: catalog SHA resolves to a different training directory.")
                if matched is None and ((same_path and catalog_sha and (sha is None or sha == catalog_sha)) or same_sha):
                    matched = model
                    sha = catalog_sha
                    raw = raw or model.get("source_path") or model.get("checkpoint")
                    matches.append({"catalog": str(catalog_path), **model})
            if raw or sha:
                metadata = (matched or {}).get("metadata", {})
                model_config = metadata.get("model_config", {})
                filename = Path(raw).name if raw else f"{sha}.pt"
                update = re.search(r"update(\d+)", filename)
                episode_match = re.search(r"ep(\d+)_step(\d+)", filename)
                step = int(update.group(1)) if update else (int(episode_match.group(2)) if episode_match else metadata.get("training_step"))
                checkpoints.append(dict(filename=filename, path=raw or "", episode=None if update else (int(episode_match.group(1)) if episode_match else None),
                                        step=step, modified_at=metadata.get("created_at"), has_benchmark=False, best_win_rate=None,
                                        sha256=sha, model_type=model_config.get("model_arch") or values.get("model_arch"),
                                        artifact_ref=row.get("artifact_uri"), progress_type="gradient_updates" if update else (values.get("progress_type") or "unknown")))
        # Deduplicate repeated registry registrations without discarding distinct checkpoints.
        checkpoints = list({(c["sha256"], c["path"]): c for c in checkpoints}.values())
        config = self.local.optional_json(source / "experiment.json") if source else {}
        algorithm = (config.get("training") or {}).get("algorithm")
        if not algorithm and matches:
            algorithm = matches[0].get("metadata", {}).get("model_config", {}).get("algorithm")
        values = _values(primary)
        algorithm = algorithm or values.get("algorithm") or "unknown"
        provenance = {"mlflow_run_ids": [r["run_id"] for r in group], "related_evaluation_runs": primary.get("related_evaluation_runs", []),
                      "source_path": str(source) if source else None, "source_checkpoints": [c["path"] for c in checkpoints],
                      "catalog_matches": matches, "artifact_refs": [r.get("artifact_uri") for r in group],
                      "output_semantics": next((_values(r).get("output_semantics") for r in group if _values(r).get("output_semantics")), None)}
        for filename in ("dataset.json", "input-sha256.json", "source-manifest.json"):
            if source:
                document = self.local.optional_json(source / filename)
                if document:
                    provenance[filename] = document
        if values.get("source_path") and source is None:
            warnings.append("Source path is outside configured local roots; local evidence was not read.")
        result = dict(primary=primary, group=group, source=source, checkpoints=checkpoints, config=config,
                      algorithm=algorithm, provenance=provenance, warnings=warnings)
        self._evidence_cache[canonical] = result
        return result

    def _arenas(self, run_id: str) -> list[dict]:
        canonical, _ = self.resolve(run_id)
        if canonical in self._arena_cache:
            return self._arena_cache[canonical]
        evidence = self._evidence(canonical)
        sources = [evidence["source"], self.local.data_root / "arena"]
        if evidence["source"]:
            sources += [p for p in evidence["source"].parents if p.is_relative_to(self.local.data_root) and p != self.local.data_root]
        candidates = set()
        for base in sources:
            if base is None:
                continue
            candidates.update(base.glob("*/report.json"))
            if (base / "report.json").is_file():
                candidates.add(base / "report.json")
        arenas = []
        identities = set()
        for report_path in sorted(candidates):
            report = self.local.optional_json(report_path)
            manifest_path = report_path.parent / "manifest.json"
            manifest = self.local.optional_json(manifest_path)
            if not manifest or not report:
                continue
            if report.get("manifest_sha256") and report["manifest_sha256"] != manifest.get("manifest_sha256"):
                evidence["warnings"].append(f"Arena manifest identity mismatch: {report_path.parent.name}")
                continue
            # Validate self hash using the producer's canonical encoding.
            if manifest.get("manifest_sha256"):
                unhashed = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
                digest = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if digest != manifest["manifest_sha256"]:
                    evidence["warnings"].append(f"Arena manifest hash invalid: {report_path.parent.name}")
                    continue
            roster = manifest.get("roster", {}).get("agents", [])
            for agent in roster:
                params = agent.get("algo", {}).get("params", {})
                for checkpoint in evidence["checkpoints"]:
                    if not checkpoint["sha256"] or params.get("sha256") != checkpoint["sha256"]:
                        continue
                    identity = (manifest.get("manifest_sha256") or str(manifest_path.resolve()), agent["id"], checkpoint["sha256"])
                    if identity in identities:
                        continue
                    identities.add(identity)
                    arenas.append(dict(root=report_path.parent, report=report, manifest=manifest, agent_id=agent["id"],
                                       checkpoint=checkpoint, evaluation_id=report.get("run_id") or report_path.parent.name,
                                       slug=(hashlib.sha256(str(manifest_path).encode()).hexdigest() + "/" +
                                           hashlib.sha256(agent["id"].encode()).hexdigest() + "/" + checkpoint["sha256"]),
                                       source_manifest={"path": str(manifest_path), "sha256": manifest.get("manifest_sha256"),
                                                        "protocol": manifest.get("protocol")}))
                    break
        self._arena_cache[canonical] = arenas
        return arenas

    def listing(self) -> tuple[list[dict], list[str]]:
        self._refresh()
        result, warnings = [], list(self._warnings)
        for run_id in list(self._groups):
            try:
                e = self._evidence(run_id)
                primary, ck = e["primary"], e["checkpoints"]
                values = {**primary.get("tags", {}), **primary.get("params", {})}
                source = e["source"]
                metric_files = self._metric_files(source)
                arenas = self._arenas(run_id)
                summary = RunSummary(run_id=run_id, algorithm=e["algorithm"], reward_mode=values.get("reward_mode"),
                    model_arch=next((c["model_type"] for c in ck if c["model_type"]), values.get("model_arch")),
                    total_episodes=None, created_at=_iso(primary.get("start_time")), modified_at=_iso(primary.get("end_time") or primary.get("start_time")),
                    latest_checkpoint=ck[-1]["filename"] if ck else None, best_checkpoint=None,
                    has_metrics=bool(metric_files or any(r.get("metrics") for r in e["group"])), has_benchmarks=bool(arenas),
                    has_checkpoints=bool(ck), has_notes=False, has_eval_replays=any((a["root"] / "blocks").is_dir() for a in arenas),
                    best_benchmark_win_rate=None, best_benchmark_opponent=None, latest_benchmark_win_rate=None,
                    available_metrics=[str(p.name) for p in metric_files], available_benchmarks=[a["evaluation_id"] for a in arenas],
                    notes_preview=None, final_avg_loss=None, final_avg_reward=None, final_epsilon=None, total_env_steps=None,
                    run_name=primary.get("run_name") or primary.get("tags", {}).get("mlflow.runName") or run_id,
                    source="mlflow", experiment_id=primary.get("experiment_id"), experiment_name=primary.get("experiment_name"),
                    status=primary.get("status"), provenance=e["provenance"])
                result.append(plain(summary))
                warnings.extend(e["warnings"])
            except (OSError, ValueError, TypeError) as exc:
                warnings.append(f"MLflow run {run_id} evidence unavailable: {exc}")
        return result, list(dict.fromkeys(warnings))

    def detail(self, run_id: str) -> dict:
        canonical, _ = self.resolve(run_id)
        e = self._evidence(canonical)
        summary = next(r for r in self.listing()[0] if r["run_id"] == canonical)
        metrics = self.metrics(canonical)
        benchmarks = self.benchmarks(canonical)["benchmarks"]
        checkpoints = []
        for checkpoint in e["checkpoints"]:
            matching = [b for b in benchmarks if b["checkpoint_sha"] == checkpoint["sha256"]]
            rates = [b["win_rate"] for b in matching if b["win_rate"] is not None]
            checkpoints.append({**checkpoint, "has_benchmark": bool(matching),
                    "best_win_rate": max(rates) if rates else None})
        return {**summary, "config": e["config"] or {"params": e["primary"].get("params", {}), "tags": e["primary"].get("tags", {})},
            "checkpoints": checkpoints, "metric_files": summary["available_metrics"], "benchmark_files": summary["available_benchmarks"],
                "notes": None, "summary_stats": {"metric_snapshots": metrics["snapshots"]},
                "provenance": {**e["provenance"], "metrics": metrics["provenance"]},
                "warnings": list(dict.fromkeys(e["warnings"] + metrics["warnings"]))}

    def _metric_files(self, source: Path | None) -> list[Path]:
        if source is None:
            return []
        paths = [p for p in sorted(source.glob("*-metrics.jsonl")) if p.is_file()]
        if not paths:
            paths = [source / name for name in ("metrics.jsonl", "metrics.csv", "metrics.json", "training_log.jsonl", "train.log")
                     if (source / name).is_file()]
        if not paths and isinstance(self.local.optional_json(source / "result.json").get("metrics"), list):
            paths = [source / "result.json"]
        return [self.local.path(p) for p in paths]

    def metrics(self, run_id: str) -> dict:
        canonical, _ = self.resolve(run_id)
        e = self._evidence(canonical)
        series = MetricSeries(canonical, [], [], warnings=list(e["warnings"]))
        column_metadata = {}
        owners = [r for r in e["group"] if r.get("metrics") and not self._checkpoint_child(r)]
        key_owners = {}
        for owner in owners:
            for key in owner["metrics"]:
                key_owners.setdefault(key, []).append(owner["run_id"])
        for key, ids in key_owners.items():
            if len(ids) > 1:
                series.warnings.append(f"Multiple metric owners for {key}: {', '.join(ids)}; histories kept as separate origin-qualified columns.")
        for row in e["group"]:
            values = _values(row)
            if self._checkpoint_child(row) and row.get("metrics"):
                series.snapshots.append({"source": "mlflow", "run_id": row["run_id"],
                                         "scope": "checkpoint", "values": row["metrics"]})
                continue
            step_type = values.get("step_type") or values.get("progress_type") or "unknown"
            for key in row.get("metrics", {}):
                column = f"{key}::mlflow:{row['run_id']}" if len(key_owners.get(key, [])) > 1 else key
                owner_algorithm = values.get("algorithm") or e["algorithm"]
                metadata = metric_metadata(key, owner_algorithm, step_type)
                metadata.update(origin={"source": "mlflow", "run_id": row["run_id"], "metric_key": key})
                if metadata["semantic"] == "episode_reward":
                    reward = values.get("reward_mode")
                    metadata.update(reward_mode=reward, semantic=f"episode_reward:{reward or 'unknown:' + row['run_id']}")
                try:
                    for item in mlflow.metric_history(row["run_id"], key):
                        if not finite(item.get("value")):
                            continue
                        column_metadata[column] = metadata
                        step = item.get("step")
                        point = MetricPoint(None, step, {column: item["value"]}, timestamp=item.get("timestamp"))
                        if step_type in ("gradient_updates", "update"):
                            point.update = point.gradient_updates = step
                        series.points.append(point)
                except Exception as exc:
                    series.warnings.append(f"MLflow metric {key} unavailable: {exc}")
        if not series.points:
            files = self._metric_files(e["source"])
            # Explicit source association permits fallback, but contradictory
            # checkpoint identity must never lend credibility to local curves.
            conflicts = [w for w in e["warnings"] if "SHA conflict" in w or "Source provenance conflict" in w]
            for checkpoint in e["checkpoints"]:
                if not checkpoint["sha256"] or not checkpoint["path"]:
                    continue
                try:
                    path = self.local.path(checkpoint["path"])
                    if path.is_file() and path.suffix in (".pt", ".pth"):
                        with path.open("rb") as handle:
                            digest = hashlib.file_digest(handle, "sha256").hexdigest()
                        if digest != checkpoint["sha256"]:
                            conflicts.append(f"Checkpoint SHA conflict: local file {path} disagrees with catalog/tag.")
                except (OSError, ValueError) as exc:
                    conflicts.append(f"Checkpoint SHA verification unavailable: {exc}")
            if conflicts:
                series.warnings.extend(conflicts)
                series.warnings.append("Local training metric fallback blocked by checkpoint provenance conflict.")
                files = []
            if files:
                series.warnings.append("No MLflow metric history; using exact source checkpoint training files (registration runs may contain no metrics).")
                series.provenance = {"source": "local_training_files", "files": [str(p) for p in files],
                                     "reason": "No MLflow metric history", "source_checkpoints": e["provenance"]["source_checkpoints"]}
            for path in files:
                try:
                    if path.name == "result.json":
                        records = self.local.json(path).get("metrics", [])
                        for row in records:
                            vals = {k: v for k, v in row.items() if k not in ("episode", "step", "update", "gradient_updates", "timestamp") and finite(v)}
                            if vals:
                                series.points.append(MetricPoint(row.get("episode"), row.get("step"), vals,
                                    timestamp=row.get("timestamp"), update=row.get("update"), gradient_updates=row.get("gradient_updates", row.get("update"))))
                        continue
                    if not path.name.endswith("-metrics.jsonl"):
                        # Keep existing parsers for ordinary historical training formats.
                        for name in ("metrics.jsonl", "metrics.csv", "metrics.json", "training_log.json", "training_log.jsonl", "train.log"):
                            self.local.path(path.parent / name)
                        legacy = discovery.load_run_metrics(path.parent)
                        series.points.extend(legacy.points)
                        series.warnings.extend(legacy.warnings)
                        break
                    for number, line in enumerate(path.read_text().splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            vals = {k: v for k, v in row.items() if k not in ("episode", "step", "update", "gradient_updates", "timestamp") and finite(v)}
                            if vals:
                                series.points.append(MetricPoint(row.get("episode"), row.get("step"), vals,
                                    timestamp=row.get("timestamp"), update=row.get("update"), gradient_updates=row.get("gradient_updates", row.get("update"))))
                        except (ValueError, TypeError, AttributeError) as exc:
                            series.warnings.append(f"{path.name}:{number}: {exc}")
                except OSError as exc:
                    series.warnings.append(f"{path.name}: {exc}")
            if not files and not series.points:
                series.warnings.append("No MLflow metric history or mounted source training curve available.")
        if e["source"]:
            for path in sorted(e["source"].glob("heldout*.json")):
                snapshot = self.local.optional_json(path)
                if snapshot:
                    series.snapshots.append({"source": str(path), "scope": "heldout", "values": snapshot})
        series.columns = sorted({k for point in series.points for k in point.values})
        step_type = ("gradient_updates" if any(p.update is not None for p in series.points) else
                 "env_steps" if any(p.step is not None for p in series.points) and
                 series.provenance.get("source") == "local_training_files" else "unknown")
        values = _values(e["primary"])
        reward_mode = (values.get("reward_mode") or e["config"].get("reward_mode")
                   or (e["config"].get("training") or {}).get("reward_mode"))
        annotate(series, e["algorithm"], step_type, reward_mode)
        for key in series.columns:
            if key in column_metadata:
                series.metadata[key] = column_metadata[key]
            else:
                series.metadata[key]["origin"] = dict(series.provenance)
        series.provenance.setdefault("metric_owners", [r["run_id"] for r in owners])
        return plain(series)

    def benchmarks(self, run_id: str) -> dict:
        canonical, _ = self.resolve(run_id)
        rows = []
        replays, _ = self._replay_index(canonical)
        for arena in self._arenas(canonical):
            ck, agent = arena["checkpoint"], arena["agent_id"]
            for opponent, result in arena["report"].get("matrix", {}).get(agent, {}).items():
                if not result or not result.get("games"):
                    continue
                games, wins, draws = result["games"], result.get("wins"), result.get("draws")
                row = BenchmarkResult(canonical, ck["filename"], ck["episode"], ck["step"], opponent,
                    discovery._infer_opponent_family(opponent), discovery._infer_opponent_depth(opponent),
                    games, wins, result.get("losses"), draws, wins / games if wins is not None else None,
                    None, None, result.get("avg_game_length"), None, None, None,
                    extra={**result, "agent_id": agent, "manifest_protocol": arena["manifest"].get("protocol")},
                    evaluation_id=arena["evaluation_id"], protocol="arena_paired",
                    score_rate=result.get("score") if result.get("score") is not None else ((wins + draws / 2) / games if wins is not None and draws is not None else None),
                    checkpoint_sha=ck["sha256"], replay_available=(arena["root"] / "blocks").is_dir(),
                    progress_type=ck["progress_type"], source_manifest=arena["source_manifest"])
                rows.append(plain(row))
            for control in arena["report"].get("standard_controls", []):
                if agent not in (control.get("a"), control.get("b")):
                    continue
                winners = control.get("winners", [])
                if not winners:
                    continue
                opponent = control["b"] if control["a"] == agent else control["a"]
                games = len(winners)
                wins = winners.count(agent)
                draws = winners.count(None)
                row = BenchmarkResult(canonical, ck["filename"], ck["episode"], ck["step"], opponent,
                    discovery._infer_opponent_family(opponent), discovery._infer_opponent_depth(opponent),
                    games, wins, games - wins - draws, draws, wins / games, None, None, None, None, None, None,
                    extra={**control, "agent_id": agent}, evaluation_id=arena["evaluation_id"], protocol="control",
                    score_rate=(wins + draws / 2) / games, checkpoint_sha=ck["sha256"],
                    replay_available=(arena["root"] / "blocks").is_dir(), progress_type=ck["progress_type"],
                    source_manifest=arena["source_manifest"])
                rows.append(plain(row))
        for row in rows:
            row["replay_available"] = any(r["replay_available"] and r["evaluation_id"] == row["evaluation_id"]
                and r["protocol"] == row["protocol"] and r["checkpoint_sha"] == row["checkpoint_sha"]
                and r["opponent_name"] == row["opponent"] for r in replays)
        return {"run_id": canonical, "benchmarks": rows, "count": len(rows),
                "warnings": self._evidence(canonical)["warnings"]}

    def diagnosis(self, run_id: str) -> dict:
        canonical, _ = self.resolve(run_id)
        series = self.metrics(canonical)
        items = [DiagnosisItem("warning", warning) for warning in series["warnings"]]
        items.append(DiagnosisItem("info", "Recorded training metrics and held-out snapshots have distinct scopes; no playing-strength inference is made from training loss."))
        if not self.benchmarks(canonical)["count"]:
            items.append(DiagnosisItem("warning", "No SHA-associated immutable Arena benchmark found."))
        return plain(DiagnosisReport(canonical, items))

    @staticmethod
    def _trajectory_reason(manifest: dict, block: dict, game: dict) -> str | None:
        if manifest.get("manifest_sha256") != block.get("manifest_sha256"):
            return "Block manifest identity mismatch"
        opening = next((o for o in manifest.get("book", []) if o.get("id") == block.get("opening_id")), None)
        # Standard controls use the engine's initial state, not an entry in book.
        # The Rust replay converter verifies that state's hash before applying actions.
        control = block.get("control") is True and block.get("opening_id") == -1
        if control:
            if not re.fullmatch(r"[0-9a-f]{64}", str(block.get("opening_hash", ""))):
                return "Missing control opening hash"
        elif opening is None or opening.get("state_hash") != block.get("opening_hash"):
            return "Opening identity/hash is not in the evaluation manifest"
        if not isinstance(game.get("actions"), list) or len(game["actions"]) != game.get("steps"):
            return "Missing or incomplete recorded actions"
        return None

    def _replay_index(self, run_id: str) -> tuple[list[dict], dict]:
        canonical, _ = self.resolve(run_id)
        if canonical in self._replay_cache:
            return self._replay_cache[canonical]
        rows, lookup = [], {}
        for arena in self._arenas(canonical):
            root, agent, ck = arena["root"], arena["agent_id"], arena["checkpoint"]
            for file in sorted((root / "blocks").glob("*.json.gz")):
                try:
                    with gzip.open(self.local.path(file), "rt") as handle:
                        block = json.load(handle)
                    if agent not in (block.get("a"), block.get("b")):
                        continue
                    block_id = file.name.removesuffix(".json.gz")
                    for index, game in enumerate(block.get("games", [])):
                        if agent not in (game.get("p1"), game.get("p2")):
                            continue
                        reason = self._trajectory_reason(arena["manifest"], block, game)
                        opponent = game["p2"] if game.get("p1") == agent else game["p1"]
                        seat = "P1" if game.get("p1") == agent else "P2"
                        winner = "P1" if game.get("winner") == game.get("p1") else ("P2" if game.get("winner") == game.get("p2") else None)
                        path = f"arena/{arena['slug']}/{block_id}/{index}"
                        meta = EvalReplayMeta(ck["episode"], ck["filename"], opponent,
                            discovery._infer_opponent_family(opponent), discovery._infer_opponent_depth(opponent), index,
                            seat, winner, "draw" if winner is None else ("win" if seat == winner else "loss"), game.get("steps"),
                            None, path, None, game_id=f"{block_id}:{index}", evaluation_id=arena["evaluation_id"],
                            protocol="control" if block.get("control") else "arena_paired", checkpoint_sha=ck["sha256"],
                            replay_available=reason is None, replay_unavailable_reason=reason, step=ck["step"], progress_type=ck["progress_type"])
                        rows.append(plain(meta))
                        lookup[path] = (arena, file, index, reason)
                except (OSError, ValueError, TypeError) as exc:
                    self._evidence(canonical)["warnings"].append(f"Replay block {file.name} unavailable: {exc}")
        self._replay_cache[canonical] = rows, lookup
        return rows, lookup

    def replays(self, run_id: str) -> dict:
        canonical, _ = self.resolve(run_id)
        rows, _ = self._replay_index(canonical)
        return {"run_id": canonical, "replays": rows, "count": len(rows), "warnings": self._evidence(canonical)["warnings"]}

    def replay(self, run_id: str, replay_path: str) -> dict:
        canonical, _ = self.resolve(run_id)
        _, lookup = self._replay_index(canonical)
        if replay_path not in lookup:
            raise KeyError(replay_path)
        arena, file, index, reason = lookup[replay_path]
        if reason:
            raise ValueError(reason)
        # Re-read the requested original block, not a regenerated game or a cached transition.
        with gzip.open(self.local.path(file), "rt") as handle:
            block = json.load(handle)
        reason = self._trajectory_reason(arena["manifest"], block, block["games"][index])
        if reason:
            raise ValueError(reason)
        from .replay import arena_replay
        return arena_replay(arena["manifest"], block, index, arena["agent_id"], canonical, arena["checkpoint"])


class CompositeAdapter:
    """One tracked identity with evidence from an exactly resolved legacy directory."""
    source = "mlflow"

    def __init__(self, tracked: MLflowAdapter, legacy: LegacyAdapter, run_id: str,
                 legacy_run_id: str, source: Path):
        self.tracked, self.legacy = tracked, legacy
        self.run_id, self.legacy_run_id, self.source_path = run_id, legacy_run_id, source

    def _validate(self, run_id: str) -> None:
        if self.tracked.resolve(run_id)[0] != self.run_id:
            raise KeyError(run_id)
        if self.legacy.require(self.legacy_run_id) != self.source_path:
            raise KeyError(self.legacy_run_id)

    @staticmethod
    def _merge(local: dict, tracked: dict) -> dict:
        result = dict(local)
        for key, value in tracked.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = CompositeAdapter._merge(result[key], value)
            elif value is not None:
                result[key] = value
        return result

    def summary(self, tracked: dict, local: dict) -> dict:
        result = self._merge(local, tracked)
        for key in ("has_metrics", "has_benchmarks", "has_checkpoints", "has_notes", "has_eval_replays"):
            result[key] = bool(tracked.get(key) or local.get(key))
        for key in ("available_metrics", "available_benchmarks"):
            result[key] = list(dict.fromkeys(tracked.get(key, []) + local.get(key, [])))
        result.update(run_id=self.run_id, source="mlflow", provenance={
            **result.get("provenance", {}), "sources": ["legacy", "mlflow"],
            "legacy_run_ids": [self.legacy_run_id]})
        return result

    def _checkpoints(self, tracked: list[dict], local: list[dict]) -> list[dict]:
        result = [dict(row) for row in tracked]

        def path(row):
            try:
                return self.tracked.local.path(row["path"]) if row.get("path") else None
            except ValueError:
                return None

        for row in local:
            matches = [i for i, other in enumerate(result)
                       if not (row.get("sha256") and other.get("sha256") and row["sha256"] != other["sha256"])
                       and ((row.get("sha256") and row["sha256"] == other.get("sha256"))
                            or (path(row) is not None and path(row) == path(other)))]
            if len(matches) != 1:
                result.append(dict(row))
                continue
            index = matches[0]
            other = result[index]
            merged = self._merge(row, other)
            merged["has_benchmark"] = bool(row.get("has_benchmark") or other.get("has_benchmark"))
            rates = [r for r in (row.get("best_win_rate"), other.get("best_win_rate")) if r is not None]
            merged["best_win_rate"] = max(rates) if rates else None
            result[index] = merged
        return result

    def detail(self, run_id: str) -> dict:
        self._validate(run_id)
        tracked = self.tracked.detail(self.run_id)
        local = self.legacy.detail(self.legacy_run_id)
        result = self.summary(tracked, local)
        for key in ("metric_files", "benchmark_files", "warnings"):
            result[key] = list(dict.fromkeys(tracked.get(key, []) + local.get(key, [])))
        result["checkpoints"] = self._checkpoints(tracked["checkpoints"], local["checkpoints"])
        result["has_checkpoints"] = bool(result["checkpoints"])
        result["notes"] = "\n\n".join(dict.fromkeys(n for n in (tracked.get("notes"), local.get("notes")) if n)) or None
        # Keep the full original config as provenance even when tracked keys win.
        result["provenance"] = {**result["provenance"], "legacy_config": local["config"]}
        return result

    def metrics(self, run_id: str) -> dict:
        self._validate(run_id)
        tracked = self.tracked.metrics(self.run_id)
        # Do not bypass the tracked adapter's checkpoint/SHA safeguards.
        blocked = any("fallback blocked" in w for w in tracked.get("warnings", []))
        if tracked["points"] or blocked:
            return tracked
        local = self.legacy.metrics(self.legacy_run_id)
        origin = {"source": "legacy", "run_id": self.legacy_run_id, "source_path": str(self.source_path)}
        return {**local, "run_id": self.run_id,
                "metadata": {key: {**value, "origin": origin} for key, value in local["metadata"].items()},
                "snapshots": tracked.get("snapshots", []) + local.get("snapshots", []),
                "warnings": list(dict.fromkeys(tracked.get("warnings", []) + local.get("warnings", []))),
                "provenance": {**tracked.get("provenance", {}), **origin,
                               "reason": "No tracked training curve; using exact legacy source"}}

    def _legacy_row(self, row: dict) -> dict:
        # Namespace evaluation identity, not protocol: an unknown protocol stays unknown.
        evaluation = f"legacy/{self.legacy_run_id}"
        if row.get("evaluation_id") is not None:
            evaluation += "/" + str(row["evaluation_id"])
        return {**row, "run_id": self.run_id, "source": "legacy", "evaluation_id": evaluation,
                "provenance": {"source": "legacy", "legacy_run_id": self.legacy_run_id,
                               "evaluation_id": row.get("evaluation_id")}}

    def benchmarks(self, run_id: str) -> dict:
        self._validate(run_id)
        tracked = self.tracked.benchmarks(self.run_id)
        local = self.legacy.benchmarks(self.legacy_run_id)
        rows = tracked["benchmarks"] + [self._legacy_row(row) for row in local["benchmarks"]]
        return {**tracked, "benchmarks": rows, "count": len(rows),
                "warnings": list(dict.fromkeys(tracked.get("warnings", []) + local.get("warnings", [])))}

    def replays(self, run_id: str) -> dict:
        self._validate(run_id)
        tracked = self.tracked.replays(self.run_id)
        local = self.legacy.replays(self.legacy_run_id)
        rows = tracked["replays"] + [{**self._legacy_row(row),
                "replay_path": f"legacy/{self.legacy_run_id}/{row['replay_path']}"} for row in local["replays"]]
        return {**tracked, "replays": rows, "count": len(rows),
                "warnings": list(dict.fromkeys(tracked.get("warnings", []) + local.get("warnings", [])))}

    def replay(self, run_id: str, replay_path: str) -> dict:
        self._validate(run_id)
        # Dispatch only exact indexed aliases; never interpret arbitrary prefixed paths.
        indexed = {row["replay_path"] for row in self.replays(run_id)["replays"]}
        if replay_path not in indexed:
            raise KeyError(replay_path)
        prefix = f"legacy/{self.legacy_run_id}/"
        if replay_path.startswith(prefix):
            return {**self.legacy.replay(self.legacy_run_id, replay_path[len(prefix):]), "run_id": self.run_id}
        return self.tracked.replay(self.run_id, replay_path)

    def diagnosis(self, run_id: str) -> dict:
        self._validate(run_id)
        tracked = self.tracked.diagnosis(self.run_id)
        local = self.legacy.diagnosis(self.legacy_run_id)
        items = tracked["items"] + [row for row in local["items"] if row not in tracked["items"]]
        return {**tracked, "items": items}


class UnifiedRunService:
    def __init__(self, roots: list[str] | None = None):
        self.legacy = LegacyAdapter(roots)
        self.mlflow = MLflowAdapter(roots)

    def provider_for(self, run_id: str):
        try:
            canonical, _ = self.mlflow.resolve(run_id)
        except KeyError:
            self.legacy.require(run_id)
            return self.legacy
        source = self.mlflow._evidence(canonical)["source"]
        composite = self._composite(canonical, source)
        return composite or self.mlflow

    def _composite(self, run_id: str, source: Path | None) -> CompositeAdapter | None:
        if source is None or self.legacy.find(source.name) != source:
            return None
        # Bind to this exact root rather than resolving the same basename elsewhere later.
        legacy = LegacyAdapter([str(source.parent)])
        return CompositeAdapter(self.mlflow, legacy, run_id, source.name, source)

    def listing(self, source: str | None = None, algorithm: str | None = None,
                experiment: str | None = None, status: str | None = None) -> dict:
        legacy, warnings = self.legacy.listing()
        remote, remote_warnings = self.mlflow.listing()
        local_by_path = {r["provenance"]["source_path"]: r for r in legacy}
        rows, consumed = [], set()
        for row in remote:
            raw = row["provenance"].get("source_path")
            composite = self._composite(row["run_id"], Path(raw) if raw else None)
            local = local_by_path.get(raw)
            if composite and local:
                row = composite.summary(row, local)
                consumed.add(raw)
            rows.append(row)
        rows += [r for r in legacy if r["provenance"]["source_path"] not in consumed]
        rows = [r for r in rows if (not source or r["source"] == source)
                and (not algorithm or r["algorithm"].lower() == algorithm.lower())
                and (not experiment or experiment in (r["experiment_id"], r["experiment_name"]))
                and (not status or (r["status"] or "").lower() == status.lower())]
        rows.sort(key=lambda r: r["modified_at"] or "", reverse=True)
        return {"runs": rows, "count": len(rows), "warnings": list(dict.fromkeys(warnings + remote_warnings)),
                "artifact_roots_scanned": self.legacy.roots}