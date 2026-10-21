"""Artifact discovery and parsing for Training Dashboard."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    BenchmarkResult,
    CheckpointInfo,
    DiagnosisItem,
    DiagnosisReport,
    EvalReplayMeta,
    MetricPoint,
    MetricSeries,
    RunDetail,
    RunSummary,
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r"ep\s+(\d+)/\d+\s*\|"
    r"\s*steps\s+(\d+)\s*\|"
    r"\s*opt_steps\s+(\d+)\s*\|"
    r"\s*eps\s+([\d.]+)\s*\|"
    r"\s*buf\s+(\d+)\s*\|"
    r"\s*avg_rew\s+([-\d.nan]+)\s*\|"
    r"\s*avg_len\s+([\d.nan]+)\s*\|"
    r"\s*avg_loss\s+([-\d.nan]+)\s*\|"
    r"\s*avg_q_max\s+([-\d.nan]+)\s*\|"
    r"\s*\+r\s+(\d+)\s*\|"
    r"\s*-r\s+(\d+)"
)

_HEADER_EP_RE = re.compile(r"DQN Training\s*[—-]+\s*(\d+) episodes")
_HEADER_OBS_RE = re.compile(r"obs_version=(\S+)")
_HEADER_DEVICE_RE = re.compile(r"device=(\S+)")
_HEADER_OBS_SIZE_RE = re.compile(r"obs_size=(\d+)")
_HEADER_ACTION_RE = re.compile(r"action_count=(\d+)")
_HEADER_BUF_RE = re.compile(r"buffer_capacity=(\d+)")
_HEADER_WARMUP_RE = re.compile(r"warmup=(\d+)")
_HEADER_BATCH_RE = re.compile(r"batch=(\d+)")
_HEADER_LR_RE = re.compile(r"lr=([\d.e+-]+)")
_HEADER_GAMMA_RE = re.compile(r"gamma=([\d.]+)")
_HEADER_EPS_RE = re.compile(r"epsilon:\s*([\d.]+)[→>]+([\d.]+) over (\d+) steps")
_HEADER_OPP_RE = re.compile(r"opponent:\s*(.+)")
_HEADER_REWARD_RE = re.compile(r"reward_mode[=:]\s*(\S+)")
_HEADER_SYNC_RE = re.compile(r"target_sync_interval[=:]\s*(\d+)")
# Device diagnostics block added in Phase 15E (new runs only)
_DIAG_REQUESTED_DEVICE_RE = re.compile(r"\[device\]\s*requested_device=(\S+)")
_DIAG_RESOLVED_DEVICE_RE = re.compile(r"\[device\]\s*resolved_device=(\S+)")
_DIAG_CUDA_NAME_RE = re.compile(r"\[device\]\s*cuda_device_name=(.+)")
# Architecture diagnostics added in Phase 16A
_HEADER_HIDDEN_LAYERS_RE = re.compile(r"hidden_layers=\[([\d,\s]+)\]")
_HEADER_PARAM_COUNT_RE = re.compile(r"parameter_count=(\d+)")
# Algorithm added in Phase 17A
_HEADER_ALGORITHM_RE = re.compile(r"algorithm=(\S+)")
# CNN model_arch and observation_shape added in Phase 18A
_HEADER_MODEL_ARCH_RE = re.compile(r"model_arch=(\S+)")
_HEADER_OBS_SHAPE_RE = re.compile(r"observation_shape=\[([\d,\s]+)\]")

_CKPT_RE = re.compile(r"ep(\d+)_step(\d+)\.pt$")

# Minimax depth inference
_MINIMAX_DEPTH_RE = re.compile(r"(?:minimax[_\s]?(?:d|depth[_\s]?)(\d+)|depth[_\s]?(\d+)|_d(\d+))", re.I)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: str | None) -> float | None:
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if f != f else f   # NaN -> None
    except (ValueError, TypeError):
        return None


def _path_mtime(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _path_ctime(path: Path) -> str | None:
    try:
        ts = path.stat().st_ctime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _infer_opponent_family(name: str | None) -> str | None:
    if not name:
        return None
    n = name.lower()
    for family in ("minimax", "greedy", "random", "dummy"):
        if family in n:
            return family
    return n.split("_")[0] if "_" in n else n


def _infer_opponent_depth(name: str | None) -> int | None:
    if not name:
        return None
    m = _MINIMAX_DEPTH_RE.search(name)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return None


def _infer_reward_mode(run_id: str, config: dict[str, Any]) -> str | None:
    rid = run_id.lower()
    if "distance_delta" in rid:
        return "distance_delta"
    if "terminal" in rid:
        return "terminal"
    if "minimax" in rid and "reward" in rid:
        return "minimax_shaping"
    return config.get("reward_mode")


def _derive_win_rate(row: dict[str, Any]) -> float | None:
    wr = row.get("win_rate")
    if wr is not None:
        return float(wr)
    wins = row.get("wins")
    games = row.get("games") or row.get("num_games")
    if wins is not None and games and games > 0:
        return round(wins / games, 4)
    return None


# ---------------------------------------------------------------------------
# Checkpoint parsing
# ---------------------------------------------------------------------------

def _parse_checkpoints(ckpt_dir: Path) -> list[CheckpointInfo]:
    result: list[CheckpointInfo] = []
    if not ckpt_dir.is_dir():
        return result
    for f in sorted(ckpt_dir.iterdir()):
        if f.suffix != ".pt":
            continue
        m = _CKPT_RE.search(f.name)
        episode = int(m.group(1)) if m else None
        step = int(m.group(2)) if m else None
        result.append(CheckpointInfo(
            filename=f.name,
            episode=episode,
            step=step,
            path=str(f),
            modified_at=_path_mtime(f),
        ))
    return result


# ---------------------------------------------------------------------------
# Train log parsing
# ---------------------------------------------------------------------------

def _parse_train_log_header(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for pattern, key, cast in [
        (_HEADER_EP_RE, "total_episodes", int),
        (_HEADER_OBS_RE, "obs_version", str),
        (_HEADER_DEVICE_RE, "device", str),
        (_HEADER_OBS_SIZE_RE, "obs_size", int),
        (_HEADER_ACTION_RE, "action_count", int),
        (_HEADER_BUF_RE, "buffer_capacity", int),
        (_HEADER_WARMUP_RE, "warmup", int),
        (_HEADER_BATCH_RE, "batch", int),
        (_HEADER_LR_RE, "lr", float),
        (_HEADER_GAMMA_RE, "gamma", float),
        (_HEADER_SYNC_RE, "target_sync_interval", int),
    ]:
        m = pattern.search(text)
        if m:
            try:
                config[key] = cast(m.group(1))
            except (ValueError, IndexError):
                pass

    m = _HEADER_EPS_RE.search(text)
    if m:
        config["epsilon_start"] = float(m.group(1))
        config["epsilon_end"] = float(m.group(2))
        config["epsilon_steps"] = int(m.group(3))

    m = _HEADER_OPP_RE.search(text)
    if m:
        config["opponent"] = m.group(1).strip()

    m = _HEADER_REWARD_RE.search(text)
    if m:
        config["reward_mode"] = m.group(1).strip()

    # Phase 15E device diagnostics block (new runs only; absent in old logs)
    m = _DIAG_REQUESTED_DEVICE_RE.search(text)
    if m:
        config["requested_device"] = m.group(1).strip()
    m = _DIAG_RESOLVED_DEVICE_RE.search(text)
    if m:
        config["resolved_device"] = m.group(1).strip()
        # Prefer the explicit resolved_device over the header-level device= field
        config["device"] = config["resolved_device"]
    m = _DIAG_CUDA_NAME_RE.search(text)
    if m:
        config["cuda_device_name"] = m.group(1).strip()

    # Phase 16A architecture diagnostics (new runs only; absent in old logs)
    m = _HEADER_HIDDEN_LAYERS_RE.search(text)
    if m:
        try:
            config["hidden_layers"] = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        except ValueError:
            pass
    m = _HEADER_PARAM_COUNT_RE.search(text)
    if m:
        try:
            config["parameter_count"] = int(m.group(1))
        except ValueError:
            pass

    # Phase 17A algorithm (new runs only; absent in old logs)
    m = _HEADER_ALGORITHM_RE.search(text)
    if m:
        config["algorithm"] = m.group(1).strip()

    # Phase 18A CNN model_arch and observation_shape
    m = _HEADER_MODEL_ARCH_RE.search(text)
    if m:
        config["model_arch_explicit"] = m.group(1).strip()
    m = _HEADER_OBS_SHAPE_RE.search(text)
    if m:
        try:
            config["observation_shape"] = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        except ValueError:
            pass

    return config


def _parse_train_log_metrics(text: str) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for line in text.splitlines():
        m = _METRIC_RE.search(line)
        if m:
            points.append(MetricPoint(
                episode=int(m.group(1)),
                step=int(m.group(2)),
                values={
                    "opt_steps": _safe_float(m.group(3)),
                    "epsilon": _safe_float(m.group(4)),
                    "buffer_size": _safe_float(m.group(5)),
                    "avg_reward": _safe_float(m.group(6)),
                    "avg_episode_length": _safe_float(m.group(7)),
                    "loss": _safe_float(m.group(8)),
                    "avg_q_max": _safe_float(m.group(9)),
                    "pos_rewards": _safe_float(m.group(10)),
                    "neg_rewards": _safe_float(m.group(11)),
                },
            ))
    return points


def _extract_trailing_json(text: str) -> dict[str, Any] | None:
    """Extract and parse the trailing top-level JSON block from a train.log."""
    lines = text.splitlines()
    json_start = None
    for i, line in enumerate(lines):
        if line == "{":   # exactly '{' at column 0
            json_start = i
    if json_start is None:
        return None
    json_text = "\n".join(lines[json_start:])
    depth = 0
    end_pos = -1
    in_str = False
    escape = False
    for idx, ch in enumerate(json_text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = idx
                break
    if end_pos == -1:
        return None
    try:
        return json.loads(json_text[: end_pos + 1])
    except json.JSONDecodeError:
        return None


def _extract_periodic_evals(summary_json: dict[str, Any], run_id: str) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for ev in summary_json.get("periodic_evals", []):
        opp = ev.get("opponent_id", "") or ""
        wr = _derive_win_rate(ev)
        results.append(BenchmarkResult(
            run_id=run_id,
            checkpoint=ev.get("checkpoint_id"),
            checkpoint_episode=ev.get("episode"),
            checkpoint_step=ev.get("env_step"),
            opponent=opp,
            opponent_family=_infer_opponent_family(opp),
            opponent_depth=_infer_opponent_depth(opp),
            games=ev.get("num_games"),
            wins=ev.get("wins"),
            losses=ev.get("losses"),
            draws=ev.get("draws"),
            win_rate=wr,
            side=None,
            seed=None,
            avg_game_length=ev.get("avg_game_length"),
            illegal_action_count=ev.get("illegal_action_count"),
            timeout_count=ev.get("timeout_count"),
            crash_count=ev.get("crash_count"),
        ))
    return results


# ---------------------------------------------------------------------------
# Per-run discovery and loading
# ---------------------------------------------------------------------------

def _best_benchmark(benchmarks: list[BenchmarkResult]) -> tuple[float | None, str | None]:
    best_wr: float | None = None
    best_opp: str | None = None
    for b in benchmarks:
        wr = b.win_rate
        if wr is not None and (best_wr is None or wr > best_wr):
            best_wr = wr
            best_opp = b.opponent
    return best_wr, best_opp


def discover_run(run_dir: Path) -> RunSummary | None:
    if not run_dir.is_dir():
        return None

    run_id = run_dir.name
    log_path = run_dir / "train.log"
    ckpt_dir = run_dir / "checkpoints"
    config_yaml_path = run_dir / "config.yaml"

    config: dict[str, Any] = {}
    summary_json: dict[str, Any] = {}
    text = ""

    # Prefer config.yaml if present; fall back to train.log header parsing
    if config_yaml_path.exists():
        try:
            import yaml as _yaml
            _yaml_cfg = _yaml.safe_load(config_yaml_path.read_text()) or {}
            # Normalise to the same keys the header parser would produce
            config["total_episodes"] = _yaml_cfg.get("episodes")
            config["obs_version"] = _yaml_cfg.get("obs_version")
            config["device"] = _yaml_cfg.get("device")
            config["buffer_capacity"] = _yaml_cfg.get("buffer_capacity")
            config["warmup"] = _yaml_cfg.get("warmup_size")
            config["batch"] = _yaml_cfg.get("batch_size")
            config["lr"] = _yaml_cfg.get("lr")
            config["gamma"] = _yaml_cfg.get("gamma")
            config["epsilon_start"] = _yaml_cfg.get("epsilon_start")
            config["epsilon_end"] = _yaml_cfg.get("epsilon_end")
            config["epsilon_steps"] = _yaml_cfg.get("epsilon_decay_steps")
            config["target_sync_interval"] = _yaml_cfg.get("target_sync_interval")
            config["opponent"] = _yaml_cfg.get("opponent")
            config["reward_mode"] = _yaml_cfg.get("reward_mode")
            config["algorithm"] = _yaml_cfg.get("algorithm")
            config["model_arch_explicit"] = _yaml_cfg.get("model_arch")
            hl = _yaml_cfg.get("hidden_layers")
            if hl:
                config["hidden_layers"] = hl if isinstance(hl, list) else [int(x) for x in str(hl).split(",") if x.strip()]
            cc = _yaml_cfg.get("cnn_channels")
            if cc:
                config["cnn_channels"] = cc if isinstance(cc, list) else [int(x) for x in str(cc).split(",") if x.strip()]
            # New-style structured fields (R6/R7): pass through for UI consumption
            if "train_opponent" in _yaml_cfg:
                config["train_opponent"] = _yaml_cfg["train_opponent"]
            if "train_opponent_schedule" in _yaml_cfg:
                config["train_opponent_schedule"] = _yaml_cfg["train_opponent_schedule"]
            if "evaluation" in _yaml_cfg:
                config["evaluation"] = _yaml_cfg["evaluation"]
            config["_from_config_yaml"] = True
        except Exception:
            pass

    if log_path.exists():
        try:
            text = log_path.read_text(errors="replace")
            if not config.get("_from_config_yaml"):
                config = _parse_train_log_header(text)
            summary_json = _extract_trailing_json(text) or {}
        except OSError:
            pass

    checkpoints = _parse_checkpoints(ckpt_dir)
    latest_ckpt = checkpoints[-1].filename if checkpoints else None

    reward_mode = _infer_reward_mode(run_id, config)
    total_episodes = summary_json.get("total_episodes") or config.get("total_episodes")

    has_metrics = bool(
        (run_dir / "metrics.jsonl").exists()
        or log_path.exists()
        or any((run_dir / f).exists() for f in ("metrics.csv", "metrics.json", "training_log.jsonl"))
    )
    has_checkpoints = bool(checkpoints)

    has_inline_bench = bool(
        (run_dir / "eval_results.jsonl").exists()
        or summary_json.get("periodic_evals")
        or summary_json.get("evaluation")
    )
    has_file_bench = any(
        (run_dir / d).is_dir() for d in ("benchmarks", "evaluation", "arena", "results")
    )
    has_benchmarks = has_inline_bench or has_file_bench

    has_notes = (run_dir / "notes.md").exists() or (run_dir / "notes.txt").exists()
    notes_preview: str | None = None
    for nf in ("notes.md", "notes.txt"):
        np = run_dir / nf
        if np.exists():
            try:
                notes_preview = np.read_text(errors="replace")[:120].strip()
            except OSError:
                pass
            break

    available_metrics: list[str] = []
    if (run_dir / "metrics.jsonl").exists():
        available_metrics.append("metrics.jsonl")
    if log_path.exists():
        available_metrics.append("train.log")

    available_benchmarks: list[str] = []
    if (run_dir / "eval_results.jsonl").exists():
        available_benchmarks.append("eval_results.jsonl")
    periodic = summary_json.get("periodic_evals", [])
    final_eval = summary_json.get("evaluation")
    if periodic:
        available_benchmarks.append("periodic_evals")
    if final_eval:
        available_benchmarks.append("final_evaluation")

    final_avg_loss = summary_json.get("final_avg_loss")
    final_avg_reward = summary_json.get("avg_episode_reward")
    total_env_steps = summary_json.get("total_env_steps")

    final_epsilon: float | None = None
    if text:
        metric_pts = _parse_train_log_metrics(text)
        if metric_pts:
            final_epsilon = metric_pts[-1].values.get("epsilon")

    benchmarks = _extract_periodic_evals(summary_json, run_id)
    if final_eval:
        opp = final_eval.get("opponent_id", "") or ""
        wr = _derive_win_rate(final_eval)
        benchmarks.append(BenchmarkResult(
            run_id=run_id,
            checkpoint=final_eval.get("checkpoint_id"),
            checkpoint_episode=summary_json.get("total_episodes"),
            checkpoint_step=summary_json.get("total_env_steps"),
            opponent=opp,
            opponent_family=_infer_opponent_family(opp),
            opponent_depth=_infer_opponent_depth(opp),
            games=final_eval.get("num_games"),
            wins=final_eval.get("wins"),
            losses=final_eval.get("losses"),
            draws=final_eval.get("draws"),
            win_rate=wr,
            side=None, seed=None,
            avg_game_length=final_eval.get("avg_game_length"),
            illegal_action_count=final_eval.get("illegal_action_count"),
            timeout_count=None, crash_count=None,
        ))

    best_wr, best_opp = _best_benchmark(benchmarks)
    latest_wr = benchmarks[-1].win_rate if benchmarks else None

    return RunSummary(
        run_id=run_id,
        algorithm="DQN",
        reward_mode=reward_mode,
        model_arch=config.get("model_arch_explicit") or config.get("model_arch") or config.get("obs_version"),
        total_episodes=total_episodes,
        created_at=_path_ctime(run_dir),
        modified_at=_path_mtime(run_dir),
        latest_checkpoint=latest_ckpt,
        best_checkpoint=None,
        has_metrics=has_metrics,
        has_benchmarks=has_benchmarks,
        has_checkpoints=has_checkpoints,
        has_notes=has_notes,
        has_eval_replays=(run_dir / "eval_replays.jsonl").exists(),
        best_benchmark_win_rate=best_wr,
        best_benchmark_opponent=best_opp,
        latest_benchmark_win_rate=latest_wr,
        available_metrics=available_metrics,
        available_benchmarks=available_benchmarks,
        notes_preview=notes_preview,
        final_avg_loss=final_avg_loss,
        final_avg_reward=final_avg_reward,
        final_epsilon=final_epsilon,
        total_env_steps=total_env_steps,
    )


def load_run_detail(run_dir: Path) -> RunDetail | None:
    if not run_dir.is_dir():
        return None

    run_id = run_dir.name
    log_path = run_dir / "train.log"
    ckpt_dir = run_dir / "checkpoints"
    config_yaml_path = run_dir / "config.yaml"

    config: dict[str, Any] = {}
    summary_json: dict[str, Any] = {}

    # Prefer config.yaml over train.log header parsing
    if config_yaml_path.exists():
        try:
            import yaml as _yaml
            config = _yaml.safe_load(config_yaml_path.read_text()) or {}
            config["_source"] = "config.yaml"
        except Exception:
            pass

    if log_path.exists():
        try:
            text = log_path.read_text(errors="replace")
            if not config.get("_source"):
                config = _parse_train_log_header(text)
            summary_json = _extract_trailing_json(text) or {}
        except OSError:
            pass

    checkpoints = _parse_checkpoints(ckpt_dir)

    benchmarks = load_run_benchmarks(run_dir)
    bench_episodes: set[int | None] = {b.checkpoint_episode for b in benchmarks}
    best_wr_by_ep: dict[int, float] = {}
    for b in benchmarks:
        ep = b.checkpoint_episode
        wr = b.win_rate
        if ep is not None and wr is not None:
            if ep not in best_wr_by_ep or wr > best_wr_by_ep[ep]:
                best_wr_by_ep[ep] = wr
    for ck in checkpoints:
        ck.has_benchmark = ck.episode in bench_episodes
        if ck.episode in best_wr_by_ep:
            ck.best_win_rate = best_wr_by_ep[ck.episode]

    metric_files: list[str] = []
    if (run_dir / "metrics.jsonl").exists():
        metric_files.append("metrics.jsonl")
    if log_path.exists():
        metric_files.append("train.log")
    for candidate in ("metrics.csv", "metrics.json", "training_log.jsonl"):
        if (run_dir / candidate).exists():
            metric_files.append(candidate)

    benchmark_files: list[str] = []
    if (run_dir / "eval_results.jsonl").exists():
        benchmark_files.append("eval_results.jsonl")
    if summary_json.get("periodic_evals"):
        benchmark_files.append("train.log (periodic_evals)")
    if summary_json.get("evaluation"):
        benchmark_files.append("train.log (final_evaluation)")
    for subdir in ("benchmarks", "evaluation", "arena", "results"):
        bdir = run_dir / subdir
        if bdir.is_dir():
            for f in sorted(bdir.iterdir()):
                if f.suffix == ".json":
                    benchmark_files.append(f"{subdir}/{f.name}")

    reward_mode = _infer_reward_mode(run_id, config)
    total_episodes = summary_json.get("total_episodes") or config.get("total_episodes")

    summary_stats: dict[str, Any] = {}
    for k in (
        "total_env_steps", "total_episodes", "total_optimizer_steps", "total_target_syncs",
        "total_illegal_actions", "avg_episode_reward", "avg_episode_length",
        "final_avg_loss", "final_avg_q_max",
        # Phase 18A CNN fields
        "model_arch", "observation_version", "observation_shape", "parameter_count",
        "cnn_channels", "reward_mode", "distance_reward_weight", "distance_delta_clip",
        "opponent",
    ):
        if k in summary_json:
            summary_stats[k] = summary_json[k]

    notes: str | None = None
    has_notes = False
    for nf in ("notes.md", "notes.txt"):
        np = run_dir / nf
        if np.exists():
            has_notes = True
            try:
                notes = np.read_text(errors="replace")
            except OSError:
                pass
            break

    has_metrics = bool(metric_files)
    has_benchmarks = bool(benchmarks)

    return RunDetail(
        run_id=run_id,
        algorithm="DQN",
        reward_mode=reward_mode,
        model_arch=config.get("model_arch_explicit") or config.get("model_arch") or config.get("obs_version"),
        total_episodes=total_episodes,
        created_at=_path_ctime(run_dir),
        modified_at=_path_mtime(run_dir),
        config=config,
        checkpoints=checkpoints,
        metric_files=metric_files,
        benchmark_files=benchmark_files,
        notes=notes,
        summary_stats=summary_stats,
        has_metrics=has_metrics,
        has_benchmarks=has_benchmarks,
        has_notes=has_notes,
        has_eval_replays=(run_dir / "eval_replays.jsonl").exists(),
    )


def load_run_metrics(run_dir: Path) -> MetricSeries:
    run_id = run_dir.name
    warnings: list[str] = []
    points: list[MetricPoint] = []

    # --- Priority 1: metrics.jsonl (structured, written by new training runs) ---
    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        try:
            for raw_line in metrics_jsonl.read_text(errors="replace").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ep = row.get("episode")
                step = row.get("step")
                values = {
                    k: v for k, v in row.items()
                    if k not in ("episode", "step", "type", "timestamp", "agent_id", "opponent")
                    and isinstance(v, (int, float))
                }
                points.append(MetricPoint(
                    episode=int(ep) if ep is not None else None,
                    step=int(step) if step is not None else None,
                    values=values,
                ))
        except Exception as e:
            warnings.append(f"Could not parse metrics.jsonl: {e}")

    # --- Priority 2: legacy structured files and train.log fallback ---
    if not points:
        for candidate in ("metrics.csv",):
            cpath = run_dir / candidate
            if cpath.exists():
                try:
                    import csv
                    with cpath.open(newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            ep = _safe_float(row.get("episode", ""))
                            step = _safe_float(row.get("step", ""))
                            values = {k: _safe_float(v) for k, v in row.items() if k not in ("episode", "step")}
                            points.append(MetricPoint(
                                episode=int(ep) if ep is not None else None,
                                step=int(step) if step is not None else None,
                                values=values,
                            ))
                except Exception as e:
                    warnings.append(f"Could not parse {candidate}: {e}")

        for candidate in ("metrics.json", "training_log.json"):
            cpath = run_dir / candidate
            if cpath.exists():
                try:
                    data = json.loads(cpath.read_text())
                    if isinstance(data, list):
                        for row in data:
                            ep = row.get("episode")
                            step = row.get("step")
                            values = {k: v for k, v in row.items() if k not in ("episode", "step") and isinstance(v, (int, float))}
                            points.append(MetricPoint(
                                episode=int(ep) if ep is not None else None,
                                step=int(step) if step is not None else None,
                                values=values,
                            ))
                except Exception as e:
                    warnings.append(f"Could not parse {candidate}: {e}")

        cpath = run_dir / "training_log.jsonl"
        if cpath.exists():
            try:
                for raw_line in cpath.read_text().splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    ep = row.get("episode")
                    step = row.get("step")
                    values = {k: v for k, v in row.items() if k not in ("episode", "step") and isinstance(v, (int, float))}
                    points.append(MetricPoint(
                        episode=int(ep) if ep is not None else None,
                        step=int(step) if step is not None else None,
                        values=values,
                    ))
            except Exception as e:
                warnings.append(f"Could not parse training_log.jsonl: {e}")

        # --- Priority 3: fallback to train.log regex parsing (legacy runs) ---
        if not points:
            log_path = run_dir / "train.log"
            if log_path.exists():
                try:
                    text = log_path.read_text(errors="replace")
                    points = _parse_train_log_metrics(text)
                    if not points:
                        warnings.append("train.log parsed but no metric lines found (legacy run — consider backfilling metrics.jsonl)")
                except OSError as e:
                    warnings.append(f"Could not read train.log: {e}")
            else:
                warnings.append("No train.log found")

    if not points:
        warnings.append("No metric data points found in this run directory")

    columns: list[str] = []
    seen: set[str] = set()
    for p in points:
        for k in p.values:
            if k not in seen:
                columns.append(k)
                seen.add(k)

    return MetricSeries(run_id=run_id, points=points, columns=columns, warnings=warnings)


def load_run_benchmarks(run_dir: Path) -> list[BenchmarkResult]:
    run_id = run_dir.name
    results: list[BenchmarkResult] = []

    # --- Priority 1: eval_results.jsonl (structured, written by new training runs) ---
    eval_jsonl = run_dir / "eval_results.jsonl"
    if eval_jsonl.exists():
        try:
            for raw_line in eval_jsonl.read_text(errors="replace").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # New schema: opponent_name / opponent_type / opponent_depth fields
                if "opponent_name" in row:
                    opp = row["opponent_name"]
                    opp_family = _infer_opponent_family(row.get("opponent_type") or opp)
                    opp_depth = row.get("opponent_depth") or _infer_opponent_depth(opp)
                else:
                    # Legacy schema: flat "opponent" field
                    opp = row.get("opponent") or ""
                    opp_family = _infer_opponent_family(opp)
                    opp_depth = _infer_opponent_depth(opp)
                wr = _derive_win_rate(row)
                results.append(BenchmarkResult(
                    run_id=run_id,
                    checkpoint=None,
                    checkpoint_episode=row.get("episode"),
                    checkpoint_step=row.get("step"),
                    opponent=opp,
                    opponent_family=opp_family,
                    opponent_depth=opp_depth,
                    games=row.get("eval_games") or row.get("num_games"),
                    wins=row.get("wins"),
                    losses=row.get("losses"),
                    draws=row.get("draws"),
                    win_rate=wr,
                    side=None,
                    seed=None,
                    avg_game_length=row.get("avg_game_length"),
                    illegal_action_count=row.get("illegal_action_count"),
                    timeout_count=None,
                    crash_count=None,
                ))
        except Exception:
            pass

    # --- Priority 2: legacy train.log inline evals ---
    if not results:
        log_path = run_dir / "train.log"
        if log_path.exists():
            try:
                text = log_path.read_text(errors="replace")
                summary_json = _extract_trailing_json(text) or {}
                results.extend(_extract_periodic_evals(summary_json, run_id))

                final_eval = summary_json.get("evaluation")
                if final_eval:
                    opp = final_eval.get("opponent_id", "") or ""
                    wr = _derive_win_rate(final_eval)
                    results.append(BenchmarkResult(
                        run_id=run_id,
                        checkpoint=final_eval.get("checkpoint_id"),
                        checkpoint_episode=summary_json.get("total_episodes"),
                        checkpoint_step=summary_json.get("total_env_steps"),
                        opponent=opp,
                        opponent_family=_infer_opponent_family(opp),
                        opponent_depth=_infer_opponent_depth(opp),
                        games=final_eval.get("num_games"),
                        wins=final_eval.get("wins"),
                        losses=final_eval.get("losses"),
                        draws=final_eval.get("draws"),
                        win_rate=wr,
                        side=None, seed=None,
                        avg_game_length=final_eval.get("avg_game_length"),
                        illegal_action_count=final_eval.get("illegal_action_count"),
                        timeout_count=None, crash_count=None,
                    ))
            except OSError:
                pass

    for subdir in ("benchmarks", "evaluation", "arena", "results"):
        bdir = run_dir / subdir
        if not bdir.is_dir():
            continue
        for f in sorted(bdir.iterdir()):
            if f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                rows = data if isinstance(data, list) else [data]
                for row in rows:
                    known = {
                        "checkpoint", "opponent", "opponent_family", "opponent_config",
                        "games", "wins", "losses", "draws", "win_rate", "side", "seed",
                        "avg_game_length", "illegal_action_count", "timeout_count",
                        "crash_count", "checkpoint_episode", "checkpoint_step",
                    }
                    extra = {k: v for k, v in row.items() if k not in known}
                    opp = row.get("opponent", "") or ""
                    wr = _derive_win_rate(row)
                    results.append(BenchmarkResult(
                        run_id=run_id,
                        checkpoint=row.get("checkpoint"),
                        checkpoint_episode=row.get("checkpoint_episode"),
                        checkpoint_step=row.get("checkpoint_step"),
                        opponent=opp,
                        opponent_family=row.get("opponent_family") or _infer_opponent_family(opp),
                        opponent_depth=_infer_opponent_depth(opp),
                        games=row.get("games") or row.get("num_games"),
                        wins=row.get("wins"),
                        losses=row.get("losses"),
                        draws=row.get("draws"),
                        win_rate=wr,
                        side=row.get("side"),
                        seed=row.get("seed"),
                        avg_game_length=row.get("avg_game_length"),
                        illegal_action_count=row.get("illegal_action_count"),
                        timeout_count=row.get("timeout_count"),
                        crash_count=row.get("crash_count"),
                        extra=extra,
                    ))
            except Exception:
                pass

    return results


def diagnose_run(run_dir: Path) -> DiagnosisReport:
    run_id = run_dir.name
    items: list[DiagnosisItem] = []

    series = load_run_metrics(run_dir)
    benchmarks = load_run_benchmarks(run_dir)

    pts = series.points
    if not pts:
        items.append(DiagnosisItem("warning", "No metric data found — cannot assess training stability."))
    else:
        losses = [p.values.get("loss") for p in pts if p.values.get("loss") is not None]
        if losses:
            if losses[-1] < losses[0] * 0.9:
                items.append(DiagnosisItem("positive", f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f} over training."))
            elif losses[-1] > losses[0] * 1.1:
                items.append(DiagnosisItem("warning", f"Loss increased from {losses[0]:.4f} to {losses[-1]:.4f} — possible instability."))
            else:
                items.append(DiagnosisItem("info", f"Loss did not change significantly ({losses[0]:.4f} → {losses[-1]:.4f})."))

        rewards = [p.values.get("avg_reward") for p in pts if p.values.get("avg_reward") is not None]
        if rewards:
            if rewards[-1] > rewards[0] + 0.05:
                items.append(DiagnosisItem("positive", f"Reward improved from {rewards[0]:.3f} to {rewards[-1]:.3f}."))
            elif rewards[-1] < rewards[0] - 0.05:
                items.append(DiagnosisItem("warning", f"Reward declined from {rewards[0]:.3f} to {rewards[-1]:.3f}."))
            else:
                items.append(DiagnosisItem("info", f"Reward stayed flat ({rewards[0]:.3f} → {rewards[-1]:.3f})."))

        epsilons = [p.values.get("epsilon") for p in pts if p.values.get("epsilon") is not None]
        if epsilons:
            final_eps = epsilons[-1]
            if final_eps > 0.3:
                items.append(DiagnosisItem("warning", f"Final epsilon is high ({final_eps:.3f}). Agent may still be exploring heavily."))
            elif final_eps < 0.1:
                items.append(DiagnosisItem("positive", f"Epsilon reached low value ({final_eps:.3f}) — exploitation phase active."))

        total_ep = pts[-1].episode if pts else None
        if total_ep is not None and total_ep < 500:
            items.append(DiagnosisItem("warning", f"Run is very short ({total_ep} episodes). Conclusions may be unreliable."))

    if not benchmarks:
        items.append(DiagnosisItem("warning", "No benchmark results found. Cannot judge playing strength."))
    else:
        wrs = [b.win_rate for b in benchmarks if b.win_rate is not None]
        if wrs:
            best_wr = max(wrs)
            if best_wr >= 0.5:
                items.append(DiagnosisItem("positive", f"Best benchmark win rate: {best_wr*100:.1f}% — agent wins majority of games."))
            elif best_wr >= 0.1:
                items.append(DiagnosisItem("info", f"Best benchmark win rate: {best_wr*100:.1f}% — agent wins some games but not majority."))
            else:
                items.append(DiagnosisItem("warning", f"Best benchmark win rate: {best_wr*100:.1f}% — agent rarely wins."))

        latest_bm_wr = benchmarks[-1].win_rate
        best_wr_overall, _ = _best_benchmark(benchmarks)
        if latest_bm_wr is not None and best_wr_overall is not None and latest_bm_wr < best_wr_overall - 0.05:
            items.append(DiagnosisItem("info", f"Latest checkpoint win rate ({latest_bm_wr*100:.1f}%) is below best observed ({best_wr_overall*100:.1f}%))."))

        total_illegal = sum(b.illegal_action_count or 0 for b in benchmarks)
        if total_illegal > 0:
            items.append(DiagnosisItem("warning", f"Total illegal actions across benchmarks: {total_illegal}."))

        total_timeout = sum(b.timeout_count or 0 for b in benchmarks)
        total_crash = sum(b.crash_count or 0 for b in benchmarks)
        if total_timeout > 0:
            items.append(DiagnosisItem("warning", f"Timeouts detected: {total_timeout}."))
        if total_crash > 0:
            items.append(DiagnosisItem("error", f"Crashes detected: {total_crash}."))

        by_ep = sorted(
            [(b.checkpoint_episode, b.win_rate) for b in benchmarks
             if b.checkpoint_episode is not None and b.win_rate is not None],
            key=lambda x: x[0]
        )
        if len(by_ep) >= 3:
            first_wr = by_ep[0][1]
            last_wr = by_ep[-1][1]
            if last_wr > first_wr + 0.05:
                items.append(DiagnosisItem("positive", f"Win rate improved across checkpoints ({first_wr*100:.1f}% → {last_wr*100:.1f}%)."))
            elif last_wr < first_wr - 0.05:
                items.append(DiagnosisItem("warning", f"Win rate declined across checkpoints ({first_wr*100:.1f}% → {last_wr*100:.1f}%)."))

    return DiagnosisReport(run_id=run_id, items=items)


# ---------------------------------------------------------------------------
# Bulk discovery
# ---------------------------------------------------------------------------

def discover_all_runs(artifact_roots: list[str]) -> tuple[list[RunSummary], list[str]]:
    summaries: list[RunSummary] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for root_str in artifact_roots:
        root = Path(root_str)
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir():
                continue
            has_log = (candidate / "train.log").exists()
            has_ckpts = (candidate / "checkpoints").is_dir()
            if not (has_log or has_ckpts):
                continue
            uid = str(candidate.resolve())
            if uid in seen:
                continue
            seen.add(uid)
            try:
                summary = discover_run(candidate)
                if summary is not None:
                    summaries.append(summary)
            except Exception as e:
                warnings.append(f"Error scanning {candidate.name}: {e}")

    # Sort by modified_at descending (latest first)
    summaries.sort(key=lambda s: s.modified_at or "", reverse=True)

    if not summaries:
        warnings.append(
            "No training runs discovered. Check TRAINING_DASHBOARD_ARTIFACT_ROOTS "
            "or ensure artifact directories contain train.log or checkpoints/."
        )

    return summaries, warnings


def load_eval_replays(run_dir: Path) -> list[EvalReplayMeta]:
    """Load the eval replay index from *run_dir*/eval_replays.jsonl.

    Returns an empty list if the file doesn't exist or contains no valid entries.
    Entries are returned in file order (chronological).
    """
    index_path = run_dir / "eval_replays.jsonl"
    if not index_path.exists():
        return []

    replays: list[EvalReplayMeta] = []
    try:
        with open(index_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                replays.append(EvalReplayMeta(
                    episode=d.get("episode"),
                    checkpoint=d.get("checkpoint"),
                    opponent_name=d.get("opponent_name"),
                    opponent_type=d.get("opponent_type"),
                    opponent_depth=d.get("opponent_depth"),
                    game_index=d.get("game_index"),
                    dqn_player_id=d.get("dqn_player_id"),
                    winner=d.get("winner"),
                    result_from_dqn_perspective=d.get("result_from_dqn_perspective"),
                    game_length=d.get("game_length"),
                    illegal_acts=d.get("illegal_acts"),
                    replay_path=d.get("replay_path", ""),
                    created_at=d.get("created_at"),
                ))
    except OSError:
        pass

    return replays
