"""Training Dashboard FastAPI application."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_artifact_roots
from .discovery import (
    diagnose_run,
    discover_all_runs,
    discover_run,
    load_eval_replays,
    load_run_benchmarks,
    load_run_detail,
    load_run_metrics,
)


def _asdict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(i) for i in obj]
    return obj


def _find_run_dir(run_id: str) -> Path | None:
    """Search artifact roots for a run directory matching run_id."""
    for root_str in get_artifact_roots():
        root = Path(root_str)
        candidate = root / run_id
        if candidate.is_dir():
            return candidate
    return None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quoridor Training Dashboard",
        version="0.1.0",
        description=(
            "Read-only dashboard for Quoridor RL training artifacts. "
            "Does not start training, run evaluation, or mutate any files."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        roots = get_artifact_roots()
        return {
            "status": "ok",
            "service": "training-dashboard",
            "artifact_roots": roots,
            "read_only": True,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        roots = get_artifact_roots()
        summaries, warnings = discover_all_runs(roots)
        return {
            "runs": [_asdict(s) for s in summaries],
            "count": len(summaries),
            "artifact_roots_scanned": roots,
            "warnings": warnings,
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run_dir = _find_run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        detail = load_run_detail(run_dir)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' could not be loaded")
        return _asdict(detail)

    @app.get("/api/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str) -> dict[str, Any]:
        run_dir = _find_run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        series = load_run_metrics(run_dir)
        return _asdict(series)

    @app.get("/api/runs/{run_id}/benchmarks")
    def get_run_benchmarks(run_id: str) -> dict[str, Any]:
        run_dir = _find_run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        results = load_run_benchmarks(run_dir)
        return {
            "run_id": run_id,
            "benchmarks": [_asdict(r) for r in results],
            "count": len(results),
        }

    @app.get("/api/runs/{run_id}/diagnosis")
    def get_run_diagnosis(run_id: str) -> dict[str, Any]:
        run_dir = _find_run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        report = diagnose_run(run_dir)
        return _asdict(report)

    @app.get("/api/summary")
    def get_summary() -> dict[str, Any]:
        roots = get_artifact_roots()
        summaries, _ = discover_all_runs(roots)
        algorithms: set[str] = set()
        reward_modes: set[str] = set()
        model_arches: set[str] = set()
        latest_run_id: str | None = None
        latest_modified: str | None = None
        with_metrics = 0
        with_benchmarks = 0
        with_checkpoints = 0
        for s in summaries:
            algorithms.add(s.algorithm)
            if s.reward_mode:
                reward_modes.add(s.reward_mode)
            if s.model_arch:
                model_arches.add(s.model_arch)
            if s.has_metrics:
                with_metrics += 1
            if s.has_benchmarks:
                with_benchmarks += 1
            if s.has_checkpoints:
                with_checkpoints += 1
            if s.modified_at is not None:
                if latest_modified is None or s.modified_at > latest_modified:
                    latest_modified = s.modified_at
                    latest_run_id = s.run_id
        return {
            "total_runs": len(summaries),
            "runs_with_metrics": with_metrics,
            "runs_with_benchmarks": with_benchmarks,
            "runs_with_checkpoints": with_checkpoints,
            "latest_run_id": latest_run_id,
            "latest_modified": latest_modified,
            "available_algorithms": sorted(algorithms),
            "available_reward_modes": sorted(reward_modes),
            "available_model_arches": sorted(model_arches),
        }

    @app.get("/api/runs/{run_id}/eval-replays")
    def get_eval_replays(run_id: str):
        """List all saved evaluation replay index entries for a run."""
        run_dir = _find_run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        replays = load_eval_replays(run_dir)
        return {"run_id": run_id, "count": len(replays), "replays": _asdict(replays)}

    @app.get("/api/runs/{run_id}/eval-replays/{replay_path:path}")
    def get_eval_replay_file(run_id: str, replay_path: str):
        """Return the raw replay JSON for a specific replay file."""
        run_dir = _find_run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        # Prevent path traversal
        try:
            full_path = (run_dir / replay_path).resolve()
            run_dir_resolved = run_dir.resolve()
            full_path.relative_to(run_dir_resolved)
        except (ValueError, OSError):
            raise HTTPException(status_code=400, detail="Invalid replay path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Replay file not found: {replay_path}")
        try:
            import json as _json
            with open(full_path) as fh:
                return _json.load(fh)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read replay: {exc}")

    return app


app = create_app()
