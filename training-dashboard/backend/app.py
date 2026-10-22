"""Training Dashboard FastAPI application."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_artifact_roots
from . import mlflow
from .providers import LegacyAdapter, UnifiedRunService


def _asdict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(i) for i in obj]
    return obj


def _find_run_dir(run_id: str) -> Path | None:
    """Search artifact roots for a run directory matching run_id."""
    return LegacyAdapter().find(run_id)


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
    app.state.runs = UnifiedRunService()

    def delegate(run_id: str, method: str, *args) -> dict:
        try:
            provider = app.state.runs.provider_for(run_id)
            return getattr(provider, method)(run_id, *args)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run or replay not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, ImportError, OSError) as exc:
            raise HTTPException(status_code=503, detail=f"Evidence unavailable: {exc}") from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        roots = get_artifact_roots()
        return {
            "status": "ok",
            "service": "training-dashboard",
            "artifact_roots": roots,
            "read_only": True,
        }

    def data_json(relative: str) -> dict[str, Any]:
        root = Path(os.environ.get("QUORIDOR_DATA_ROOT", "/data/quoridor")).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=400, detail="Invalid data path")
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No benchmark data available yet")
        except (OSError, ValueError):
            raise HTTPException(status_code=503, detail="Benchmark data unavailable")

    @app.get("/api/arena/latest")
    def latest_matrix() -> dict[str, Any]:
        run_id = data_json("arena/latest.json")["run_id"]
        if not isinstance(run_id, str) or not run_id.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid run ID")
        return data_json(f"arena/{run_id}/report.json")

    @app.get("/api/arena/runs/{run_id}/manifest")
    def matrix_manifest(run_id: str) -> dict[str, Any]:
        if not run_id.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid run ID")
        return data_json(f"arena/{run_id}/manifest.json")

    @app.get("/api/models")
    def model_catalog() -> dict[str, Any]:
        catalog = data_json("catalog.json")
        try:
            registry = data_json("registry.json")
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            registry = {"models": []}
        return {"models": catalog["models"], "registry": registry["models"]}

    @app.get("/api/mlflow/experiments")
    def mlflow_experiments():
        try:
            return {"experiments": mlflow.experiments()}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/mlflow/experiments/{experiment_id}/runs")
    def mlflow_runs(experiment_id: str):
        try:
            return {"experiment_id": experiment_id, "runs": mlflow.runs(experiment_id)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/mlflow/runs/{run_id}/metrics/{metric_key}")
    def mlflow_metric_history(run_id: str, metric_key: str):
        try:
            return {"run_id": run_id, "metric_key": metric_key, "metrics": mlflow.metric_history(run_id, metric_key)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/mlflow/runs/{run_id}/arena-evaluation")
    def mlflow_arena_evaluation(run_id: str) -> dict[str, Any]:
        """Map a registered checkpoint to immutable CNN benchmark evidence."""
        try:
            run = mlflow.run(run_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        source_path = run["params"].get("source_path")
        report_path = Path(os.environ.get(
            "CNN_ARCHITECTURE_ARENA_REPORT",
            "/data/quoridor/experiments/cnn-architecture-20261021-213018/arena-v2/report.json",
        ))
        try:
            report = app.state.runs.mlflow.local.json(report_path)
        except FileNotFoundError:
            return {"available": False, "reason": "No immutable Arena evidence is mounted for this run."}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Arena evidence is unavailable") from exc

        agent = next(
            (candidate for candidate in report.get("agents", [])
             if candidate.get("algo", {}).get("params", {}).get("checkpoint") == source_path
             and run.get("tags", {}).get("checkpoint_sha256")
             and candidate.get("algo", {}).get("params", {}).get("sha256") == run["tags"]["checkpoint_sha256"]),
            None,
        )
        if agent is None:
            return {"available": False, "reason": "This run has no matching frozen Arena checkpoint evidence."}
        agent_id = agent["id"]
        rows = [
            {"opponent": opponent, **result}
            for opponent, result in report.get("matrix", {}).get(agent_id, {}).items()
            if result and result.get("games", 0) > 0
        ]
        ranking = next((item for item in report.get("ranking", []) if item.get("id") == agent_id), None)
        return {
            "available": True,
            "run_id": run_id,
            "checkpoint_sha256": agent["algo"]["params"]["sha256"],
            "agent_id": agent_id,
            "evaluation_id": report.get("run_id"),
            "summary": ranking,
            "opponents": rows,
            "game_records_available": False,
            "game_records_reason": "Arena blocks are immutable raw evidence but are not Dashboard replay records.",
        }

    @app.get("/api/runs")
    def list_runs(source: str | None = None, algorithm: str | None = None,
                  experiment: str | None = None, status: str | None = None) -> dict[str, Any]:
        return app.state.runs.listing(source, algorithm, experiment, status)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return delegate(run_id, "detail")

    @app.get("/api/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str) -> dict[str, Any]:
        return delegate(run_id, "metrics")

    @app.get("/api/runs/{run_id}/benchmarks")
    def get_run_benchmarks(run_id: str) -> dict[str, Any]:
        return delegate(run_id, "benchmarks")

    @app.get("/api/runs/{run_id}/diagnosis")
    def get_run_diagnosis(run_id: str) -> dict[str, Any]:
        return delegate(run_id, "diagnosis")

    @app.get("/api/summary")
    def get_summary() -> dict[str, Any]:
        listing = app.state.runs.listing()
        summaries = [SimpleNamespace(**row) for row in listing["runs"]]
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
            "warnings": listing["warnings"],
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
        return delegate(run_id, "replays")

    @app.get("/api/runs/{run_id}/eval-replays/{replay_path:path}")
    def get_eval_replay_file(run_id: str, replay_path: str):
        """Return the raw replay JSON for a specific replay file."""
        return delegate(run_id, "replay", replay_path)

    return app


app = create_app()
