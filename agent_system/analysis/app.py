"""Standalone HTTP entry point for Quoridor Decision Lab."""
from contextlib import contextmanager
import math
from pathlib import Path
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from . import explain, neural_catalog
from .state import ENGINE, Move, Position, build_state, presets, snapshot

WEB = Path(__file__).resolve().parents[2] / "decision-lab" / "web"
app = FastAPI(title="Quoridor Decision Lab", version="1.1.0")
slots = threading.BoundedSemaphore(2)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Position
    algorithm: Literal["greedy", "minimax", "dqn"] = "greedy"
    depth: StrictInt = Field(default=2, ge=1, le=3)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_model(self):
        if self.algorithm == "dqn" and not self.model_id:
            raise ValueError("DQN 分析需要选择 model_id。")
        return self


class BranchRequest(AnalyzeRequest):
    action: Move


class TensorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Position
    model_id: str = Field(min_length=1, max_length=128)
    kind: Literal["activation", "weight"]
    layer: str = Field(min_length=1, max_length=128)
    channel: StrictInt = Field(default=0, ge=0, le=100000)
    input_channel: StrictInt = Field(default=0, ge=0, le=100000)
    offset: StrictInt = Field(default=0, ge=0, le=20_000_000)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Position
    model_id: str = Field(min_length=1, max_length=128)
    simulations: StrictInt = Field(default=50, ge=1, le=200)
    c_puct: float = Field(default=1.5, gt=0, le=10, allow_inf_nan=False)


class MultiCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Position
    model_ids: list[str] = Field(min_length=1, max_length=8)
    actual_action: Move | None = None


def neural_module():
    try:
        from . import neural
        return neural
    except ImportError as exc:
        raise HTTPException(503, "神经网络分析需要当前环境安装项目的 PyTorch 推理依赖。") from exc


@contextmanager
def guarded():
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "已有两个分析正在运行，请稍后重试。")
    try:
        yield
    except explain.AnalysisLimit as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        slots.release()


@app.get("/health")
def health():
    return {"status": "ok", "component": "decision-lab"}


@app.get("/api/meta")
def meta():
    return {"presets": presets(), "provenance": explain.PROVENANCE,
            "limits": {"max_depth": 3, "max_search_nodes": 75000, "seconds": 6, "trace_nodes": 1200}}


@app.get("/api/models")
def models():
    with guarded():
        return neural_catalog.catalog()


@app.post("/api/position")
def validate_position(position: Position):
    with guarded():
        return snapshot(build_state(position))


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    with guarded():
        if request.algorithm == "dqn":
            return neural_module().analyze(request.position, request.model_id)
        return explain.analyze(request.position, request.algorithm, request.depth)


@app.post("/api/multi-compare")
def multi_compare(request: MultiCompareRequest):
    """Compare all selected CNNs with canonical Greedy and exact D2 branches."""
    with guarded():
        state = build_state(request.position)
        legal = snapshot(state)["legal_actions"]
        greedy = explain.analyze(request.position, "greedy", 1)
        d1 = explain.analyze(request.position, "minimax", 1)
        d2_rows = []
        for action in legal:
            branch = explain.branch(request.position, Move.model_validate(action), "minimax", 2,
                                    node_limit=75_000, seconds=6, trace_limit=0)
            d2_rows.append({"id": branch["action_id"], "action": action, "value": branch["value"],
                            "after": branch["after"], "immediate_win": branch["after"]["winner"] == request.position.turn})
        def score_value(value):
            return math.inf if value == "+inf" else -math.inf if value == "-inf" else value

        best = max(score_value(row["value"]) for row in d2_rows)
        optimal = [row["id"] for row in d2_rows if score_value(row["value"]) == best]
        neural = neural_module()
        models = []
        for model_id in request.model_ids:
            result = neural.analyze(request.position, model_id)
            ranked = sorted(result["candidates"], key=lambda row: (-row["value"], row["action_index"]))
            selected = result["selected_id"]
            models.append({
                "model_id": model_id,
                "selected_id": selected,
                "selected_d2_optimal": selected in optimal,
                "top_actions": ranked[:5],
                "output_semantics": result["neural"]["model"]["output_semantics"],
            })
        selected_ids = {row["model_id"].split("_seed-")[0]: row["selected_id"] for row in models}
        return {
            "position": request.position.model_dump(mode="json"),
            "snapshot": snapshot(state),
            "legal_actions": legal,
            "actual_action": request.actual_action.model_dump(mode="json") if request.actual_action else None,
            "greedy": {"selected_id": greedy["selected_id"]},
            "minimax_d1": {"selected_id": d1["selected_id"]},
            "minimax_d2": {"best_value": best, "equivalent_optimal_ids": optimal, "actions": d2_rows},
            "models": models,
            "classifications": {
                "common_d2_suboptimal": bool(models) and all(item["selected_id"] not in optimal for item in models),
                "resnet_advantage": selected_ids.get("C-resnet") in optimal
                    and all(selected_ids.get(name) not in optimal for name in ("A-baseline", "B-deep", "D-wide")
                            if name in selected_ids),
                "resnet_regression": selected_ids.get("C-resnet") not in optimal
                    and any(selected_ids.get(name) in optimal for name in ("A-baseline", "B-deep") if name in selected_ids),
                "immediate_win_available": any(row["immediate_win"] for row in d2_rows),
            },
        }


@app.post("/api/branch")
def branch(request: BranchRequest):
    with guarded():
        if request.algorithm == "dqn":
            return neural_module().branch(request.position, request.action, request.model_id)
        return explain.branch(request.position, request.action, request.algorithm, request.depth)


@app.post("/api/neural/tensor")
def tensor(request: TensorRequest):
    with guarded():
        return neural_module().tensor(**request.model_dump(exclude={"position"}), position=request.position)


@app.post("/api/neural/search")
def search(request: SearchRequest):
    with guarded():
        return neural_module().search(**request.model_dump(exclude={"position"}), position=request.position)


@app.post("/api/apply")
def apply(request: BranchRequest):
    with guarded():
        return snapshot(ENGINE.apply_action(build_state(request.position), request.action.engine_action()))


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/assets", StaticFiles(directory=WEB), name="assets")
