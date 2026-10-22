"""Read-only MLflow Tracking API adapter for the Training Dashboard."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.68.88:8742").rstrip("/")


def _request(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{MLFLOW_URI}/api/2.0/mlflow/{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except (HTTPError, URLError, ValueError) as exc:
        raise RuntimeError(f"MLflow Tracking API unavailable: {exc}") from exc


def experiments() -> list[dict]:
    data = _request("experiments/search", {"max_results": 100})
    return [{"id": row["experiment_id"], "name": row["name"], "lifecycle_stage": row["lifecycle_stage"]}
            for row in data.get("experiments", [])]


def runs(experiment_id: str, max_results: int = 100) -> list[dict]:
    data = _request("runs/search", {"experiment_ids": [experiment_id], "max_results": max_results})
    result = []
    for row in data.get("runs", []):
        info, values = row["info"], row.get("data", {})
        params = {item["key"]: item["value"] for item in values.get("params", [])}
        tags = {item["key"]: item["value"] for item in values.get("tags", [])}
        metrics = {item["key"]: item["value"] for item in values.get("metrics", [])}
        result.append({"run_id": info["run_id"], "run_name": info.get("run_name"), "status": info["status"],
                       "start_time": info.get("start_time"), "end_time": info.get("end_time"),
                       "artifact_uri": info.get("artifact_uri"), "params": params, "tags": tags, "metrics": metrics})
    return result


def run(run_id: str) -> dict:
    data = _request(f"runs/get?{urlencode({'run_id': run_id})}")
    row = data["run"]
    info, values = row["info"], row.get("data", {})
    return {
        "run_id": info["run_id"],
        "params": {item["key"]: item["value"] for item in values.get("params", [])},
        "tags": {item["key"]: item["value"] for item in values.get("tags", [])},
    }


def metric_history(run_id: str, metric_key: str) -> list[dict]:
    query = urlencode({"run_id": run_id, "metric_key": metric_key})
    return _request(f"metrics/get-history?{query}").get("metrics", [])
