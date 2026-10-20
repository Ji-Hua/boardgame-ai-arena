"""Record a completed benchmark and its full evidence in MLflow; run in container."""
import argparse
import json
from pathlib import Path
from mlflow import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("--registry", type=Path, default=Path("/quoridor-data/registry.json"))
args = parser.parse_args()
if not args.run_id.replace("-", "").replace("_", "").isalnum():
    parser.error("Invalid run ID")
data = Path("/quoridor-data")
folder = data / "arena" / args.run_id
report = json.loads((folder / "report.json").read_text())
registry = json.loads(args.registry.read_text())
versions = {v["id"]: v for v in registry["models"]}
for agent in report["agents"]:
    if agent["algo"]["type"] == "dqn":
        if versions.get(agent["id"], {}).get("sha256") != agent["algo"]["params"]["sha256"]:
            raise ValueError(f"Registry does not match evaluated checkpoint: {agent['id']}")
if report["status"] != "complete":
    raise ValueError("Only completed benchmarks can be registered")
client = MlflowClient(tracking_uri="http://127.0.0.1:8742")
experiment_name = "Quoridor / paired benchmarks"
experiment = client.get_experiment_by_name(experiment_name)
exp_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
existing = client.search_runs([exp_id], filter_string=f"tags.benchmark_manifest = '{report['manifest_sha256']}'")
if any(run.info.status == "FINISHED" for run in existing):
    print("Already registered", report["run_id"])
    raise SystemExit(0)
parent = client.create_run(exp_id, tags={
    "mlflow.runName": report["run_id"], "benchmark_manifest": report["manifest_sha256"],
    "source_sha256": report["source_sha256"], "protocol": report["protocol"],
})
rid = parent.info.run_id
try:
    client.log_param(rid, "pairs_per_match", report["pairs_per_match"])
    client.log_param(rid, "max_plies", report["max_plies"])
    client.log_param(rid, "confidence", report["confidence"])
    client.log_param(rid, "rating", report["rating"])
    for key in ("ranked_games", "control_games", "unique_openings"):
        client.log_metric(rid, key, report[key])
    client.log_metric(rid, "forfeits", len(report["forfeits"]))
    for filename in ("report.json", "manifest.json", "source.tar.gz"):
        client.log_artifact(rid, str(folder / filename), artifact_path="evidence")
    client.log_artifacts(rid, str(folder / "blocks"), artifact_path="evidence/blocks")
    for row in report["ranking"]:
        tags = {"mlflow.runName": row["id"], "mlflow.parentRunId": rid, "agent_id": row["id"],
                "protocol": report["protocol"], "benchmark_run_id": report["run_id"]}
        version = versions.get(row["id"])
        if version:
            tags.update(checkpoint_sha256=version["sha256"], registry_name=version["name"],
                        registry_version=version["version"])
        child = client.create_run(exp_id, tags=tags).info.run_id
        for key in ("score", "elo", "wins", "draws", "losses", "games", "forfeits",
                    "p1_score", "p2_score", "mean_decision_ms"):
            client.log_metric(child, key, row[key])
        for opponent, cell in report["matrix"][row["id"]].items():
            if cell is not None:
                for key in ("score", "wins", "draws", "losses"):
                    client.log_metric(child, f"vs_{opponent}_{key}", cell[key])
        client.set_terminated(child)
        if version:
            for key, value in {
                "validation_status": "failed" if any(f["offender"] == row["id"] for f in report["forfeits"]) else "evaluated",
                "latest_benchmark": report["run_id"],
                "benchmark_manifest": report["manifest_sha256"],
            }.items():
                client.set_model_version_tag(version["name"], version["version"], key, value)
    client.set_terminated(rid)
except Exception:
    client.set_terminated(rid, status="FAILED")
    raise
print(json.dumps({"benchmark": report["run_id"], "mlflow_run_id": rid, "experiment_id": exp_id}))
