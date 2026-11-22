"""Run in the MLflow container: record CNN experiments, diagnostics and models."""
import argparse
import hashlib
import json
from pathlib import Path
from mlflow import MlflowClient


def mounted(path):
    return Path("/quoridor-data") / Path(path).relative_to("/home/jihua/quoridor-data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", default="Quoridor / CNN repair")
    parser.add_argument("--suite", type=Path, default=None)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    registry = json.loads(Path("/quoridor-data/registry.json").read_text())
    registered = {m["id"]: m for m in registry["models"]}
    client = MlflowClient(tracking_uri="http://127.0.0.1:8742")
    name = args.experiment_name
    experiment = client.get_experiment_by_name(name)
    exp_id = experiment.experiment_id if experiment else client.create_experiment(name)
    versions = {(v.name, v.tags.get("checkpoint_sha256")): v for v in client.search_model_versions()
                if v.name in ("quoridor-cnn-dqn", "quoridor-cnn-policy")}
    registry_names = {m.name for m in client.search_registered_models()}
    parents, suites = {}, set()
    for model in catalog["models"]:
        if "training_experiment" not in model:
            continue
        folder = mounted(model["training_experiment"])
        if args.suite is not None and folder.parent != args.suite:
            continue
        suites.add(folder.parent)
        old = registered.get(model["id"])
        if old and old["sha256"] == model["sha256"]:
            continue
        semantics = model["metadata"].get("eval_summary", {}).get("output_semantics")
        registry_name = "quoridor-cnn-policy" if semantics == "policy_logits" or model["metadata"]["model_config"].get("algorithm") in {"policy_distillation", "ppo", "ppo_imitation", "selfplay_ppo"} else "quoridor-cnn-dqn"
        if registry_name not in registry_names:
            client.create_registered_model(registry_name)
            registry_names.add(registry_name)
        version = versions.get((registry_name, model["sha256"]))
        if version:
            registered[model["id"]] = {"id": model["id"], "name": registry_name,
                "version": version.version, "run_id": version.run_id, "sha256": model["sha256"]}
            continue
        config = json.loads((folder / "experiment.json").read_text())
        source_hash = hashlib.sha256((folder / "source-manifest.json").read_bytes()).hexdigest()
        if str(folder) not in parents:
            parent = client.create_run(exp_id, tags={"mlflow.runName": folder.name,
                "source_manifest_sha256": source_hash, "hypothesis": config["hypothesis"],
                "training_status": model.get("training_status", "complete")}).info.run_id
            parents[str(folder)] = parent
            for key, value in config["training"].items():
                client.log_param(parent, key, json.dumps(value))
            for key, value in config.get("demonstrations", {}).items():
                client.log_param(parent, "demo_" + key, value)
            for key, value in config.get("teacher", {}).items():
                client.log_param(parent, "teacher_" + key, value)
            for key, value in config.get("distillation", {}).items():
                client.log_param(parent, "distillation_" + key, json.dumps(value))
            for key, value in config.get("research", {}).items():
                client.log_param(parent, "research_" + key, json.dumps(value))
            for key, value in config.get("alphazero", {}).items():
                client.log_param(parent, "alphazero_" + key, json.dumps(value))
            for filename in ("experiment.json", "source-manifest.json", "source.tar.gz", "book.json",
                             "metrics.jsonl", "demo-metrics.jsonl", "teacher-metrics.jsonl", "distillation-metrics.jsonl", "dataset.json",
                             "training-state-keys.json", "training-games.jsonl.gz", "input-sha256.json",
                             "initialization.json", "result.json", "research-metrics.jsonl", "opponent-pool.json",
                             "az-metrics.jsonl", "calibration.json", "search-diagnoses.json"):
                if (folder / filename).is_file():
                    client.log_artifact(parent, str(folder / filename), artifact_path="training")
            az_metrics = folder / "az-metrics.jsonl"
            if az_metrics.is_file():
                if (folder / "calibration.json").is_file():
                    calibration = json.loads((folder / "calibration.json").read_text())
                    for key, value in calibration["heldout"].items():
                        if isinstance(value, (int, float)):
                            client.log_metric(parent, "calibration_" + key, value)
                for line in az_metrics.read_text().splitlines():
                    row = json.loads(line)
                    for key, value in row.items():
                        if isinstance(value, (int, float)):
                            client.log_metric(parent, key, value, step=row["episode"])
                    for group in ("totals", "gpu_cuda_event_seconds"):
                        for key, value in row.get(group, {}).items():
                            client.log_metric(parent, group + "_" + key, value, step=row["episode"])
                client.log_artifacts(parent, str(folder / "training-data"), artifact_path="training_data")
                if (folder / "training-state.pt").is_file():
                    client.log_artifact(parent, str(folder / "training-state.pt"), artifact_path="training_state")
                if (folder / "search-diagnostics").is_dir():
                    client.log_artifacts(parent, str(folder / "search-diagnostics"), artifact_path="search_diagnostics")
            research_metrics = folder / "research-metrics.jsonl"
            if research_metrics.is_file():
                for line in research_metrics.read_text().splitlines():
                    row = json.loads(line)
                    for key, value in row.items():
                        if isinstance(value, (int, float)):
                            client.log_metric(parent, key, value, step=row["episode"])
                    for group in ("totals", "gpu_cuda_event_seconds", "rollout"):
                        for key, value in row.get(group, {}).items():
                            client.log_metric(parent, group + "_" + key, value, step=row["episode"])
                # Retain critic/optimizer snapshots separately from serving weights.
                client.log_artifacts(parent, str(folder / "training-state"), artifact_path="training_state")
                aggregate = folder / "dagger-aggregate.pt"
                if aggregate.is_file():
                    client.log_artifact(parent, str(aggregate), artifact_path="training_data")
            distillation_metrics = folder / "distillation-metrics.jsonl"
            if distillation_metrics.is_file():
                for line in distillation_metrics.read_text().splitlines():
                    row = json.loads(line)
                    for key, value in row.items():
                        if key != "update":
                            client.log_metric(parent, key, value, step=row["update"])
                shared = mounted(config["distillation"]["dataset"]).parent
                for filename in ("dataset.json", "training-state-keys.json", "games.jsonl.gz"):
                    client.log_artifact(parent, str(shared / filename), artifact_path="shared_data")
            dataset = folder / "demonstrations.pt"
            if dataset.is_file():
                with dataset.open("rb") as stream:
                    client.set_tag(parent, "demonstrations_sha256", hashlib.file_digest(stream, "sha256").hexdigest())
            replay = folder / "teacher-replay.pt"
            if replay.is_file():
                with replay.open("rb") as stream:
                    client.set_tag(parent, "teacher_replay_sha256", hashlib.file_digest(stream, "sha256").hexdigest())
                client.log_param(parent, "teacher_replay_path", str(replay))
            teacher_metrics = folder / "teacher-metrics.jsonl"
            if teacher_metrics.is_file():
                for line in teacher_metrics.read_text().splitlines():
                    row = json.loads(line)
                    for key in ("updates", "replay_size", "teacher_probability", "epsilon", "loss",
                                "td_loss", "mc_loss", "expert_margin", "chosen_q_abs_max"):
                        if key in row:
                            client.log_metric(parent, key, row[key], step=row["episode"])
                    for key, value in row["totals"].items():
                        client.log_metric(parent, "training_" + key, value, step=row["episode"])
        rid = client.create_run(exp_id, tags={"mlflow.runName": model["id"],
            "mlflow.parentRunId": parents[str(folder)], "checkpoint_sha256": model["sha256"],
            "source_manifest_sha256": source_hash, "validation_status": "evaluated"}).info.run_id
        try:
            for key, value in model["metadata"].items():
                client.log_param(rid, key, json.dumps(value))
            diag_dir = mounted(model["diagnostics_dir"])
            diag = json.loads((diag_dir / "diagnosis.json").read_text())
            for key in ("score", "games", "max_abs_legal_q", "tactical_wins", "hidden_units_ever_active"):
                client.log_metric(rid, key, diag[key])
            if "max_abs_legal_output" in diag:
                client.log_metric(rid, "max_abs_legal_output", diag["max_abs_legal_output"])
            for key in ("wins", "draws", "losses", "all_hidden_zero", "missed_immediate_win", "immediate_win_available"):
                client.log_metric(rid, key, diag["counts"].get(key, 0))
            client.log_artifacts(rid, str(diag_dir), artifact_path="diagnostics")
            client.log_artifact(rid, str(mounted(model["checkpoint"])), artifact_path="checkpoint")
            version = client.create_model_version(registry_name, source=f"runs:/{rid}/checkpoint", run_id=rid,
                tags={"checkpoint_sha256": model["sha256"], "agent_id": model["id"], "validation_status": "evaluated",
                      "source_manifest_sha256": source_hash, "format": "quoridor_dqn_checkpoint",
                      "algorithm": model["metadata"]["model_config"].get("algorithm", "dqn"),
                      "output_semantics": model["metadata"].get("eval_summary", {}).get("output_semantics", "action_values")},
                description="CNN repair experiment; see Greedy matches and decision diagnostics before use.")
            if "teacher" in config:
                client.set_model_version_tag(registry_name, version.version, "teacher_depth", config["teacher"]["depth"])
                client.set_model_version_tag(registry_name, version.version, "training_episodes", model["metadata"]["episode_count"])
            if "distillation" in config:
                client.set_model_version_tag(registry_name, version.version, "teacher_depth", config["distillation"]["depth"])
                client.set_model_version_tag(registry_name, version.version, "sampled_games", model["metadata"]["episode_count"])
            if "research" in config:
                for key, value in {"training_method": config["research"]["method"],
                    "initialization": config["research"]["initialization"],
                    "parent_checkpoint_sha256": config["research"]["parent_sha256"],
                    "new_training_episodes": model["metadata"]["episode_count"],
                    "gradient_updates": model["metadata"]["training_step"]}.items():
                    client.set_model_version_tag(registry_name, version.version, key, value)
            if "alphazero" in config:
                for key, value in {"training_method": config["alphazero"]["phase"],
                    "training_run_status": model.get("training_status", "complete"),
                    "parent_checkpoint_sha256": config["alphazero"]["parent_sha256"],
                    "value_head_status": model["metadata"]["eval_summary"]["value_head_status"],
                    "search_simulations": config["alphazero"]["search"]["simulations"],
                    "new_training_episodes": model["metadata"]["episode_count"],
                    "gradient_updates": model["metadata"]["training_step"]}.items():
                    client.set_model_version_tag(registry_name, version.version, key, value)
            client.set_registered_model_alias(registry_name, model["id"], version.version)
            registered[model["id"]] = {"id": model["id"], "name": registry_name, "version": version.version,
                "run_id": rid, "sha256": model["sha256"]}
            client.set_terminated(rid)
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
    for folder, parent in parents.items():
        result = json.loads((Path(folder) / "result.json").read_text())
        client.set_terminated(parent, status="FAILED" if result.get("status") == "failed" else "FINISHED")
    for suite in suites:
        comparison = suite / "holdout/comparison.json"
        if not comparison.is_file():
            continue
        digest = hashlib.sha256(comparison.read_bytes()).hexdigest()
        if client.search_runs([exp_id], filter_string=f"tags.holdout_sha256 = '{digest}' and attributes.status = 'FINISHED'"):
            continue
        data = json.loads(comparison.read_text())
        rid = client.create_run(exp_id, tags={"mlflow.runName": suite.name + " / independent holdout",
            "holdout_sha256": digest, "candidate_id": data["selection"]["candidate_id"],
            "baseline_id": data["selection"]["baseline_id"]}).info.run_id
        try:
            client.log_param(rid, "opening_pairs", data["selection"]["pairs"])
            client.log_metric(rid, "score_improvement", data["score_improvement"])
            for identifier, result in data["results"].items():
                client.log_metric(rid, identifier + "_score", result["score"])
                for key in ("wins", "draws", "losses", "missed_immediate_win"):
                    client.log_metric(rid, identifier + "_" + key, result["counts"].get(key, 0))
                version = registered[identifier]
                client.set_model_version_tag(version["name"], version["version"], "holdout_score", result["score"])
                client.set_model_version_tag(version["name"], version["version"], "holdout_sha256", digest)
            client.log_artifacts(rid, str(comparison.parent), artifact_path="holdout")
            if (suite / "assessment.json").is_file():
                assessment = json.loads((suite / "assessment.json").read_text())
                for identifier, row in assessment.get("serial_latency", {}).get("summary", {}).items():
                    for key in ("mean_ms", "p50_ms", "p95_ms"):
                        client.log_metric(rid, identifier + "_serial_" + key, row[key])
                for identifier, row in assessment.get("common_opponent_diagnostics", {}).items():
                    for key in ("mean_decision_ms", "p50_decision_ms", "p95_decision_ms", "repetition_rate", "wall_fraction"):
                        client.log_metric(rid, identifier + "_" + key, row[key])
                    for key, value in row["decision_diagnostics"].items():
                        client.log_metric(rid, identifier + "_" + key, value)
                    for opponent in assessment["fixed_opponents"]:
                        cell = assessment["matrix"][identifier][opponent]
                        if cell:
                            for key in ("wins", "draws", "losses", "score"):
                                client.log_metric(rid, identifier + "_vs_" + opponent + "_" + key, cell[key])
            if (suite / "decision-probes").is_dir():
                client.log_artifacts(rid, str(suite / "decision-probes"), artifact_path="decision-probes")
            for filename in ("plan.json", "plan-initial.json", "assessment.json", "benchmark-report.md", "validation.json", "failed-attempt.json", "protocol-amendment.json", "analysis-plan.json", "evaluation-commands.json", "evaluation-time.json", "arena-roster.json", "run-evaluation.py", "assessment.py", "outcome-label-audit.json", "audit-outcomes.py", "decision-probes.py", "pytest-final.xml", "node-test-final.log"):
                if (suite / filename).is_file():
                    client.log_artifact(rid, str(suite / filename), artifact_path="evaluation_plan")
            client.set_terminated(rid)
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
    args.output.write_text(json.dumps({"models": list(registered.values())}, indent=2))
    print(json.dumps({"experiment_id": exp_id, "models": len(registered), "registry": str(args.output)}))


if __name__ == "__main__":
    main()
