"""Validate an isolated unified platform and Lab pair; no live services."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid
from verify_runtime_image import docker, request


def verify(research_image, runtime_image, data_root, artifacts, catalog):
    data_root = data_root.resolve(strict=True)
    artifacts = artifacts.resolve(strict=True)
    catalog = catalog.resolve(strict=True)
    if not catalog.is_relative_to(data_root):
        raise ValueError("catalog outside data root")
    research = json.loads(docker("image", "inspect", research_image))[0]
    runtime = json.loads(docker("image", "inspect", runtime_image))[0]
    network = "quoridor-migration-check-" + uuid.uuid4().hex[:10]
    containers = []
    docker("network", "create", network)
    try:
        common = ["--network", network, "--read-only", "--cap-drop=ALL",
                  "--security-opt=no-new-privileges", "--memory=2g", "--cpus=1",
                  "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
                  "--mount", f"type=bind,src={data_root},dst={data_root},readonly"]
        containers.append(docker("run", "-d", *common, "--network-alias", "lab",
            "-e", "LAB_MODEL_CATALOG=" + str(catalog), runtime["Id"],
            "python", "-m", "uvicorn", "agent_system.analysis.app:app",
            "--host", "0.0.0.0", "--port", "8744"))
        containers.append(docker("run", "-d", *common, "-p", "127.0.0.1::8741",
            "--mount", f"type=bind,src={artifacts},dst=/data/training_artifacts,readonly",
            "-e", "QUORIDOR_DATA_ROOT=" + str(data_root),
            "-e", "TRAINING_DASHBOARD_ARTIFACT_ROOTS=/data/training_artifacts/dqn:/data/training_artifacts",
            "-e", "MLFLOW_TRACKING_URI=http://192.168.68.88:8742", research["Id"]))
        base = "http://" + docker("port", containers[-1], "8741/tcp")
        for attempt in range(30):
            try:
                assert request(base, "/health")["status"] == "ok"
                assert request(base, "/research-lab/health")["status"] == "ok"
                break
            except (OSError, ValueError):
                time.sleep(1)
        else:
            raise RuntimeError("platform or Lab startup failed")
        for route in ("/", "/training", "/play", "/research-lab/"):
            req = urllib.request.Request(base + route, headers={"Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=10) as response:
                assert b"<html" in response.read().lower()
        listing = request(base, "/api/platform/models")
        assert listing["models"], "model index unexpectedly empty"
        result = request(base, "/research-lab/api/analyze", {"position": {}, "algorithm": "greedy"})
        assert result["selected_id"] in {r["id"] for r in result["candidates"]}
        return {"status": "passed", "research_image_id": research["Id"],
                "runtime_image_id": runtime["Id"],
                "source_commit": research["Config"]["Labels"]["org.opencontainers.image.revision"],
                "indexed_models": len(listing["models"]), "lab_proxy_analysis": "passed",
                "spa_routes": ["/", "/training", "/play"],
                "note": "HTML/API acceptance; no browser interaction or playable session claimed"}
    finally:
        for container in containers:
            subprocess.run(["docker", "rm", "-f", container], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "network", "rm", network], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research_image")
    parser.add_argument("runtime_image")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.research_image, args.runtime_image, args.data_root,
                            args.artifacts, args.catalog), indent=2))
