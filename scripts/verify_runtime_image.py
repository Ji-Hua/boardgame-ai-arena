"""Isolated CPU image acceptance; never starts a live deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def verify(image, data_root, catalog, model_id):
    data_root = data_root.resolve(strict=True)
    catalog = catalog.resolve(strict=True)
    if not catalog.is_relative_to(data_root):
        raise ValueError("catalog must be inside data root")
    rows = json.loads(catalog.read_text())["models"]
    row, = [r for r in rows if r["id"] == model_id]
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = catalog.parent / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to(data_root):
        raise ValueError("checkpoint must be inside read-only data root")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise ValueError("checkpoint SHA mismatch before preview")
    info = json.loads(docker("image", "inspect", image))[0]
    image_id = info["Id"]
    report = {"image_id": image_id, "source_commit": info["Config"]["Labels"].get(
        "org.opencontainers.image.revision"), "model_id": model_id, "checkpoint_sha256": digest}
    containers = []
    try:
        endpoints = {}
        for component, module, port in [("runtime", "agent_system.runtime.service.server", 8743),
                                        ("lab", "agent_system.analysis.app", 8744)]:
            name = "quoridor-migration-check-" + component + "-" + uuid.uuid4().hex[:10]
            args = ["run", "-d", "--name", name, "--read-only", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--memory=2g", "--cpus=1",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m", "-p", f"127.0.0.1::{port}",
                    "--mount", f"type=bind,src={data_root},dst={data_root},readonly",
                    "-e", "LAB_MODEL_CATALOG=" + str(catalog), image_id,
                    "python", "-m", "uvicorn", module + ":app", "--host", "0.0.0.0", "--port", str(port)]
            container = docker(*args)
            containers.append(container)
            mapping = docker("port", container, str(port) + "/tcp")
            if not mapping.startswith("127.0.0.1:"):
                raise RuntimeError("preview must bind loopback only")
            base = "http://" + mapping
            for attempt in range(30):
                try:
                    assert request(base, "/health")["status"] == "ok"
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise RuntimeError(component + " failed startup")
            endpoints[component] = base
        types = request(endpoints["runtime"], "/agent/types")["agent_types"]
        if len(types) < 4:
            raise RuntimeError("builtin agent definitions missing")
        with urllib.request.urlopen(endpoints["lab"] + "/", timeout=10) as response:
            assert b"<html" in response.read().lower()
        result = request(endpoints["lab"], "/api/analyze",
                         {"position": {}, "algorithm": "dqn", "model_id": model_id})
        assert result["provenance"]["checkpoint_sha256"] == digest
        assert result["selected_id"] in {r["id"] for r in result["candidates"]}
        assert result["complete"] is True
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        report.update(status="passed", runtime_agent_types=len(types),
                      legal_candidates=len(result["candidates"]), selected_id=result["selected_id"],
                      output_semantics=result["neural"]["model"]["output_semantics"])
    finally:
        failures = []
        for container in containers:
            result = subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)
            if result.returncode:
                failures.append(container)
        if failures:
            raise RuntimeError("preview cleanup failed: " + ", ".join(failures))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.image, args.data_root, args.catalog, args.model_id), indent=2))
