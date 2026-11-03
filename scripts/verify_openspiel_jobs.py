"""Isolated real-Docker acceptance. Explicit NEW data root; never starts a service."""
import argparse
import json
import os
from pathlib import Path
import time

from agent_system.games.contracts import KUHN_POKER
from apps.jobs.controller import Controller
from apps.jobs.docker import Docker, Git

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    git, docker = Git(Path.cwd()), Docker()
    def controller():
        return Controller(data_root=root, git=git, docker=docker,
            image="unused-quoridor-profile", openspiel_image=args.image,
            source_ref=args.source, uid_gid=f"{os.getuid()}:{os.getgid()}")
    c = controller()
    c.projects.create(dict(id="poker", title="Acceptance", question="Does the worker round-trip?",
                          game_ref=KUHN_POKER.wire()))
    def submit(algorithm, iterations):
        return c.submit(dict(entrypoint="train_openspiel", project_id="poker",
            config=dict(algorithm=algorithm, iterations=iterations, eval_every=20, seed=42),
            resources=dict(cpus=1, memory_gb=2, timeout_h=0.05)))
    def wait(identifier, expected):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            c.tick()
            job = c.store.get(identifier)
            if job["status"] not in ("queued", "preparing", "running"):
                if job["status"] != expected:
                    raise RuntimeError(json.dumps(job) + "\n" + c.log_text(identifier))
                return job
            time.sleep(0.2)
        raise TimeoutError(identifier)
    jobs = []
    try:
        for algorithm in ("cfr", "tabular_q_learning"):
            job = submit(algorithm, 100)
            jobs.append(job)
            c.tick()
            # Recreate the controller while its container is running: no re-export or rerun.
            c.store.db.close(); c.projects.db.close(); c.policies.db.close()
            c = controller()
            finished = wait(job["id"], "succeeded")
            detail = c.detail(job["id"])
            assert len(detail["research_artifacts"]) == 1
            assert finished["registered"] == []
        policies = c.policies.listing("poker")
        assert len(policies) == 2
        for opponent in (None, policies[1]["sha256"]):
            evaluation = c.submit(dict(entrypoint="evaluate_research_policy", project_id="poker",
                config=dict(candidate_sha=policies[0]["sha256"], opponent_sha=opponent, pairs=10, seed=123),
                resources=dict(cpus=1, memory_gb=2, timeout_h=0.05)))
            jobs.append(evaluation)
            wait(evaluation["id"], "succeeded")
            assert c.detail(evaluation["id"])["research_evaluation"]["games"] == 20
        cancelled = submit("cfr", 100000)
        jobs.append(cancelled)
        c.tick()
        c.cancel(cancelled["id"])
        wait(cancelled["id"], "cancelled")
        result = dict(source=git.resolve(args.source), image=docker.image_id(args.image),
            jobs=[dict(id=j["id"], status=c.store.get(j["id"])["status"],
                       metrics=c.metrics(j["id"])["metrics"][-1:]) for j in jobs])
        (root/"acceptance.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        for job in jobs:
            name = "qjob-" + job["id"]
            if docker.state(name) is not None:
                docker.stop(name, grace=1)
                docker.remove(name)

if __name__ == "__main__":
    main()
