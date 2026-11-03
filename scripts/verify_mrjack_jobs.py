"""Isolated CPU-container acceptance; never mutates live services or rosters."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import zipfile
from agent_system.games.contracts import MR_JACK
from apps.jobs.controller import Controller
from apps.jobs.docker import Docker,Git

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--image",required=True)
    p.add_argument("--source",required=True)
    args=p.parse_args()
    root=args.data_root.resolve();root.mkdir(parents=True,exist_ok=False)
    git,docker=Git(Path.cwd()),Docker()
    def controller():
        return Controller(data_root=root,git=git,docker=docker,image="unused",
            neural_image=args.image,source_ref=args.source,uid_gid=f"{os.getuid()}:{os.getgid()}")
    c=controller()
    c.projects.create(dict(id="jack",title="Mr. Jack pilot",question="End-to-end CPU research",
                          game_ref=MR_JACK.wire()))
    jobs=[];reports=[]
    def wait(identifier,expected="succeeded"):
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            c.tick();j=c.store.get(identifier)
            if j["status"] not in ("queued","preparing","running"):
                if j["status"]!=expected:raise RuntimeError(json.dumps(j)+"\n"+c.log_text(identifier))
                return j
            time.sleep(.25)
        raise TimeoutError(identifier)
    def submit(entrypoint,config):
        j=c.submit(dict(entrypoint=entrypoint,project_id="jack",config=config,
            resources=dict(cpus=2,memory_gb=2,timeout_h=.1)))
        jobs.append(j);return j
    try:
        for seed in (7,42,101):
            j=submit("train_mrjack_ppo",dict(seed=seed,timesteps=128,n_steps=64,batch_size=64))
            c.tick()
            c.store.db.close();c.projects.db.close();c.policies.db.close();c=controller()
            wait(j["id"])
            assert c.store.get(j["id"])["registered"]==[]
            policy=next(p for p in c.policies.listing("jack") if p["job_id"]==j["id"])
            archive=zipfile.ZipFile(io.BytesIO(c.policies.bundle("jack",policy["sha256"])))
            assert set(archive.namelist())=={"policy.json","actor.npz"}
            for opponent in ("random","heuristic"):
                e=submit("evaluate_mrjack_policy",dict(candidate_sha=policy["sha256"],
                    opponent=opponent,pairs=10,seed=123))
                wait(e["id"]);report=c.detail(e["id"])["research_evaluation"]
                assert report["games"]==20 and report["exploitability"] is None
                assert set(report["role_returns"])=={"detective","jack"}
                reports.append(report)
            audit=c.research_eligibility.record("jack",dict(sha256=policy["sha256"],
                role="detective",target="research-play",evaluation_job=e["id"],
                decision="approve",note="Isolated acceptance only; no deployment"))
            assert audit["deployment_changed"] is False
            assert audit["agent_spec"]["role"]=="detective"
        # Restore actual gameplay from the same immutable numeric policy.
        from apps.gameplay.research import ResearchGames
        games=ResearchGames(root/"games"/"acceptance.sqlite",root)
        created=games.create("jack",MR_JACK.wire(),"detective",policy["sha256"])
        view=created["view"]
        games.db.close();games=ResearchGames(root/"games"/"acceptance.sqlite",root)
        while view["status"]=="active":
            action=view["legal_actions"][0]["id"] if view["actor"]=="detective" else None
            view=games.step(created["id"],created["token"],view["revision"],action)
        assert view["status"]=="finished"
        assert games.replay(created["id"],created["token"])["frames"][-1]["observation"]==view["observation"]
        assert "identity" not in games.view(created["id"],created["watch_token"])["observation"]
        cancelled=submit("train_mrjack_ppo",dict(timesteps=999936,n_steps=64,batch_size=64))
        c.tick();c.cancel(cancelled["id"]);wait(cancelled["id"],"cancelled")
        assert not any(p["job_id"]==cancelled["id"] for p in c.policies.listing("jack"))
        result=dict(source=git.resolve(args.source),image=docker.image_id(args.image),
                    jobs=[dict(id=j["id"],status=c.store.get(j["id"])["status"]) for j in jobs],
                    reports=reports,gameplay=dict(status=view["status"],revision=view["revision"]))
        (root/"acceptance.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
    finally:
        for j in jobs:
            name="qjob-"+j["id"]
            if docker.state(name) is not None:docker.stop(name,grace=1);docker.remove(name)
if __name__=="__main__":main()
