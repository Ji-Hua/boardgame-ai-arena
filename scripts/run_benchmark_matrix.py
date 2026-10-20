"""Run/resume a pinned, seat-balanced benchmark; artifacts live outside Git."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
import os
import platform
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"backend-server")]
from agent_system.evaluation.arena.matrix import (
    make_openings, standard_opening, play_block, summarize, stable_hash, worker_init,
)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def source_files():
    folders = ["agent_system/evaluation/arena", "agent_system/runtime/inference",
               "agent_system/runtime/service", "agent_system/training/dqn",
               "agent_system/definition", "backend-server/backend/adapters", "engine"]
    files = [p for folder in folders for p in (ROOT/folder).rglob("*")
             if p.suffix in (".py", ".rs", ".toml") and "tests" not in p.parts and "target" not in p.parts]
    return sorted(files + [Path(__file__).resolve()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=500)
    args = parser.parse_args()
    if min(args.pairs, args.workers, args.max_plies) <= 0:
        parser.error("pairs, workers and max-plies must be positive")
    roster = json.loads(args.roster.read_text())
    specs = roster["agents"]
    if len(specs)<2 or len({a["id"] for a in specs})!=len(specs):
        parser.error("Need at least two unique agents")
    for spec in specs:
        if spec["algo"]["type"] == "dqn":
            params = spec["algo"]["params"]
            with Path(params["checkpoint"]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest()==params["sha256"]
    paths = source_files()
    digest = stable_hash({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    opening_book = make_openings(args.pairs, args.seed)
    manifest = {
        "protocol": "quoridor-paired-opening-v1", "roster": roster, "book": opening_book,
        "source_sha256": digest, "git_head": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "max_plies": args.max_plies, "seed": args.seed, "pairs": args.pairs,
        "versions": {p: importlib.metadata.version(p) for p in ["torch","numpy","PyYAML"]},
        "python": platform.python_version(), "platform": platform.platform(),
    }
    import quoridor_engine
    engine_module = Path(quoridor_engine.__file__)
    binaries = list(engine_module.parent.glob("*.so")) if engine_module.suffix == ".py" else [engine_module]
    manifest["engine_binaries"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = out/"manifest.json"
    if existing.exists():
        if json.loads(existing.read_text()) != manifest:
            raise ValueError("Resume rejected: manifest, source or models changed; use a new output directory")
    else:
        atomic_json(existing, manifest)
        with tarfile.open(out/"source.tar.gz", "w:gz") as archive:
            for p in paths:
                archive.add(p, arcname=p.relative_to(ROOT))
    (out/"blocks").mkdir(exist_ok=True)
    tasks = []
    for a, b in itertools.combinations(specs,2):
        for opening in [standard_opening(), *opening_book]:
            key = stable_hash([a["id"], b["id"], opening["id"]])[:20]
            tasks.append((key, a, b, opening))
    blocks = []
    pending = []
    for key, a, b, opening in tasks:
        path = out/"blocks"/(key+".json.gz")
        if path.exists():
            with gzip.open(path,"rt") as f:
                block = json.load(f)
            if block["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("Resume rejected: block manifest mismatch")
            blocks.append(block)
        else:
            pending.append((key,a,b,opening))
    started = time.monotonic()
    def publish():
        report = summarize(specs,blocks,len(tasks))
        report.update(run_id=out.name, protocol=manifest["protocol"],
                      manifest_sha256=manifest["manifest_sha256"], source_sha256=digest,
                      pairs_per_match=args.pairs, max_plies=args.max_plies,
                      unique_openings=len({o["state_hash"] for o in opening_book}),
                      updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        atomic_json(out/"report.json",report)
        atomic_json(out.parent/"latest.json",{"run_id":out.name})
    publish()
    print(json.dumps({"run_id":out.name,"agents":len(specs),"blocks":len(tasks),"remaining":len(pending)}),flush=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,mp_context=multiprocessing.get_context("spawn"),initializer=worker_init
    ) as executor:
        futures = {executor.submit(play_block,a,b,opening,args.max_plies,args.seed):key
                   for key,a,b,opening in pending}
        for future in concurrent.futures.as_completed(futures):
            block = future.result()
            key = futures[future]
            block.update(block_id=key,manifest_sha256=manifest["manifest_sha256"])
            path = out/"blocks"/(key+".json.gz")
            temp = path.with_suffix(".tmp")
            with gzip.open(temp,"wt") as f:
                json.dump(block,f,allow_nan=False)
            temp.replace(path)
            blocks.append(block)
            if len(blocks)%10==0 or len(blocks)==len(tasks):
                publish()
                print(json.dumps({"completed":len(blocks),"total":len(tasks),"seconds":round(time.monotonic()-started,1)}),flush=True)
    publish()
    print("COMPLETE "+str(out/"report.json"),flush=True)


if __name__=="__main__":
    main()
