"""Immutable research policies, separate from the legacy Torch model store."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

from agent_system.training.openspiel_job import PolicyAgent

SHA = re.compile(r"[0-9a-f]{64}\Z")

class PolicyStore:
    def __init__(self, root):
        self.root = Path(root).resolve() / "policies"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root/"catalog.sqlite", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("create table if not exists policies(project_id text, sha256 text, job_id text, metadata text not null, primary key(project_id, sha256, job_id))")

    def path(self, sha):
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError("Policy identity must be a full lowercase SHA-256")
        path = self.root / (sha + ".json")
        if path.is_symlink():
            raise ValueError("Policy store must not contain symbolic links")
        return path

    def register(self, job, source):
        data = Path(source).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        bundle = json.loads(data)
        PolicyAgent(bundle)
        if bundle["project_id"] != job["project_id"] or bundle["game_ref"] != job["spec"]["game_ref"]:
            raise ValueError("Policy project/game mismatch")
        metadata = dict(sha256=sha, project_id=job["project_id"], job_id=job["id"],
            game_ref=bundle["game_ref"], source=bundle["source"], algorithm=bundle["config"]["algorithm"],
            policy_kind=bundle["policy_kind"], output_semantics=bundle["output_semantics"])
        with self.lock:
            target = self.path(sha)
            if not target.exists():
                fd, temp = tempfile.mkstemp(prefix=".policy-", dir=self.root)
                try:
                    with os.fdopen(fd, "wb") as out:
                        out.write(data); out.flush(); os.fsync(out.fileno())
                    os.chmod(temp, 0o444)
                    try:
                        os.link(temp, target)  # atomic no-overwrite publication
                    except FileExistsError:
                        pass
                finally:
                    os.unlink(temp)
            if target.read_bytes() != data:
                raise ValueError("Immutable policy store corruption")
            self.db.execute("insert or ignore into policies values(?,?,?,?)",
                (job["project_id"], sha, job["id"], json.dumps(metadata)))
        return metadata

    def listing(self, project_id):
        with self.lock:
            rows = self.db.execute("select metadata from policies where project_id=? order by sha256,job_id",
                                   (project_id,)).fetchall()
        return [json.loads(row["metadata"]) for row in rows]

    def get(self, project_id, sha):
        path = self.path(sha)
        with self.lock:
            row = self.db.execute("select metadata from policies where project_id=? and sha256=? order by job_id limit 1",
                                  (project_id, sha)).fetchone()
        if row is None:
            raise KeyError("Policy not registered in this project")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Immutable policy store corruption")
        bundle = json.loads(data)
        PolicyAgent(bundle)
        meta = json.loads(row["metadata"])
        if bundle["project_id"] != project_id or bundle["game_ref"] != meta["game_ref"]:
            raise ValueError("Policy provenance mismatch")
        return meta, data
