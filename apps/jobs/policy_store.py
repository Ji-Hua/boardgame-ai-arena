"""Immutable research policies, separate from the legacy Torch model store."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

from agent_system.runtime.research_policy import validate_bundle, runtime

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
        validate_bundle(bundle)
        runtime(bundle, Path(source).parent)
        if bundle["project_id"] != job["project_id"] or bundle["game_ref"] != job["spec"]["game_ref"]:
            raise ValueError("Policy project/game mismatch")
        from agent_system.definition.research_spec import policy_spec
        from agent_system.games.registry import create_game
        specs=[policy_spec(bundle,sha,role) for role in create_game(bundle["game_ref"]).roles]
        metadata = dict(agent_specs=specs,sha256=sha, project_id=job["project_id"], job_id=job["id"],
            game_ref=bundle["game_ref"], source=bundle["source"], algorithm=bundle["config"]["algorithm"],
            policy_kind=bundle["policy_kind"], output_semantics=bundle["output_semantics"])
        with self.lock:
            for dependency in bundle.get('files',[]):
                source_path=Path(source).parent/dependency['name']
                model=source_path.read_bytes()
                if hashlib.sha256(model).hexdigest()!=dependency['sha256']:
                    raise ValueError('Model dependency SHA mismatch')
                target_model=self.root/(dependency['sha256']+'.npz')
                self._publish(model,target_model)
            target = self.path(sha)
            self._publish(data,target)
            self.db.execute("insert or ignore into policies values(?,?,?,?)",
                (job["project_id"], sha, job["id"], json.dumps(metadata)))
        return metadata

    def _publish(self,data,target):
        if target.is_symlink():
            raise ValueError('Symbolic link in policy store')
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
        validate_bundle(bundle)
        meta = json.loads(row["metadata"])
        if bundle["project_id"] != project_id or bundle["game_ref"] != meta["game_ref"]:
            raise ValueError("Policy provenance mismatch")
        runtime(bundle,self.root,stored=True)  # verifies dependent numeric weights too
        return meta, data

    def bundle(self, project_id, sha):
        import io
        import zipfile
        _, data = self.get(project_id, sha)
        bundle = json.loads(data)
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("policy.json", data)
            for item in bundle.get("files", []):
                path = self.root / (item["sha256"] + ".npz")
                if path.is_symlink(): raise ValueError("Symbolic model dependency")
                model = path.read_bytes()
                if hashlib.sha256(model).hexdigest() != item["sha256"]:
                    raise ValueError("Model dependency SHA mismatch")
                archive.writestr(item["name"], model)
        return out.getvalue()
