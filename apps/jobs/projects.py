"""Project records owned by the Job controller. No inferred cross-game identities."""
import json
import os
import tempfile
import re
import sqlite3
import threading
from pathlib import Path
from agent_system.games.contracts import QUORIDOR
from agent_system.games.registry import validate_ref

LEGACY_PROJECT = "quoridor-legacy"
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")

class Projects:
    def __init__(self, path):
        self.manifest = Path(path).with_suffix(".json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("pragma journal_mode=wal")
            self.db.execute("create table if not exists projects(id text primary key, title text not null, question text not null, game_ref text not null)")
            self.db.execute("insert or ignore into projects values(?,?,?,?)",
                            (LEGACY_PROJECT, "Quoridor 历史研究", "兼容既有任务与证据", json.dumps(QUORIDOR.wire())))

        self.publish()

    def publish(self):
        with self.lock:
            data=json.dumps(dict(schema_version=1,projects=self.listing()),ensure_ascii=False)
            fd,temp=tempfile.mkstemp(prefix=".projects-",dir=self.manifest.parent)
            try:
                with os.fdopen(fd,"w") as out:
                    out.write(data);out.flush();os.fsync(out.fileno())
                os.chmod(temp,0o644)  # public project catalog must be readable by the gameplay UID
                os.replace(temp,self.manifest)
            finally:
                if os.path.exists(temp):os.unlink(temp)

    @staticmethod
    def row(raw):
        if raw is None:
            raise KeyError("Unknown project")
        value = dict(raw)
        value["game_ref"] = json.loads(value["game_ref"])
        validate_ref(value["game_ref"])
        return value

    def get(self, identifier):
        with self.lock:
            return self.row(self.db.execute("select * from projects where id=?", (identifier,)).fetchone())

    def listing(self):
        with self.lock:
            projects=[]
            for row in self.db.execute("select * from projects order by id").fetchall():
                try:projects.append(self.row(row))
                except ValueError:continue  # retained historical data is not an active supported project
            return projects

    def create(self, value):
        if set(value) != {"id", "title", "question", "game_ref"}:
            raise ValueError("Project requires id, title, question and game_ref")
        if not isinstance(value["id"], str) or not SLUG.fullmatch(value["id"]):
            raise ValueError("Invalid project id")
        for key, limit in (("title", 160), ("question", 4000)):
            if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > limit:
                raise ValueError("Invalid project " + key)
        ref = validate_ref(value["game_ref"])
        with self.lock:
            try:
                self.db.execute("insert into projects values(?,?,?,?)",
                                (value["id"], value["title"].strip(), value["question"].strip(), json.dumps(ref.wire())))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Project already exists; choose another id") from exc
        self.publish()
        return self.get(value["id"])
