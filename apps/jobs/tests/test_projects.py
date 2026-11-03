import json
import sqlite3
import tempfile
from pathlib import Path
import unittest
from fastapi.testclient import TestClient
from apps.jobs.app import create_app
from apps.jobs.controller import Controller
from apps.jobs.projects import LEGACY_PROJECT
from apps.jobs.store import SCHEMA, Store
from apps.jobs.tests.test_controller import FakeGit, FakeDocker, YAML
from agent_system.games.contracts import QUORIDOR, MR_JACK

class ProjectsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controller = Controller(data_root=self.root, git=FakeGit(), docker=FakeDocker(),
                                     image="test", source_ref="feature/x")

    def project(self, id, ref=QUORIDOR):
        return self.controller.projects.create(dict(id=id, title=id, question="compare policies", game_ref=ref.wire()))

    def test_projects_isolate_same_experiment_and_reject_wrong_game(self):
        self.project("a"); self.project("b"); self.project("jack", MR_JACK)
        a = self.controller.submit(dict(entrypoint="train_dqn", config=YAML, project_id="a", experiment="same"))
        b = self.controller.submit(dict(entrypoint="train_dqn", config=YAML, project_id="b", experiment="same"))
        self.assertEqual([j["id"] for j in self.controller.store.listing(project_id="a")], [a["id"]])
        self.assertEqual([j["id"] for j in self.controller.store.listing(project_id="b")], [b["id"]])
        self.assertEqual(a["spec"]["game_ref"], QUORIDOR.wire())
        for extra in (dict(project_id="jack"), dict(project_id="a", game_ref=MR_JACK.wire())):
            with self.assertRaises(ValueError):
                self.controller.submit(dict(entrypoint="train_dqn", config=YAML, **extra))
        with self.assertRaises(KeyError):
            self.controller.submit(dict(entrypoint="train_dqn", config=YAML, project_id="missing"))

    def test_old_database_migration_preserves_original_spec(self):
        path = self.root / "old.sqlite"
        db = sqlite3.connect(path)
        db.execute(SCHEMA)
        original = '{"source": {"commit": "historical"}}'
        db.execute("insert into jobs(id,entrypoint,spec,status,created) values(?,?,?,?,?)",
                   ("old", "train_dqn", original, "succeeded", 1))
        db.commit(); db.close()
        migrated = Store(path)
        self.assertEqual(migrated.get("old")["project_id"], LEGACY_PROJECT)
        self.assertEqual(migrated.db.execute("select spec from jobs").fetchone()[0], original)
        again = Store(path)
        self.assertEqual(again.get("old")["spec"], json.loads(original))
        migrated.db.close(); again.db.close()

    def test_api_requires_token_and_exposes_scope(self):
        token = self.root / "token"; token.write_text("secret")
        client = TestClient(create_app(self.controller, token))
        body = dict(id="experiment-a", title="A", question="Q", game_ref=QUORIDOR.wire())
        self.assertEqual(client.post("/api/v1/jobs/projects", json=body).status_code, 401)
        self.assertEqual(client.post("/api/v1/jobs/projects", json=body, headers={"X-Platform-Token":"secret"}).status_code, 200)
        submitted = client.post("/api/v1/jobs", json=dict(entrypoint="train_dqn", config=YAML, project_id="experiment-a"),
                                headers={"X-Platform-Token":"secret"})
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(client.get("/api/v1/jobs?project_id=quoridor-legacy").json()["jobs"], [])
        self.assertEqual(len(client.get("/api/v1/jobs?project_id=experiment-a").json()["jobs"]), 1)
        self.assertEqual(client.get("/api/v1/jobs?project_id=absent").status_code, 404)
        self.assertEqual(client.get("/api/v1/jobs/projects").status_code, 200)
