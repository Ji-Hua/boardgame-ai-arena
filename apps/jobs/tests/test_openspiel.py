import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from agent_system.games.contracts import KUHN_POKER
from apps.jobs.controller import Controller
from apps.jobs.app import create_app
from apps.jobs.tests.test_controller import FakeGit, FakeDocker

class OpenSpielControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docker = FakeDocker()
        self.c = Controller(data_root=self.root, git=FakeGit(), docker=self.docker,
                            image="gpu", source_ref="feature/x", openspiel_image="spiel:tested")
        self.c.projects.create(dict(id="poker", title="Poker", question="Baseline",
                                    game_ref=KUHN_POKER.wire()))

    def submit(self, **extra):
        return self.c.submit(dict(entrypoint="train_openspiel", project_id="poker",
                                  config=dict(iterations=2), **extra))

    def test_game_profile_validation_and_api_filter(self):
        rows = self.c.entrypoints("poker")["entrypoints"]
        self.assertEqual(list(rows), ["train_openspiel", "evaluate_research_policy"])
        self.assertTrue(rows["train_openspiel"]["enabled"])
        self.assertNotIn("train_openspiel", self.c.entrypoints("quoridor-legacy")["entrypoints"])
        client = TestClient(create_app(self.c))
        self.assertEqual(list(client.get("/api/v1/jobs/entrypoints?project_id=poker").json()["entrypoints"]), ["train_openspiel", "evaluate_research_policy"])
        with self.assertRaises(ValueError):
            self.submit(follow_up=dict(gate="gate-v1"))
        with self.assertRaises(ValueError):
            self.c.submit(dict(entrypoint="train_openspiel", config={}))
        self.c.profile_images["openspiel_cpu"] = None
        self.assertFalse(self.c.entrypoints("poker")["entrypoints"]["train_openspiel"]["enabled"])
        with self.assertRaisesRegex(ValueError, "not configured"):
            self.submit()

    def test_cpu_job_completion_metrics_and_no_legacy_registration(self):
        job = self.submit()
        self.assertEqual(job["spec"]["image"], "sha256:" + "d"*64)
        self.assertFalse(job["spec"]["resources"]["gpu"])
        self.c.tick()
        args = self.docker.runs[-1]
        self.assertEqual(args[-1], job["spec"]["image"])
        self.assertNotIn("--device", args)
        run = self.c.runs/job["id"]
        from agent_system.training.openspiel_job import run_openspiel
        from agent_system.training.jobs import Events
        run_openspiel(job["spec"], run, Events(run))
        live = self.c.store.get(job["id"])
        self.docker.states[live["container"]] = dict(Running=False, ExitCode=0)
        self.c.tick()
        self.assertEqual(self.c.store.get(job["id"])["status"], "succeeded")
        self.assertEqual(self.c.metrics(job["id"])["metrics"][-1]["step"], 2)
        self.assertEqual(self.c.store.get(job["id"])["registered"], [])
        self.assertFalse((self.root/"catalog-jobs.json").exists())
        self.assertEqual(len(self.c.detail(job["id"])["research_artifacts"]), 1)
        (run/"policy.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.c.verify_research_policy(job, run)
        self.assertEqual(self.c.detail(job["id"])["research_artifacts"], [])
