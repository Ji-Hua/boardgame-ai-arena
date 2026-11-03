import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from agent_system.games.contracts import MR_JACK
from apps.jobs.controller import Controller
from apps.jobs.app import create_app
from apps.jobs.tests.test_controller import FakeGit, FakeDocker

class MrJackControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docker = FakeDocker()
        self.c = Controller(data_root=self.root, git=FakeGit(), docker=self.docker,
                            image="gpu", source_ref="feature/x", neural_image="mrjack:tested")
        self.c.projects.create(dict(id="jack", title="Mr. Jack", question="Baseline",
                                    game_ref=MR_JACK.wire()))

    def submit(self, **extra):
        return self.c.submit(dict(entrypoint="train_mrjack_ppo", project_id="jack",
                                  config=dict(timesteps=64,n_steps=64,batch_size=64), **extra))

    def test_game_profile_validation_and_api_filter(self):
        rows = self.c.entrypoints("jack")["entrypoints"]
        self.assertEqual(list(rows), ["train_mrjack_ppo", "evaluate_mrjack_policy"])
        self.assertTrue(rows["train_mrjack_ppo"]["enabled"])
        self.assertNotIn("train_mrjack_ppo", self.c.entrypoints("quoridor-legacy")["entrypoints"])
        client = TestClient(create_app(self.c))
        self.assertEqual(list(client.get("/api/v1/jobs/entrypoints?project_id=jack").json()["entrypoints"]), ["train_mrjack_ppo", "evaluate_mrjack_policy"])
        with self.assertRaises(ValueError):
            self.submit(follow_up=dict(gate="gate-v1"))
        with self.assertRaises(ValueError):
            self.c.submit(dict(entrypoint="train_mrjack_ppo", config={}))
        self.c.profile_images["research_neural_cpu"] = None
        self.assertFalse(self.c.entrypoints("jack")["entrypoints"]["train_mrjack_ppo"]["enabled"])
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
        from agent_system.training.mrjack_job import train
        from agent_system.training.jobs import Events
        train(job["spec"], run, Events(run))
        live = self.c.store.get(job["id"])
        self.docker.states[live["container"]] = dict(Running=False, ExitCode=0)
        self.c.tick()
        self.assertEqual(self.c.store.get(job["id"])["status"], "succeeded")
        self.assertEqual(self.c.metrics(job["id"])["metrics"][-1]["step"], 64)
        self.assertEqual(self.c.store.get(job["id"])["registered"], [])
        self.assertFalse((self.root/"catalog-jobs.json").exists())
        self.assertEqual(len(self.c.detail(job["id"])["research_artifacts"]), 1)
        (run/"policy.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.c.verify_research_policy(job, run)
        self.assertEqual(self.c.detail(job["id"])["research_artifacts"], [])
