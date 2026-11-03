import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from apps.jobs.controller import Controller
from apps.jobs.app import create_app
from apps.jobs.tests.test_controller import FakeGit, FakeDocker
from agent_system.games.contracts import MR_JACK, stable_hash
from agent_system.training.jobs import Events
from agent_system.training.mrjack_job import train
from agent_system.runtime.research_policy import validate_bundle
from agent_system.evaluation.mrjack_policies import run_evaluation

class PolicyStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = Controller(data_root=self.root, git=FakeGit(), docker=FakeDocker(),
                            image="gpu", source_ref="feature/x", neural_image="cpu")
        for project in ("jack", "other"):
            self.c.projects.create(dict(id=project, title=project, question="Q", game_ref=MR_JACK.wire()))
        self.job = self.c.submit(dict(entrypoint="train_mrjack_ppo", project_id="jack", config=dict(timesteps=64,n_steps=64,batch_size=64)))
        self.run = self.c.runs/self.job["id"]
        train(self.job["spec"], self.run, Events(self.run))
        self.c.register(self.job)
        self.sha = self.c.policies.listing("jack")[0]["sha256"]

    def test_registration_idempotency_scope_and_download(self):
        self.c.register(self.job)
        self.assertEqual(len(self.c.policies.listing("jack")), 1)
        self.assertEqual(self.c.policies.listing("other"), [])
        with self.assertRaises(KeyError):
            self.c.policies.get("other", self.sha)
        token = self.root/"token"; token.write_text("secret")
        client = TestClient(create_app(self.c, token))
        url = f"/api/v1/jobs/projects/jack/policies/{self.sha}/download"
        self.assertEqual(client.get(url).status_code, 401)
        response = client.get(url, headers={"X-Platform-Token":"secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, (self.run/"policy.json").read_bytes())
        self.assertEqual(client.get(url.replace("/jack/", "/other/"), headers={"X-Platform-Token":"secret"}).status_code, 404)
        self.assertEqual(len(client.get("/api/v1/jobs/projects/jack/policies").json()["policies"]), 1)
        (self.run/"policy.json").unlink()
        self.assertEqual(self.c.policies.get("jack", self.sha)[1], response.content)

    def test_evaluation_reuses_immutable_policy_and_rejects_cross_project(self):
        payload = dict(entrypoint="evaluate_mrjack_policy", project_id="jack",
                       config=dict(candidate_sha=self.sha, pairs=5, seed=123))
        job = self.c.submit(payload)
        with self.assertRaises(KeyError):
            self.c.submit({**payload, "project_id":"other"})
        reports = []
        for suffix in ("a", "b"):
            run = self.root/suffix; run.mkdir()
            spec = dict(job["spec"], data_root=str(self.root))
            run_evaluation(spec, run, Events(run))
            reports.append(json.loads((run/"research-evaluation.json").read_text()))
        self.assertEqual(reports[0], reports[1])
        training = [json.loads(s) for s in (self.run/"research-metrics.jsonl").read_text().splitlines()]
        self.assertIsNone(reports[0]["exploitability"])
        self.assertEqual(reports[0]["games"], 10)
        destination = self.c.runs/job["id"]
        (destination/"research-evaluation.json").write_text(json.dumps(reports[0]))
        self.assertEqual(self.c.register(job), [])
        self.c.store.update(job["id"], status="succeeded")
        self.assertEqual(self.c.detail(job["id"])["research_evaluation"]["games"], 10)
        reports[0]["mean_return"] = 123
        (destination/"research-evaluation.json").write_text(json.dumps(reports[0]))
        with self.assertRaises(ValueError):
            self.c.research_evaluation(job)

    def test_corruption_invalid_probabilities_and_identity_are_rejected(self):
        bundle = json.loads((self.run/"policy.json").read_text())
        bundle["files"][0]["name"] = "../secret"
        bundle["digest"] = stable_hash({k:v for k,v in bundle.items() if k != "digest"})
        with self.assertRaises(ValueError):
            validate_bundle(bundle)
        path = self.c.policies.path(self.sha)
        path.chmod(0o644); path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "corruption"):
            self.c.policies.get("jack", self.sha)
        with self.assertRaises(ValueError):
            self.c.policies.get("jack", "../secret")
        payload = dict(entrypoint="evaluate_mrjack_policy", project_id="jack",
                       config=dict(candidate_sha=self.sha, pairs=2))
        with self.assertRaisesRegex(ValueError, "corruption"):
            self.c.submit(payload)
