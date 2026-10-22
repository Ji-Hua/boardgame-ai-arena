"""Network-free gateway, catalog and launcher boundary regression tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import platform_entry as gateway


class GatewayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "dist"
        (self.dist / "assets").mkdir(parents=True)
        (self.dist / "index.html").write_text("<html>platform SPA</html>")
        (self.dist / "assets/app.js").write_text("console.log('platform')")
        self.backend = FastAPI()

        @self.backend.get("/api/example")
        def example():
            return {"integrated": True}

        @self.backend.get("/health")
        def health():
            return {"read_only": True}

        self.requests = []
        self.reply_status = 200
        self.reply_headers = {}
        self.reply_body = b'{"isolated":true}'

        def upstream(request):
            self.requests.append(request)
            return httpx.Response(self.reply_status, content=self.reply_body,
                                  headers=self.reply_headers)

        self.app = gateway.create_app(backend_app=self.backend, dist=self.dist,
                                      transport=httpx.MockTransport(upstream))
        self.client = self.enterContext(TestClient(self.app))

    def test_spa_deep_links_and_assets(self):
        for path in ("/", "/models", "/runs/mlflow%3Aabc", "/replays/some-run"):
            response = self.client.get(path, headers={"Accept": "text/html"})
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("platform SPA", response.text)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)
        for path in ("/assets/missing.js", "/missing.js", "/api/not-found", "/research-lab/missing"):
            response = self.client.get(path, headers={"Accept": "text/html"})
            self.assertEqual(response.status_code, 404, path)
            self.assertNotIn("platform SPA", response.text)
        self.assertEqual(self.requests, [])

    def test_static_symlink_cannot_escape(self):
        (self.root / "secret.txt").write_text("secret")
        (self.dist / "leak.txt").symlink_to(self.root / "secret.txt")
        self.assertEqual(self.client.get("/leak.txt").status_code, 404)

    def test_existing_api_is_same_process_and_read_only(self):
        self.assertEqual(self.client.get("/api/example").json(), {"integrated": True})
        self.assertEqual(self.client.get("/health").json(), {"read_only": True})
        self.assertEqual(self.client.post("/api/example").status_code, 405)
        self.assertEqual(self.client.get("/api").status_code, 404)
        self.assertEqual(self.requests, [])

    def test_slash_redirect_preserves_query_not_host(self):
        response = self.client.get("/research-lab?model_sha=a%2Fb&ply=2",
                                   headers={"Host": "attacker.invalid"}, follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/research-lab/?model_sha=a%2Fb&ply=2")
        self.assertEqual(self.requests, [])

    def test_lab_root_assets_and_api_strip_prefix_preserve_query(self):
        for path in ("/", "/assets/app.js?v=1", "/api/models?x=a%2Fb", "/health"):
            self.assertEqual(self.client.get("/research-lab" + path).status_code, 200)
            self.assertEqual(str(self.requests[-1].url), "http://127.0.0.1:8754" + path)

    def test_get_never_analyzes(self):
        for path in sorted(gateway.POST_ROUTES):
            response = self.client.get("/research-lab" + path)
            self.assertEqual(response.status_code, 405, path)
        self.assertEqual(self.requests, [])

    def test_post_explicit_allowlist_only_and_no_credentials_forwarded(self):
        payload = {"position": {"turn": 1}}
        response = self.client.post("/research-lab/api/analyze", json=payload,
                                    headers={"Authorization": "Bearer secret", "Cookie": "secret=1"})
        self.assertEqual(response.status_code, 200)
        request = self.requests[-1]
        self.assertEqual(json.loads(request.content), payload)
        self.assertNotIn("authorization", request.headers)
        self.assertNotIn("cookie", request.headers)
        self.assertEqual(request.headers["host"], "127.0.0.1:8754")
        for method in ("put", "patch", "delete", "options"):
            self.assertEqual(getattr(self.client, method)("/research-lab/api/analyze").status_code, 405)
        self.assertEqual(self.client.post("/research-lab/api/models").status_code, 405)
        self.assertEqual(self.client.post("/research-lab/api/deploy").status_code, 404)
        self.assertEqual(len(self.requests), 1)

    def test_no_arbitrary_target_or_encoded_traversal(self):
        for path in ("http://evil.invalid/api/models", "/evil.invalid/api/models",
                     "assets/%252e%252e/secret", "assets/%2e%2e%2fsecret",
                     "assets/a%5cb", "api/models/extra"):
            self.assertEqual(self.client.get("/research-lab/" + path).status_code, 404, path)
        self.assertEqual(self.requests, [])
        self.client.get("/research-lab/api/models?url=http://evil.invalid&target=http://evil.invalid")
        self.assertEqual(self.requests[-1].url.host, "127.0.0.1")

    def test_request_body_bound_including_streamed_without_length(self):
        response = self.client.post("/research-lab/api/analyze", content=b"x" * (gateway.MAX_BODY + 1))
        self.assertEqual(response.status_code, 413)
        def body():
            yield b"x" * gateway.MAX_BODY
            yield b"extra"
        response = self.client.post("/research-lab/api/analyze", content=body())
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.requests, [])

    def test_response_bound_and_redirect_rejected(self):
        self.reply_status = 302
        self.reply_headers = {"Location": "http://evil.invalid/", "Set-Cookie": "bad=1"}
        self.assertEqual(self.client.get("/research-lab/api/models").status_code, 502)
        self.assertEqual(len(self.requests), 1)
        self.reply_status = 200
        self.reply_body = b"x" * 30
        with patch.object(gateway, "MAX_RESPONSE", 20):
            response = self.client.get("/research-lab/api/models")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("location", response.headers)
        self.assertNotIn("set-cookie", response.headers)

    def test_network_error_and_timeout_are_bounded_errors(self):
        for error, expected in ((httpx.ConnectError("down"), 502), (httpx.ReadTimeout("slow"), 504)):
            def fail(request):
                raise error
            app = gateway.create_app(backend_app=self.backend, dist=self.dist,
                                      transport=httpx.MockTransport(fail))
            with TestClient(app) as client:
                self.assertEqual(client.get("/research-lab/api/models").status_code, expected)

    def test_only_configured_loopback_upstream(self):
        for value in ("http://evil.invalid:8754", "https://127.0.0.1:8754", "http://127.0.0.1:8754/api",
                      "http://user@127.0.0.1:8754", "http://127.0.0.1:8754?url=x", "http://127.0.0.1:80"):
            with self.assertRaises(ValueError, msg=value):
                gateway.create_app(backend_app=self.backend, dist=self.dist, lab_upstream=value)

    def test_missing_dist_errors_without_importing_backend(self):
        with patch.object(gateway, "dashboard_module", side_effect=AssertionError("must not import")):
            with self.assertRaisesRegex(RuntimeError, "npm run build"):
                gateway.create_app(dist=self.root / "missing")

    def test_private_backend_loader_does_not_replace_generic_backend(self):
        sentinel = types.ModuleType("backend")
        with patch.dict(sys.modules, {"backend": sentinel}):
            module = gateway.dashboard_module("platform_models")
            self.assertIs(sys.modules["backend"], sentinel)
            self.assertTrue(module.__name__.startswith(gateway.BACKEND_PACKAGE + "."))
            self.assertEqual(Path(module.__file__).resolve().parent, Path(__file__).resolve().parent)


class CatalogTests(unittest.TestCase):
    def test_deterministic_external_catalog_filters_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = dict(observation_version="dqn_obs_cnn_v1", action_space_version="dqn_action_v1",
                            action_count=209, model_config=dict(model_arch="cnn", algorithm="policy_distillation"))
            models = []
            for number, alias in enumerate(("approved", "collision", "collision", "missing")):
                checkpoint = root / f"{number}.pt"
                if alias != "missing":
                    checkpoint.write_bytes(str(number).encode())
                models.append(dict(id=alias, sha256=hashlib.sha256(str(number).encode()).hexdigest(),
                                   checkpoint=str(checkpoint), metadata=metadata, training_status="research"))
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps(dict(models=models)))
            original = catalog.read_bytes()
            index = gateway.dashboard_module("platform_models").ModelIndex(
                data_root=root, catalog_paths=[catalog])
            output = root / "platform-p1-p2/catalogs"
            with patch.object(index, "_hash", side_effect=AssertionError("must be static")):
                first = gateway.research_catalog(output, index=index)
                second = gateway.research_catalog(output, index=index)
            self.assertEqual(first, second)
            self.assertEqual(catalog.read_bytes(), original)
            result = json.loads(first.read_text())
            self.assertEqual(len(result["models"]), 1)
            model = result["models"][0]
            self.assertEqual(model["id"], models[0]["sha256"])
            self.assertEqual(model["metadata"], metadata)
            self.assertEqual(model["training_status"], "research")
            self.assertEqual(model["source_evidence"][0]["source"]["path"], str(catalog))
            self.assertIn("static_eligibility_only", model["static_analysis_reasons"])
            self.assertEqual(index.listing()["models"][0]["requires_isolated_research_catalog"], True)
            with self.assertRaises(ValueError):
                gateway.research_catalog(root / "not-approved", index=index)

    def test_launcher_syntax_and_missing_dist_preflight(self):
        launcher = gateway.ROOT / "scripts/start_unified_platform.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ, PLATFORM_FRONTEND_DIST=temporary, QUORIDOR_PYTHON=sys.executable)
            result = subprocess.run(["bash", str(launcher)], env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("npm run build", result.stderr)


if __name__ == "__main__":
    unittest.main()