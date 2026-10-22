import asyncio
import os
from unittest.mock import patch
import unittest

import httpx

from backend.main import create_app


SHA = "a" * 64


class RuntimeStub:
    def __init__(self):
        self.failure = None
        self.released = []

    async def create_research(self, sha256, room_id, seat):
        if sha256 != SHA:
            raise RuntimeError("Checkpoint not approved or missing")
        return dict(session_id=room_id, token="runtime-secret", identity=dict(
            sha256=sha256, actual_sha256=sha256, room_id=room_id, agent_seat=seat,
            runtime_version="test", agent_spec={"policy_type": "top_k", "policy_params": {"k": 1}}))

    async def research_action(self, binding, state, legal):
        if self.failure:
            raise self.failure
        pawns = [action for action in legal if action["type"] == "pawn"]
        seat = binding["identity"]["agent_seat"]
        return min(pawns, key=lambda action: action["target"][1] if seat == 2 else -action["target"][1])

    async def destroy_research(self, binding):
        self.released.append(binding["session_id"])


class ResearchPlayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        environment = patch.dict(os.environ, QUORIDOR_RESEARCH_SERVICE_KEY="test")
        environment.start()
        self.addCleanup(environment.stop)
        self.app = create_app()
        self.runtime = RuntimeStub()
        self.app.state.agent_adapter = self.runtime
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
            headers={"X-Research-Gateway": "test"})
        self.addAsyncCleanup(self.client.aclose)

    async def create(self, seat=1):
        response = await self.client.post("/api/research/games", json=dict(sha256=SHA, human_seat=seat))
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.identifier = data["game_id"]
        self.headers = {"Authorization": "Bearer " + data["token"]}
        self.url = "/api/research/games/" + self.identifier
        await self.drain()
        return await self.snapshot()

    async def drain(self):
        task = self.app.state.research_games[self.identifier].task
        if task:
            await task

    async def snapshot(self):
        response = await self.client.get(self.url, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        return response.json()

    async def move(self, state, action):
        result = await self.client.post(self.url + "/actions", headers=self.headers,
            json=dict(expected_step=state["step_count"], type=action["type"], target=action["target"]))
        await self.drain()
        return result

    async def test_complete_game_both_seats_and_identity(self):
        for seat in (1, 2):
            state = await self.create(seat)
            self.assertEqual(state["identity"]["actual_sha256"], SHA)
            self.assertEqual(state["state"]["current_player"], seat)
            for turn in range(40):
                if state["phase"] == "finished":
                    break
                pawns = [action for action in state["legal_actions"] if action["type"] == "pawn"]
                action = min(pawns, key=lambda action: action["target"][1] if seat == 2 else -action["target"][1])
                response = await self.move(state, action)
                self.assertEqual(response.status_code, 200, response.text)
                state = await self.snapshot()
            self.assertEqual(state["phase"], "finished")
            self.assertEqual(state["result"]["termination"], "goal")
            self.assertTrue(state["actions"])
            self.assertEqual(state["initial_state"]["walls_remaining"], {"1": 10, "2": 10})
            await self.client.delete(self.url, headers=self.headers)
        self.assertEqual(len(self.runtime.released), 2)

    async def test_wall_precision_refresh_illegal_and_stale(self):
        state = await self.create()
        result = await self.move(state, dict(type="vertical", target=[7, 7]))
        self.assertEqual(result.status_code, 200)
        state = await self.snapshot()
        self.assertEqual(state["state"]["walls"], [dict(x=7, y=7, orientation="vertical")])
        self.assertEqual(int(state["state"]["wall_state"]["vertical_heads"]), 1 << 63)
        self.assertEqual(state["state"]["walls_remaining"]["1"], 9)
        invalid = await self.move(state, dict(type="vertical", target=[7, 7]))
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual((await self.snapshot())["step_count"], state["step_count"])
        stale = await self.move(dict(step_count=0), dict(type="pawn", target=[4, 1]))
        self.assertEqual(stale.status_code, 409)
        self.assertEqual((await self.client.get(self.url)).status_code, 404)

    async def test_timeout_and_connection_failure_no_fallback(self):
        for failure in (TimeoutError("Agent timed out"), RuntimeError("Connection refused")):
            state = await self.create()
            self.runtime.failure = failure
            response = await self.move(state, dict(type="pawn", target=[4, 1]))
            self.assertEqual(response.status_code, 200)
            state = await self.snapshot()
            self.assertEqual(state["step_count"], 1)
            self.assertIn(str(failure), state["error"])
            self.assertFalse(state["pending"])
            self.runtime.failure = None
            await self.client.post(self.url + "/retry", headers=self.headers)
            await self.drain()
            self.assertEqual((await self.snapshot())["step_count"], 2)
            await self.client.delete(self.url, headers=self.headers)

    async def test_missing_weights_create_failure_and_surrender(self):
        missing = await self.client.post("/api/research/games", json=dict(sha256="b" * 64))
        self.assertEqual(missing.status_code, 503)
        self.assertFalse(self.app.state.research_games)
        await self.create()
        result = await self.client.post(self.url + "/surrender", headers=self.headers)
        self.assertEqual(result.json()["result"], dict(winner_seat=2, termination="surrender"))

    async def test_gateway_and_session_isolation_expiry(self):
        forbidden = await self.client.post("/api/research/games",
            headers={"X-Research-Gateway": "wrong"}, json=dict(sha256=SHA))
        self.assertEqual(forbidden.status_code, 403)
        await self.create()
        unauthorized = await self.client.get(self.url, headers={"Authorization": "Bearer wrong"})
        self.assertEqual(unauthorized.status_code, 404)
        self.app.state.research_games[self.identifier].created = 0
        self.assertEqual((await self.client.get(self.url, headers=self.headers)).status_code, 410)

    async def test_research_step_limit_is_explicit_and_releases_instance(self):
        state = await self.create()
        with patch("backend.transport.rest.research_routes.MAX_GAME_PLIES", 1):
            response = await self.move(state, dict(type="pawn", target=[4, 1]))
        self.assertEqual(response.status_code, 200)
        final = await self.snapshot()
        self.assertEqual(final["result"]["termination"], "forced")
        self.assertIn("resource limit", final["error"])
        self.assertEqual(final["step_count"], 1)
        self.assertEqual(len(self.runtime.released), 1)


class ResearchAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_http_transport_errors_and_changed_identity(self):
        from backend.adapters.agent_service_adapter import AgentServiceAdapter
        binding = dict(session_id="session", token="token", identity=dict(room_id="room", agent_seat=2,
            sha256=SHA, actual_sha256=SHA))
        with patch.dict(os.environ, QUORIDOR_RESEARCH_SERVICE_KEY="test"):
            for failure, expected in ((httpx.ReadTimeout("slow"), TimeoutError),
                                      (httpx.ConnectError("refused"), RuntimeError)):
                def handler(request):
                    raise failure
                adapter = AgentServiceAdapter("http://test-runtime")
                await adapter._async_client.aclose()
                adapter._async_client = httpx.AsyncClient(base_url="http://test-runtime", transport=httpx.MockTransport(handler))
                try:
                    with self.assertRaises(expected):
                        await adapter.research_action(binding, {}, [])
                finally:
                    await adapter._async_client.aclose()
                    adapter._client.close()
            adapter = AgentServiceAdapter("http://test-runtime")
            await adapter._async_client.aclose()
            adapter._async_client = httpx.AsyncClient(base_url="http://test-runtime", transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=dict(action={}, identity={"sha256": "b" * 64}))))
            try:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    await adapter.research_action(binding, {}, [])
            finally:
                await adapter._async_client.aclose()
                adapter._client.close()