"""Bounded, capability-scoped research instances; never mutate the roster."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

from agent_system.definition.agent_spec import AgentSpec
from agent_system.runtime.service.yaml_loader import AgentDefinition, create_agent_from_definition


class ResearchSessions:
    def __init__(self, catalog, roots, *, capacity=4, ttl=14400,
                 factory=create_agent_from_definition, version=None):
        self.catalog = Path(catalog)
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.capacity = capacity
        self.ttl = ttl
        self.factory = factory
        self.version = version or os.environ.get("QUORIDOR_RUNTIME_VERSION", "research-v1-unversioned")
        self.sessions = {}
        self.lock = threading.RLock()
        self.execution = threading.BoundedSemaphore(1)

    def _expire(self):
        now = time.monotonic()
        for identifier in list(self.sessions):
            if self.sessions[identifier]["expires"] < now:
                del self.sessions[identifier]

    def create(self, sha, room_id, seat):
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or seat not in (1, 2) or not room_id:
            raise ValueError("Full SHA, room and seat are required")
        with self.lock:
            self._expire()
            if len(self.sessions) >= self.capacity:
                raise RuntimeError("Research session capacity reached")
            rows = json.loads(self.catalog.read_text())["models"]
            matches = [row for row in rows if row["sha256"] == sha]
            if len(matches) != 1:
                raise ValueError("Checkpoint is not uniquely approved for research Play")
            row = matches[0]
            meta = row["metadata"]
            if (row.get("conflicts") or row.get("architecture") != "cnn"
                    or meta.get("observation_version") != "dqn_obs_cnn_v1"
                    or meta.get("action_space_version") != "dqn_action_v1"
                    or meta.get("action_count") != 209
                    or row.get("output_semantics") not in ("policy_logits", "q_values")):
                raise ValueError("Incompatible architecture, encoder, actions or output semantics")
            path = Path(row["checkpoint"]).resolve(strict=True)
            if not any(path.is_relative_to(root) for root in self.roots):
                raise ValueError("Checkpoint outside approved roots")
            if path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError("Checkpoint exceeds research size limit")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                    raise ValueError("Checkpoint SHA mismatch")
            spec = AgentSpec(id="research-" + sha, algo_type="dqn",
                algo_params={"checkpoint": str(path), "sha256": sha, "device": "cpu"},
                policy_type="top_k", policy_params={"k": 1},
                display_name=row.get("name", sha))
            agent = self.factory(AgentDefinition(spec))
            inference = agent.inference
            actual = inference.metadata
            semantics = actual.get("output_semantics")
            if semantics == "action_values":
                semantics = "q_values"
            if (inference.sha256 != sha or actual.get("observation_version") != "dqn_obs_cnn_v1"
                    or actual.get("action_space_version") != "dqn_action_v1"
                    or actual.get("action_count") != 209
                    or actual.get("model_config", {}).get("model_arch") != "cnn"
                    or semantics != row["output_semantics"]):
                raise ValueError("Loaded checkpoint metadata does not match approved identity")
            if sum(parameter.numel() for parameter in inference.network.parameters()) > 20_000_000:
                raise ValueError("Model exceeds research parameter limit")
            if actual.get("algorithm") != meta.get("model_config.algorithm"):
                raise ValueError("Loaded algorithm differs from approved catalog")
            for key, value in meta.items():
                if key.startswith("model_config.") and key != "model_config.algorithm":
                    if actual["model_config"].get(key.removeprefix("model_config.")) != value:
                        raise ValueError("Loaded architecture configuration differs from catalog: " + key)
            identifier, token = secrets.token_urlsafe(24), secrets.token_urlsafe(32)
            identity = dict(sha256=sha, actual_sha256=inference.sha256,
                name=row.get("name", sha), architecture="cnn",
                algorithm=actual["algorithm"],
                output_semantics=row["output_semantics"], agent_spec=asdict(spec),
                runtime_version=self.version, room_id=room_id, agent_seat=seat,
                actual_load_verified=True)
            self.sessions[identifier] = dict(agent=agent, token=token, identity=identity,
                expires=time.monotonic() + self.ttl)
            return dict(session_id=identifier, token=token, identity=identity, ttl_seconds=self.ttl)

    def _get(self, identifier, token):
        self._expire()
        session = self.sessions.get(identifier)
        if session is None or not secrets.compare_digest(session["token"], token):
            raise PermissionError("Unknown, expired or unauthorized research session")
        return session

    def action(self, identifier, token, room_id, seat, state, legal):
        if not self.execution.acquire(blocking=False):
            raise RuntimeError("Research inference busy; retry the same turn")
        try:
            with self.lock:
                session = self._get(identifier, token)
                identity = session["identity"]
                if identity["room_id"] != room_id or identity["agent_seat"] != seat:
                    raise PermissionError("Research session is bound to another room or seat")
                agent = session["agent"]
            action = agent.make_action(state, legal)
            if action not in legal:
                raise ValueError("Research agent returned an action outside engine legal actions")
            return dict(action=action, identity=identity)
        finally:
            self.execution.release()

    def destroy(self, identifier, token):
        with self.lock:
            self._get(identifier, token)
            del self.sessions[identifier]