"""Checksum-pinned DQN inference shared by Arena and online gameplay."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from quoridor_engine import RuleEngine

from agent_system.runtime.service.agents.greedy_agent import _build_state, _engine_action_to_wire
from agent_system.training.dqn.action_space import encode_engine_action
from agent_system.training.dqn.checkpoint import load_checkpoint
from agent_system.training.dqn.observation import encode_observation, OBSERVATION_VERSION
from agent_system.training.dqn.observation_cnn import encode_observation_cnn, CNN_OBSERVATION_VERSION


class DQNInference:
    def __init__(self, checkpoint: str, sha256: str, device: str = "cpu"):
        path = Path(checkpoint)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: expected {sha256}, got {actual}")
        loaded = load_checkpoint(path)
        if loaded.observation_version == CNN_OBSERVATION_VERSION:
            self.encoder = encode_observation_cnn
        elif loaded.observation_version == OBSERVATION_VERSION:
            self.encoder = encode_observation
        else:
            raise ValueError(f"Unsupported observation version: {loaded.observation_version}")
        self.network = loaded.network.to(device).eval()
        self.device = torch.device(device)
        self.engine = RuleEngine.standard()
        self.sha256 = actual

    def score(self, game_state: dict) -> list[tuple[dict, float]]:
        raw = _build_state(game_state)
        legal = sorted(self.engine.legal_actions(raw), key=encode_engine_action)
        if not legal:
            raise ValueError("No legal actions")
        x = torch.tensor(self.encoder(raw), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            values = self.network(x).reshape(-1).cpu()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Non-finite Q values")
        return [
            (_engine_action_to_wire(a, game_state["current_player"]), float(values[encode_engine_action(a)]))
            for a in legal
        ]
