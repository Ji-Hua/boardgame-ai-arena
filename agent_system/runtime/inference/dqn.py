"""Checksum-pinned DQN inference shared by Arena and online gameplay."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from quoridor_engine import RuleEngine

from agent_system.runtime.service.agents.greedy_agent import _build_state, _engine_action_to_wire
from agent_system.training.dqn.action_space import encode_engine_action
from agent_system.training.dqn.checkpoint import load_checkpoint
from agent_system.training.dqn.model import network_outputs, route_mask
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
        self.metadata = {key: getattr(loaded, key) for key in (
            "checkpoint_id", "agent_id", "training_step", "episode_count", "model_config",
            "observation_version", "observation_size", "action_space_version", "action_count", "created_at", "eval_summary",
        )}
        self.metadata["output_semantics"] = loaded.eval_summary.get("output_semantics",
            "policy_logits" if loaded.algorithm in {"policy_distillation", "ppo", "ppo_imitation", "selfplay_ppo", "alphazero_lite", "value_calibration"}
            else "action_values")
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
            values, type_q = network_outputs(self.network, x)
            values = values.reshape(-1)
            if type_q is not None:
                if not bool(torch.isfinite(type_q).all()):
                    raise ValueError("Non-finite action-type Q values")
                mask = torch.zeros_like(values, dtype=torch.bool)
                mask[[encode_engine_action(a) for a in legal]] = True
                routed, _, _ = route_mask(type_q, mask)
                legal = [a for a in legal if bool(routed[encode_engine_action(a)])]
            values = values.cpu()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Non-finite Q values")
        return [
            (_engine_action_to_wire(a, game_state["current_player"]), float(values[encode_engine_action(a)]))
            for a in legal
        ]
