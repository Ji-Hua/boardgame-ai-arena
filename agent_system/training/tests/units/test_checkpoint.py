"""Unit tests for agent_system/training/dqn/checkpoint.py.

Test areas:
    - save_checkpoint creates a readable file
    - saved payload contains all required metadata keys
    - loaded DQNCheckpoint has correct attribute types
    - loaded QNetwork produces Q-values of correct shape
    - auto-generated checkpoint_id is a non-empty string
    - explicit checkpoint_id is preserved
    - training_step / episode_count round-trip correctly
    - model weights round-trip (same output on identical input)
    - optimizer state dict is saved when optimizer is provided
    - optimizer state dict is empty dict when optimizer is not provided
    - eval_summary is saved and loaded
    - eval_summary defaults to empty dict when not provided
    - incompatible observation_version raises ValueError
    - incompatible observation_size raises ValueError
    - incompatible action_count raises ValueError
    - incompatible model_config.obs_size raises ValueError
    - incompatible model_config.action_count raises ValueError
    - missing required keys raises ValueError
    - FileNotFoundError for non-existent path
    - parent directories are created automatically
    - DQNCheckpoint.hidden_size returns correct value
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.optim as optim

from agent_system.training.dqn.action_space import ACTION_COUNT, ACTION_SPACE_VERSION
from agent_system.training.dqn.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    DQNCheckpoint,
    _REQUIRED_KEYS,
    load_checkpoint,
    save_checkpoint,
)
from agent_system.training.dqn.model import DEFAULT_HIDDEN_SIZE, QNetwork
from agent_system.training.dqn.observation import OBSERVATION_SIZE, OBSERVATION_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def network():
    return QNetwork()


@pytest.fixture()
def optimizer(network):
    return optim.Adam(network.parameters(), lr=1e-3)


@pytest.fixture()
def tmp_ckpt(network, tmp_path):
    """Save a valid checkpoint and return its path."""
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, network)
    return path


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestSaveCheckpointFileCreation:
    def test_creates_file(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        assert path.exists()

    def test_returns_path_object(self, network, tmp_path):
        path = tmp_path / "test.pt"
        result = save_checkpoint(path, network)
        assert isinstance(result, Path)
        assert result == path

    def test_creates_parent_directories(self, network, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "ckpt.pt"
        save_checkpoint(path, network)
        assert path.exists()

    def test_accepts_string_path(self, network, tmp_path):
        path = str(tmp_path / "test.pt")
        save_checkpoint(path, network)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Payload keys
# ---------------------------------------------------------------------------

class TestSaveCheckpointPayloadKeys:
    def test_all_required_keys_present(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        for key in _REQUIRED_KEYS:
            assert key in raw, f"Missing required key: {key}"

    def test_checkpoint_format_version_key_present(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION

    def test_observation_version_matches_current(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == OBSERVATION_VERSION

    def test_action_space_version_matches_current(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["action_space_version"] == ACTION_SPACE_VERSION

    def test_observation_size_matches_current(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_size"] == OBSERVATION_SIZE

    def test_action_count_matches_current(self, network, tmp_path):
        path = tmp_path / "test.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["action_count"] == ACTION_COUNT


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------

class TestSaveCheckpointMetadata:
    def test_auto_checkpoint_id_is_nonempty_string(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert isinstance(ckpt.checkpoint_id, str)
        assert len(ckpt.checkpoint_id) > 0

    def test_explicit_checkpoint_id_preserved(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, checkpoint_id="my-id-123")
        ckpt = load_checkpoint(path)
        assert ckpt.checkpoint_id == "my-id-123"

    def test_agent_id_preserved(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, agent_id="my_dqn_agent")
        ckpt = load_checkpoint(path)
        assert ckpt.agent_id == "my_dqn_agent"

    def test_training_step_preserved(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, training_step=12345)
        ckpt = load_checkpoint(path)
        assert ckpt.training_step == 12345

    def test_episode_count_preserved(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, episode_count=99)
        ckpt = load_checkpoint(path)
        assert ckpt.episode_count == 99

    def test_created_at_is_nonempty_string(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert isinstance(ckpt.created_at, str)
        assert len(ckpt.created_at) > 0

    def test_model_config_has_obs_size(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert ckpt.model_config["obs_size"] == OBSERVATION_SIZE

    def test_model_config_has_action_count(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert ckpt.model_config["action_count"] == ACTION_COUNT

    def test_model_config_has_hidden_size(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert ckpt.model_config["hidden_size"] == DEFAULT_HIDDEN_SIZE

    def test_hidden_size_property(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert ckpt.hidden_size == DEFAULT_HIDDEN_SIZE

    def test_eval_summary_default_empty(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert ckpt.eval_summary == {}

    def test_eval_summary_preserved(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        summary = {"win_rate": 0.75, "games": 20}
        save_checkpoint(path, network, eval_summary=summary)
        ckpt = load_checkpoint(path)
        assert ckpt.eval_summary == summary

    def test_optimizer_state_empty_when_not_provided(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        assert ckpt.optimizer_state_dict == {}

    def test_optimizer_state_saved_when_provided(self, network, optimizer, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, optimizer=optimizer)
        ckpt = load_checkpoint(path)
        # Should be a non-empty dict with optimizer state keys
        assert isinstance(ckpt.optimizer_state_dict, dict)
        assert len(ckpt.optimizer_state_dict) > 0


# ---------------------------------------------------------------------------
# Model weight round-trip
# ---------------------------------------------------------------------------

class TestModelWeightRoundTrip:
    def test_loaded_network_q_values_shape(self, tmp_ckpt):
        ckpt = load_checkpoint(tmp_ckpt)
        obs = torch.zeros(OBSERVATION_SIZE)
        with torch.no_grad():
            q = ckpt.network(obs)
        assert q.shape == (ACTION_COUNT,)

    def test_loaded_network_same_output_as_original(self, network, tmp_path):
        obs = torch.randn(OBSERVATION_SIZE)
        with torch.no_grad():
            original_q = network(obs).clone()

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network)
        ckpt = load_checkpoint(path)

        with torch.no_grad():
            loaded_q = ckpt.network(obs)

        assert torch.allclose(original_q, loaded_q)


# ---------------------------------------------------------------------------
# Compatibility validation errors
# ---------------------------------------------------------------------------

class TestCompatibilityValidation:
    def _save_and_patch(self, network, tmp_path, patch_fn):
        """Save a valid checkpoint, load raw, apply patch, re-save."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        patch_fn(raw)
        bad_path = tmp_path / "bad_ckpt.pt"
        torch.save(raw, bad_path)
        return bad_path

    def test_wrong_observation_version_raises(self, network, tmp_path):
        def patch(raw):
            raw["observation_version"] = "dqn_obs_STALE"
        bad = self._save_and_patch(network, tmp_path, patch)
        with pytest.raises(ValueError, match="observation_version"):
            load_checkpoint(bad)

    def test_wrong_observation_size_raises(self, network, tmp_path):
        def patch(raw):
            raw["observation_size"] = 9999
        bad = self._save_and_patch(network, tmp_path, patch)
        with pytest.raises(ValueError, match="observation_size"):
            load_checkpoint(bad)

    def test_wrong_action_count_raises(self, network, tmp_path):
        def patch(raw):
            raw["action_count"] = 100
        bad = self._save_and_patch(network, tmp_path, patch)
        with pytest.raises(ValueError, match="action_count"):
            load_checkpoint(bad)

    def test_wrong_model_config_obs_size_raises(self, network, tmp_path):
        def patch(raw):
            raw["model_config"]["obs_size"] = 9999
        bad = self._save_and_patch(network, tmp_path, patch)
        with pytest.raises(ValueError, match="model_config.obs_size"):
            load_checkpoint(bad)

    def test_wrong_model_config_action_count_raises(self, network, tmp_path):
        def patch(raw):
            raw["model_config"]["action_count"] = 100
        bad = self._save_and_patch(network, tmp_path, patch)
        with pytest.raises(ValueError, match="model_config.action_count"):
            load_checkpoint(bad)

    def test_missing_required_key_raises(self, network, tmp_path):
        def patch(raw):
            del raw["training_step"]
        bad = self._save_and_patch(network, tmp_path, patch)
        with pytest.raises(ValueError, match="missing required keys"):
            load_checkpoint(bad)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_expected_obs_version_mismatch_raises(self, network, tmp_path):
        """load_checkpoint with wrong expected_obs_version raises ValueError."""
        path = tmp_path / "ckpt_v1.pt"
        save_checkpoint(path, network, obs_version=OBSERVATION_VERSION)
        with pytest.raises(ValueError, match="observation_version"):
            load_checkpoint(path, expected_obs_version="dqn_obs_v2")

    def test_expected_obs_version_match_succeeds(self, network, tmp_path):
        """load_checkpoint with matching expected_obs_version succeeds."""
        path = tmp_path / "ckpt_v1.pt"
        save_checkpoint(path, network, obs_version=OBSERVATION_VERSION)
        ckpt = load_checkpoint(path, expected_obs_version=OBSERVATION_VERSION)
        assert ckpt.observation_version == OBSERVATION_VERSION

    def test_explicit_obs_version_saved_in_payload(self, network, tmp_path):
        """obs_version parameter is reflected in checkpoint metadata."""
        path = tmp_path / "ckpt_v2.pt"
        save_checkpoint(path, network, obs_version="dqn_obs_v2")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == "dqn_obs_v2"

    def test_v2_checkpoint_rejected_by_v1_default_load(self, network, tmp_path):
        """A v2 checkpoint must not silently load when default (v1) expected."""
        path = tmp_path / "ckpt_v2.pt"
        save_checkpoint(path, network, obs_version="dqn_obs_v2")
        with pytest.raises(ValueError, match="observation_version"):
            load_checkpoint(path)  # default expected = v1


# ---------------------------------------------------------------------------
# obs_version parameter on save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpointObsVersion:
    def test_default_obs_version_equals_module_constant(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == OBSERVATION_VERSION

    def test_explicit_v1_obs_version(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, obs_version="dqn_obs_v1")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == "dqn_obs_v1"

    def test_explicit_v2_obs_version(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, obs_version="dqn_obs_v2")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == "dqn_obs_v2"

    def test_none_obs_version_falls_back_to_default(self, network, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, network, obs_version=None)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == OBSERVATION_VERSION
