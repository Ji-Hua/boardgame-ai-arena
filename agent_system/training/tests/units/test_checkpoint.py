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
    - Phase 18A CNN checkpoint round-trip
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
from agent_system.training.dqn.model import DEFAULT_HIDDEN_SIZE, CNNQNetwork, QNetwork, CNN_DEFAULT_CHANNELS
from agent_system.training.dqn.observation import OBSERVATION_SIZE, OBSERVATION_VERSION
from agent_system.training.dqn.observation_cnn import (
    CNN_CHANNELS,
    CNN_OBSERVATION_SHAPE,
    CNN_OBSERVATION_SIZE,
    CNN_OBSERVATION_VERSION,
)


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


# ---------------------------------------------------------------------------
# TestHiddenLayersRoundTrip (Phase 16A)
# ---------------------------------------------------------------------------

class TestHiddenLayersRoundTrip:
    """Verify hidden_layers is saved and loaded correctly for all architectures,
    and that old checkpoints (hidden_size only) load via backward compat."""

    def test_default_network_saves_hidden_layers(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["model_config"]["hidden_layers"] == [256, 256]

    def test_512x2_saves_hidden_layers(self, tmp_path):
        net = QNetwork(hidden_layers=[512, 512])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["model_config"]["hidden_layers"] == [512, 512]

    def test_512x512x256_saves_hidden_layers(self, tmp_path):
        net = QNetwork(hidden_layers=[512, 512, 256])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["model_config"]["hidden_layers"] == [512, 512, 256]

    def test_load_checkpoint_hidden_layers_property_default(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.hidden_layers == [256, 256]

    def test_load_checkpoint_hidden_layers_property_512x2(self, tmp_path):
        net = QNetwork(hidden_layers=[512, 512])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.hidden_layers == [512, 512]

    def test_load_checkpoint_hidden_layers_property_three_layers(self, tmp_path):
        net = QNetwork(hidden_layers=[512, 512, 256])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.hidden_layers == [512, 512, 256]

    def test_load_checkpoint_network_architecture_512x2(self, tmp_path):
        """Loaded network must have same hidden_layers as saved network."""
        net = QNetwork(hidden_layers=[512, 512])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.network.hidden_layers == [512, 512]

    def test_load_checkpoint_network_architecture_three_layers(self, tmp_path):
        net = QNetwork(hidden_layers=[512, 512, 256])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.network.hidden_layers == [512, 512, 256]

    def test_backward_compat_hidden_size_only(self, tmp_path):
        """Old checkpoints without hidden_layers must load as [hidden_size, hidden_size]."""
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)

        # Simulate an old checkpoint by removing hidden_layers from model_config
        raw = torch.load(path, map_location="cpu", weights_only=False)
        del raw["model_config"]["hidden_layers"]
        torch.save(raw, path)

        ckpt = load_checkpoint(path)
        assert ckpt.hidden_layers == [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE]
        assert ckpt.network.hidden_layers == [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE]

    def test_backward_compat_weights_survive(self, tmp_path):
        """Weights loaded from an old-format checkpoint must match original model output."""
        net = QNetwork()
        obs = torch.zeros(OBSERVATION_SIZE)
        expected_q = net(obs).detach()

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)

        raw = torch.load(path, map_location="cpu", weights_only=False)
        del raw["model_config"]["hidden_layers"]
        torch.save(raw, path)

        ckpt = load_checkpoint(path)
        loaded_q = ckpt.network(obs).detach()
        assert torch.allclose(expected_q, loaded_q)

    def test_hidden_size_property_reflects_first_layer(self, tmp_path):
        net = QNetwork(hidden_layers=[512, 512, 256])
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.hidden_size == 512

    def test_default_checkpoint_hidden_size_is_256(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.hidden_size == DEFAULT_HIDDEN_SIZE


# ---------------------------------------------------------------------------
# TestPhase17AMetadata  (Phase 17A)
# ---------------------------------------------------------------------------

class TestPhase17AMetadata:
    """Verify algorithm, model_arch, param_count, and reward metadata round-trip."""

    def test_saves_algorithm_dqn(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net, algorithm="dqn")
        ckpt = load_checkpoint(path)
        assert ckpt.algorithm == "dqn"
        assert ckpt.is_double_dqn is False

    def test_saves_algorithm_double_dqn(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net, algorithm="double_dqn")
        ckpt = load_checkpoint(path)
        assert ckpt.algorithm == "double_dqn"
        assert ckpt.is_double_dqn is True

    def test_saves_model_arch(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net, model_arch="mlp")
        ckpt = load_checkpoint(path)
        assert ckpt.model_arch == "mlp"

    def test_saves_param_count(self, tmp_path):
        net = QNetwork()
        expected_count = net.parameter_count()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        ckpt = load_checkpoint(path)
        assert ckpt.param_count == expected_count

    def test_old_checkpoint_loads_with_algorithm_dqn_default(self, tmp_path):
        """Old checkpoints missing 'algorithm' key should default to 'dqn'."""
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net)
        # Remove algorithm from model_config to simulate old checkpoint
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw["model_config"].pop("algorithm", None)
        torch.save(raw, path)
        ckpt = load_checkpoint(path)
        assert ckpt.algorithm == "dqn"
        assert ckpt.is_double_dqn is False

    def test_saves_reward_metadata(self, tmp_path):
        net = QNetwork()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            path, net,
            reward_mode="distance_delta",
            distance_reward_weight=0.04,
            distance_delta_clip=1.0,
        )
        ckpt = load_checkpoint(path)
        assert ckpt.reward_mode == "distance_delta"
        assert abs(ckpt.distance_reward_weight - 0.04) < 1e-9
        assert abs(ckpt.distance_delta_clip - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# TestPhase18ACNNCheckpoint
# ---------------------------------------------------------------------------

class TestPhase18ACNNCheckpoint:
    """Verify CNN checkpoint save/load round-trip (Phase 18A)."""

    def test_save_cnn_arch_field(self, tmp_path):
        """Saving with model_arch='cnn' stores 'cnn' in model_config."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["model_config"]["model_arch"] == "cnn"

    def test_save_cnn_obs_version(self, tmp_path):
        """CNN checkpoint must store dqn_obs_cnn_v1 as obs_version."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_version"] == CNN_OBSERVATION_VERSION

    def test_save_cnn_observation_size(self, tmp_path):
        """CNN checkpoint must store CNN_OBSERVATION_SIZE (567)."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["observation_size"] == CNN_OBSERVATION_SIZE

    def test_save_cnn_stores_cnn_channels(self, tmp_path):
        """CNN checkpoint model_config must include cnn_channels."""
        net = CNNQNetwork(cnn_channels=[16, 32, 32])
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn", cnn_channels=[16, 32, 32])
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["model_config"]["cnn_channels"] == [16, 32, 32]

    def test_save_cnn_stores_in_channels(self, tmp_path):
        """CNN checkpoint model_config must include in_channels."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["model_config"]["in_channels"] == CNN_CHANNELS

    def test_load_cnn_returns_cnnqnetwork(self, tmp_path):
        """Loading a CNN checkpoint must return a CNNQNetwork instance."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        assert isinstance(ckpt.network, CNNQNetwork)

    def test_load_cnn_model_arch_attribute(self, tmp_path):
        """DQNCheckpoint.model_arch must equal 'cnn' after loading CNN ckpt."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        assert ckpt.model_arch == "cnn"

    def test_load_cnn_obs_version_attribute(self, tmp_path):
        """DQNCheckpoint.observation_version must be dqn_obs_cnn_v1."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        assert ckpt.observation_version == CNN_OBSERVATION_VERSION

    def test_load_cnn_weights_roundtrip(self, tmp_path):
        """Loaded CNN network must produce same output as the original."""
        net = CNNQNetwork()
        net.eval()
        obs = torch.randn(CNN_CHANNELS, 9, 9)
        with torch.no_grad():
            q_before = net(obs).clone()

        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        ckpt.network.eval()
        with torch.no_grad():
            q_after = ckpt.network(obs)

        assert torch.allclose(q_before, q_after, atol=1e-6)

    def test_load_cnn_forward_shape(self, tmp_path):
        """Loaded CNN network must produce [ACTION_COUNT] output for single obs."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        ckpt.network.eval()
        obs = torch.zeros(CNN_CHANNELS, 9, 9)
        with torch.no_grad():
            out = ckpt.network(obs)
        assert out.shape == torch.Size([ACTION_COUNT])

    def test_load_cnn_cnn_channels_property(self, tmp_path):
        """DQNCheckpoint.cnn_channels must round-trip correctly."""
        net = CNNQNetwork(cnn_channels=[16, 32])
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn", cnn_channels=[16, 32])
        ckpt = load_checkpoint(path)
        assert ckpt.cnn_channels == [16, 32]

    def test_cnn_checkpoint_not_confused_with_mlp(self, tmp_path):
        """Loading CNN checkpoint must NOT return a QNetwork (MLP)."""
        net = CNNQNetwork()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        assert not isinstance(ckpt.network, QNetwork)

    def test_mlp_checkpoint_not_confused_with_cnn(self, tmp_path):
        """Loading MLP checkpoint must NOT return a CNNQNetwork."""
        net = QNetwork()
        path = tmp_path / "mlp_ckpt.pt"
        save_checkpoint(path, net, model_arch="mlp")
        ckpt = load_checkpoint(path)
        assert not isinstance(ckpt.network, CNNQNetwork)
        assert isinstance(ckpt.network, QNetwork)

    def test_cnn_param_count_positive(self, tmp_path):
        """Loaded CNN checkpoint.param_count should be > 0."""
        net = CNNQNetwork()
        expected = net.parameter_count()
        path = tmp_path / "cnn_ckpt.pt"
        save_checkpoint(path, net, model_arch="cnn",
                         cnn_channels=list(CNN_DEFAULT_CHANNELS))
        ckpt = load_checkpoint(path)
        assert ckpt.param_count == expected


