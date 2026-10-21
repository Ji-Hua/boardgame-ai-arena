"""Unit tests for agent_system/training/dqn/model.py — Phase 4 / Phase 18A.

Test groups
-----------
TestQNetworkConstruction        — default and custom construction
TestQNetworkForward             — single obs and batch forward pass
TestSelectGreedyAction          — greedy selection correctness
TestSelectEpsilonGreedyAction   — epsilon-greedy exploration
TestDQNPolicy                   — DQNPolicy.select_action integration
TestDQNPolicyRollout            — untrained DQN full-game rollout
TestCNNQNetwork                 — Phase 18A CNN model
TestBuildQNetwork               — Phase 18A factory function
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from agent_system.training.dqn.action_space import ACTION_COUNT
from agent_system.training.dqn.model import (
    CNN_DEFAULT_CHANNELS,
    CNN_MODEL_VERSION,
    DEFAULT_HIDDEN_SIZE,
    MODEL_VERSION,
    OBSERVATION_SIZE,
    CNNQNetwork,
    DQNPolicy,
    QNetwork,
    _apply_mask,
    _validate_hidden_layers,
    build_q_network,
    select_epsilon_greedy_action,
    select_greedy_action,
)
from agent_system.training.dqn.observation import OBSERVATION_SIZE as OBS_SIZE_FROM_OBS
from agent_system.training.dqn.observation_cnn import CNN_CHANNELS, CNN_BOARD_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_legal_mask() -> list[bool]:
    """All-True mask — every action is legal."""
    return [True] * ACTION_COUNT


def _single_legal_mask(action_id: int) -> list[bool]:
    """Mask where only one action_id is legal."""
    mask = [False] * ACTION_COUNT
    mask[action_id] = True
    return mask


def _q_values_zeros() -> torch.Tensor:
    return torch.zeros(ACTION_COUNT)


def _q_values_ranked() -> torch.Tensor:
    """Q-values where action 5 has the highest value."""
    q = torch.zeros(ACTION_COUNT)
    q[5] = 10.0
    return q


# ---------------------------------------------------------------------------
# TestQNetworkConstruction
# ---------------------------------------------------------------------------

class TestQNetworkConstruction:
    def test_default_construction_succeeds(self):
        net = QNetwork()
        assert net is not None

    def test_obs_size_attribute(self):
        net = QNetwork()
        assert net.obs_size == OBSERVATION_SIZE

    def test_action_count_attribute(self):
        net = QNetwork()
        assert net.action_count == ACTION_COUNT

    def test_hidden_size_default(self):
        net = QNetwork()
        assert net.hidden_size == DEFAULT_HIDDEN_SIZE

    def test_custom_hidden_size(self):
        net = QNetwork(hidden_size=128)
        assert net.hidden_size == 128

    def test_observation_size_matches_observation_module(self):
        """OBSERVATION_SIZE imported into model must equal OBSERVATION_SIZE from observation.py."""
        assert OBSERVATION_SIZE == OBS_SIZE_FROM_OBS

    def test_model_version_string(self):
        assert MODEL_VERSION == "dqn_model_v1"


# ---------------------------------------------------------------------------
# TestQNetworkForward
# ---------------------------------------------------------------------------

class TestQNetworkForward:
    def test_forward_single_obs_output_shape(self):
        net = QNetwork()
        obs = torch.zeros(OBSERVATION_SIZE)
        q = net(obs)
        assert q.shape == (ACTION_COUNT,)

    def test_forward_single_obs_output_dtype_float32(self):
        net = QNetwork()
        obs = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
        q = net(obs)
        assert q.dtype == torch.float32

    def test_forward_batch_output_shape(self):
        net = QNetwork()
        batch_size = 8
        obs = torch.zeros(batch_size, OBSERVATION_SIZE)
        q = net(obs)
        assert q.shape == (batch_size, ACTION_COUNT)

    def test_forward_batch_size_1(self):
        net = QNetwork()
        obs = torch.zeros(1, OBSERVATION_SIZE)
        q = net(obs)
        assert q.shape == (1, ACTION_COUNT)

    def test_forward_returns_finite_values(self):
        net = QNetwork()
        obs = torch.zeros(OBSERVATION_SIZE)
        q = net(obs)
        assert torch.all(torch.isfinite(q))

    def test_forward_deterministic_for_same_input(self):
        net = QNetwork()
        obs = torch.ones(OBSERVATION_SIZE)
        q1 = net(obs)
        q2 = net(obs)
        assert torch.equal(q1, q2)

    def test_forward_output_count_is_209(self):
        """Explicit check that output dimension is exactly 209."""
        net = QNetwork()
        obs = torch.zeros(OBSERVATION_SIZE)
        q = net(obs)
        assert q.shape[0] == 209

    def test_forward_custom_hidden_size(self):
        net = QNetwork(hidden_size=64)
        obs = torch.zeros(OBSERVATION_SIZE)
        q = net(obs)
        assert q.shape == (ACTION_COUNT,)


# ---------------------------------------------------------------------------
# TestSelectGreedyAction
# ---------------------------------------------------------------------------

class TestSelectGreedyAction:
    def test_greedy_selects_highest_q_legal_action(self):
        q = _q_values_ranked()  # action 5 has highest Q
        mask = _all_legal_mask()
        assert select_greedy_action(q, mask) == 5

    def test_greedy_ignores_illegal_high_q(self):
        q = torch.zeros(ACTION_COUNT)
        q[5] = 100.0  # illegal
        mask = _all_legal_mask()
        mask[5] = False  # make action 5 illegal
        # With action 5 illegal, result must NOT be 5
        result = select_greedy_action(q, mask)
        assert result != 5

    def test_greedy_single_legal_action_returns_that_action(self):
        q = torch.zeros(ACTION_COUNT)
        legal_id = 42
        mask = _single_legal_mask(legal_id)
        assert select_greedy_action(q, mask) == legal_id

    def test_greedy_all_zero_q_returns_first_legal(self):
        q = torch.zeros(ACTION_COUNT)
        mask = _single_legal_mask(0)
        assert select_greedy_action(q, mask) == 0

    def test_greedy_raises_for_wrong_mask_length(self):
        q = _q_values_zeros()
        bad_mask = [True] * 10
        with pytest.raises(ValueError, match="Mask length"):
            select_greedy_action(q, bad_mask)

    def test_greedy_raises_for_all_false_mask(self):
        q = _q_values_zeros()
        mask = [False] * ACTION_COUNT
        with pytest.raises(ValueError):
            select_greedy_action(q, mask)

    def test_greedy_returns_int(self):
        q = _q_values_ranked()
        mask = _all_legal_mask()
        result = select_greedy_action(q, mask)
        assert isinstance(result, int)

    def test_greedy_result_is_valid_action_id(self):
        q = _q_values_ranked()
        mask = _all_legal_mask()
        result = select_greedy_action(q, mask)
        assert 0 <= result < ACTION_COUNT


# ---------------------------------------------------------------------------
# TestSelectEpsilonGreedyAction
# ---------------------------------------------------------------------------

class TestSelectEpsilonGreedyAction:
    def test_epsilon_zero_is_greedy(self):
        q = _q_values_ranked()  # action 5 has highest Q
        mask = _all_legal_mask()
        result = select_epsilon_greedy_action(q, mask, epsilon=0.0)
        assert result == 5

    def test_epsilon_one_samples_legal_only(self):
        rng = random.Random(42)
        q = _q_values_zeros()
        # Only actions 10–20 legal
        mask = [False] * ACTION_COUNT
        for i in range(10, 21):
            mask[i] = True
        for _ in range(50):
            result = select_epsilon_greedy_action(q, mask, epsilon=1.0, rng=rng)
            assert 10 <= result <= 20

    def test_epsilon_one_never_selects_illegal(self):
        rng = random.Random(0)
        q = torch.zeros(ACTION_COUNT)
        q[3] = 999.0  # action 3 has huge Q but is illegal
        mask = _all_legal_mask()
        mask[3] = False
        for _ in range(100):
            result = select_epsilon_greedy_action(q, mask, epsilon=1.0, rng=rng)
            assert result != 3

    def test_epsilon_zero_deterministic(self):
        q = _q_values_ranked()
        mask = _all_legal_mask()
        results = {select_epsilon_greedy_action(q, mask, epsilon=0.0) for _ in range(10)}
        assert len(results) == 1

    def test_epsilon_greedy_raises_for_wrong_mask_length(self):
        q = _q_values_zeros()
        bad_mask = [True] * 5
        with pytest.raises(ValueError, match="Mask length"):
            select_epsilon_greedy_action(q, bad_mask, epsilon=0.5)

    def test_epsilon_greedy_raises_for_all_false_mask(self):
        q = _q_values_zeros()
        mask = [False] * ACTION_COUNT
        with pytest.raises(ValueError):
            select_epsilon_greedy_action(q, mask, epsilon=0.0)

    def test_epsilon_greedy_returns_int(self):
        result = select_epsilon_greedy_action(_q_values_zeros(), _all_legal_mask(), 0.0)
        assert isinstance(result, int)

    def test_reproducible_with_rng(self):
        q = _q_values_zeros()
        mask = _all_legal_mask()
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        results1 = [select_epsilon_greedy_action(q, mask, 0.5, rng1) for _ in range(20)]
        results2 = [select_epsilon_greedy_action(q, mask, 0.5, rng2) for _ in range(20)]
        assert results1 == results2


# ---------------------------------------------------------------------------
# TestDQNPolicy
# ---------------------------------------------------------------------------

class TestDQNPolicy:
    def _make_policy(self) -> DQNPolicy:
        return DQNPolicy(QNetwork())

    def test_construction_succeeds(self):
        policy = self._make_policy()
        assert policy is not None

    def test_network_attribute(self):
        net = QNetwork()
        policy = DQNPolicy(net)
        assert policy.network is net

    def test_device_is_cpu_by_default(self):
        policy = self._make_policy()
        assert policy.device == torch.device("cpu")

    def test_select_action_returns_int(self):
        policy = self._make_policy()
        obs = [0.0] * OBSERVATION_SIZE
        mask = _all_legal_mask()
        result = policy.select_action(obs, mask)
        assert isinstance(result, int)

    def test_select_action_returns_valid_id(self):
        policy = self._make_policy()
        obs = [0.0] * OBSERVATION_SIZE
        mask = _all_legal_mask()
        result = policy.select_action(obs, mask)
        assert 0 <= result < ACTION_COUNT

    def test_select_action_respects_mask(self):
        policy = self._make_policy()
        obs = [0.0] * OBSERVATION_SIZE
        legal_id = 77
        mask = _single_legal_mask(legal_id)
        result = policy.select_action(obs, mask, epsilon=0.0)
        assert result == legal_id

    def test_select_action_epsilon_one_samples_legal(self):
        policy = self._make_policy()
        obs = [0.0] * OBSERVATION_SIZE
        mask = [False] * ACTION_COUNT
        for i in range(50, 60):
            mask[i] = True
        rng = random.Random(1)
        for _ in range(30):
            result = policy.select_action(obs, mask, epsilon=1.0, rng=rng)
            assert 50 <= result < 60

    def test_select_action_raises_for_bad_mask_length(self):
        policy = self._make_policy()
        obs = [0.0] * OBSERVATION_SIZE
        mask = [True] * 10
        with pytest.raises(ValueError, match="Mask length"):
            policy.select_action(obs, mask)

    def test_select_action_raises_for_all_false_mask(self):
        policy = self._make_policy()
        obs = [0.0] * OBSERVATION_SIZE
        mask = [False] * ACTION_COUNT
        with pytest.raises(ValueError):
            policy.select_action(obs, mask)

    def test_network_stays_in_eval_mode(self):
        policy = self._make_policy()
        assert not policy.network.training


# ---------------------------------------------------------------------------
# TestDQNPolicyRollout — untrained DQN full-game rollout
# ---------------------------------------------------------------------------

class TestDQNPolicyRollout:
    """Prove: encoded observation -> Q-network -> legal masked action -> engine step.

    An untrained DQN in pure greedy mode (epsilon=0.0) can cycle infinitely because
    identical observations always produce identical Q-values and therefore the same
    action is repeated. Rollout termination tests therefore use epsilon=1.0 (pure
    random from legal mask), which still exercises the full pipeline:
        list[float] obs  ->  QNetwork.forward()  ->  masked epsilon-greedy  ->  env.step()
    Greedy selection correctness is tested exhaustively in TestSelectGreedyAction
    and TestDQNPolicy; those tests do not require game termination.
    """

    def _run_rollout(self, epsilon: float = 1.0, seed: int = 0) -> dict:
        """Run one full game, returning stats dict."""
        from quoridor_engine import RuleEngine
        from agent_system.training.dqn.env import QuoridorEnv

        engine = RuleEngine.standard()
        env = QuoridorEnv(engine)
        policy = DQNPolicy(QNetwork())
        rng = random.Random(seed)

        obs = env.reset()
        done = False
        step_count = 0
        illegal_selections = 0

        while not done:
            mask = env.legal_action_mask()
            action_id = policy.select_action(obs, mask, epsilon=epsilon, rng=rng)

            # Verify selected action is in the legal mask
            if not mask[action_id]:
                illegal_selections += 1

            obs, reward, done, info = env.step(action_id)
            step_count += 1

            # Safety: prevent infinite loops on broken engine / mask
            assert step_count < 5000, "Rollout exceeded 5000 steps"

        return {
            "step_count": step_count,
            "illegal_selections": illegal_selections,
            "reward": reward,
            "done": done,
        }

    def test_rollout_completes(self):
        stats = self._run_rollout(epsilon=1.0, seed=0)
        assert stats["done"] is True

    def test_rollout_zero_illegal_actions_random_exploration(self):
        """epsilon=1.0 still uses the mask; every selected action must be legal."""
        stats = self._run_rollout(epsilon=1.0, seed=0)
        assert stats["illegal_selections"] == 0

    def test_rollout_zero_illegal_actions_epsilon_greedy(self):
        stats = self._run_rollout(epsilon=0.5, seed=42)
        assert stats["illegal_selections"] == 0

    def test_rollout_terminal_reward_nonzero(self):
        """Terminal reward is ±1 when a learner_player is configured."""
        from quoridor_engine import Player, RuleEngine
        from agent_system.training.dqn.env import QuoridorEnv

        engine = RuleEngine.standard()
        env = QuoridorEnv(engine, learner_player=Player.P1)
        policy = DQNPolicy(QNetwork())
        rng = random.Random(1)

        obs = env.reset()
        done = False
        reward = 0.0
        step_count = 0
        while not done:
            action_id = policy.select_action(obs, env.legal_action_mask(), epsilon=1.0, rng=rng)
            obs, reward, done, _ = env.step(action_id)
            step_count += 1
            assert step_count < 5000

        assert reward in (1.0, -1.0)


# ---------------------------------------------------------------------------
# TestQNetworkHiddenLayers (Phase 16A)
# ---------------------------------------------------------------------------

class TestQNetworkHiddenLayers:
    """Tests for variable-depth MLP architecture introduced in Phase 16A."""

    # --- Construction ---

    def test_default_hidden_layers_is_two_256(self):
        net = QNetwork()
        assert net.hidden_layers == [256, 256]

    def test_hidden_size_compat_produces_correct_layers(self):
        """Passing hidden_size=512 without hidden_layers gives [512, 512]."""
        net = QNetwork(hidden_size=512)
        assert net.hidden_layers == [512, 512]
        assert net.hidden_size == 512

    def test_hidden_layers_two_layers(self):
        net = QNetwork(hidden_layers=[512, 512])
        assert net.hidden_layers == [512, 512]
        assert net.hidden_size == 512

    def test_hidden_layers_three_layers(self):
        net = QNetwork(hidden_layers=[512, 512, 256])
        assert net.hidden_layers == [512, 512, 256]
        assert net.hidden_size == 512

    def test_hidden_layers_one_layer(self):
        net = QNetwork(hidden_layers=[128])
        assert net.hidden_layers == [128]
        assert net.hidden_size == 128

    def test_hidden_layers_four_layers(self):
        net = QNetwork(hidden_layers=[256, 256, 128, 64])
        assert net.hidden_layers == [256, 256, 128, 64]

    def test_hidden_layers_overrides_hidden_size(self):
        """When hidden_layers is given, hidden_size arg is ignored."""
        net = QNetwork(hidden_size=128, hidden_layers=[512, 256])
        assert net.hidden_layers == [512, 256]
        assert net.hidden_size == 512

    def test_hidden_layers_stored_as_list_not_reference(self):
        """Mutation of the original list does not affect net.hidden_layers."""
        layers = [256, 256]
        net = QNetwork(hidden_layers=layers)
        layers.append(128)
        assert net.hidden_layers == [256, 256]

    # --- Validation ---

    def test_empty_hidden_layers_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_hidden_layers([])

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            _validate_hidden_layers([256, 0])

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            _validate_hidden_layers([-1])

    def test_float_width_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            _validate_hidden_layers([256.0])  # type: ignore[list-item]

    def test_qnetwork_empty_hidden_layers_raises(self):
        with pytest.raises(ValueError):
            QNetwork(hidden_layers=[])

    # --- Forward shape ---

    def test_forward_shape_two_layers_512(self):
        net = QNetwork(hidden_layers=[512, 512])
        obs = torch.zeros(OBSERVATION_SIZE)
        out = net(obs)
        assert out.shape == (ACTION_COUNT,)

    def test_forward_shape_three_layers(self):
        net = QNetwork(hidden_layers=[512, 512, 256])
        obs = torch.zeros(OBSERVATION_SIZE)
        out = net(obs)
        assert out.shape == (ACTION_COUNT,)

    def test_forward_batch_shape_three_layers(self):
        net = QNetwork(hidden_layers=[512, 512, 256])
        obs = torch.zeros(8, OBSERVATION_SIZE)
        out = net(obs)
        assert out.shape == (8, ACTION_COUNT)

    # --- Parameter counts ---

    def test_parameter_count_default_256x2(self):
        net = QNetwork()
        # obs=292, h0=256, h1=256, out=209
        expected = (292 * 256 + 256) + (256 * 256 + 256) + (256 * 209 + 209)
        assert net.parameter_count() == expected

    def test_parameter_count_512x2(self):
        net = QNetwork(hidden_layers=[512, 512])
        expected = (292 * 512 + 512) + (512 * 512 + 512) + (512 * 209 + 209)
        assert net.parameter_count() == expected

    def test_parameter_count_512x512x256(self):
        net = QNetwork(hidden_layers=[512, 512, 256])
        expected = (292 * 512 + 512) + (512 * 512 + 512) + (512 * 256 + 256) + (256 * 209 + 209)
        assert net.parameter_count() == expected

    def test_parameter_count_larger_than_default(self):
        small = QNetwork(hidden_layers=[256, 256])
        big = QNetwork(hidden_layers=[512, 512])
        assert big.parameter_count() > small.parameter_count()

    # --- Backward compat: default == old architecture ---

    def test_default_equals_hidden_size_256(self):
        """QNetwork() must produce exactly the same architecture as old QNetwork(hidden_size=256)."""
        net_default = QNetwork()
        net_compat = QNetwork(hidden_size=256)
        assert net_default.hidden_layers == net_compat.hidden_layers
        assert net_default.parameter_count() == net_compat.parameter_count()


# ---------------------------------------------------------------------------
# TestCNNQNetwork  (Phase 18A)
# ---------------------------------------------------------------------------

class TestCNNQNetwork:
    """Tests for the CNNQNetwork introduced in Phase 18A."""

    # --- Constants ---

    def test_cnn_model_version_string(self):
        assert CNN_MODEL_VERSION == "cnn_model_v1"

    def test_default_channels_constant(self):
        assert CNN_DEFAULT_CHANNELS == [32, 64, 64]

    # --- Construction ---

    def test_default_construction(self):
        net = CNNQNetwork()
        assert net is not None

    def test_custom_in_channels(self):
        net = CNNQNetwork(in_channels=CNN_CHANNELS)
        assert net is not None

    def test_custom_cnn_channels(self):
        net = CNNQNetwork(in_channels=7, cnn_channels=[16, 32])
        assert net is not None

    def test_custom_action_count(self):
        net = CNNQNetwork(action_count=ACTION_COUNT)
        assert net is not None

    def test_custom_dense_width(self):
        net = CNNQNetwork(dense_width=128)
        assert net is not None

    # --- Forward: single observation ---

    def test_forward_single_obs_shape(self):
        """Single [7, 9, 9] observation -> [ACTION_COUNT]."""
        net = CNNQNetwork()
        obs = torch.zeros(CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert out.shape == torch.Size([ACTION_COUNT])

    def test_forward_single_obs_no_extra_batch_dim(self):
        """The forward pass must NOT return [1, ACTION_COUNT] for single inputs."""
        net = CNNQNetwork()
        obs = torch.zeros(CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert out.dim() == 1

    # --- Forward: batch ---

    def test_forward_batch_shape(self):
        """Batch [B, 7, 9, 9] -> [B, ACTION_COUNT]."""
        net = CNNQNetwork()
        obs = torch.zeros(8, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert out.shape == torch.Size([8, ACTION_COUNT])

    def test_forward_batch_dim_preserved(self):
        for batch_size in (1, 4, 16):
            net = CNNQNetwork()
            obs = torch.zeros(batch_size, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
            out = net(obs)
            assert out.shape[0] == batch_size

    # --- Backward ---

    def test_backward_pass_runs(self):
        """Loss.backward() must not raise for a CNN network."""
        net = CNNQNetwork()
        obs = torch.zeros(4, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        loss = out.sum()
        loss.backward()  # should not raise

    def test_gradients_are_not_none_after_backward(self):
        net = CNNQNetwork()
        obs = torch.zeros(4, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        out.sum().backward()
        for name, p in net.named_parameters():
            assert p.grad is not None, f"Gradient is None for param: {name}"

    # --- Parameter count ---

    def test_parameter_count_is_positive(self):
        net = CNNQNetwork()
        assert net.parameter_count() > 0

    def test_parameter_count_matches_manual(self):
        """Verify parameter_count() against torch's own sum."""
        net = CNNQNetwork()
        expected = sum(p.numel() for p in net.parameters())
        assert net.parameter_count() == expected

    def test_parameter_count_smaller_channels_is_less(self):
        big = CNNQNetwork(cnn_channels=[32, 64, 64])
        small = CNNQNetwork(cnn_channels=[8, 16])
        assert small.parameter_count() < big.parameter_count()

    # --- Output values ---

    def test_output_finite_random_input(self):
        net = CNNQNetwork()
        obs = torch.randn(4, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert torch.all(torch.isfinite(out))

    def test_random_input_produces_non_constant_output(self):
        """With random weights and random input, output should not be all equal."""
        net = CNNQNetwork()
        obs = torch.randn(1, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        # Very unlikely to have all equal Q-values with random weights
        assert not torch.all(out == out[0][0])

    # --- Eval mode ---

    def test_eval_mode_runs(self):
        net = CNNQNetwork()
        net.eval()
        with torch.no_grad():
            obs = torch.zeros(CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
            out = net(obs)
        assert out.shape == torch.Size([ACTION_COUNT])

    def test_train_mode_runs(self):
        net = CNNQNetwork()
        net.train()
        obs = torch.zeros(1, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert out.shape == torch.Size([1, ACTION_COUNT])

    # --- is_cnn_network utility ---

    def test_cnnqnetwork_is_not_qnetwork(self):
        cnn = CNNQNetwork()
        assert not isinstance(cnn, QNetwork)


# ---------------------------------------------------------------------------
# TestBuildQNetwork  (Phase 18A)
# ---------------------------------------------------------------------------

class TestBuildQNetwork:
    """Tests for the build_q_network factory function."""

    # --- MLP path ---

    def test_build_mlp_returns_qnetwork(self):
        net = build_q_network("mlp", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                               hidden_layers=[256, 256], in_channels=7,
                               cnn_channels=[32, 64, 64], dense_width=256)
        assert isinstance(net, QNetwork)

    def test_build_mlp_default_hidden_layers(self):
        net = build_q_network("mlp", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                               hidden_layers=[256, 256], in_channels=7,
                               cnn_channels=[32, 64, 64], dense_width=256)
        assert net.hidden_layers == [256, 256]

    def test_build_mlp_custom_hidden_layers(self):
        net = build_q_network("mlp", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                               hidden_layers=[512, 512, 256], in_channels=7,
                               cnn_channels=[32, 64, 64], dense_width=256)
        assert net.hidden_layers == [512, 512, 256]

    def test_build_mlp_forward_shape(self):
        net = build_q_network("mlp", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                               hidden_layers=[256, 256], in_channels=7,
                               cnn_channels=[32, 64, 64], dense_width=256)
        obs = torch.zeros(OBSERVATION_SIZE)
        out = net(obs)
        assert out.shape == torch.Size([ACTION_COUNT])

    # --- CNN path ---

    def test_build_cnn_returns_cnnqnetwork(self):
        net = build_q_network("cnn", ACTION_COUNT, obs_size=567,
                               hidden_layers=[256, 256], in_channels=CNN_CHANNELS,
                               cnn_channels=[32, 64, 64], dense_width=256)
        assert isinstance(net, CNNQNetwork)

    def test_build_cnn_forward_single_shape(self):
        net = build_q_network("cnn", ACTION_COUNT, obs_size=567,
                               hidden_layers=[256, 256], in_channels=CNN_CHANNELS,
                               cnn_channels=[32, 64, 64], dense_width=256)
        obs = torch.zeros(CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert out.shape == torch.Size([ACTION_COUNT])

    def test_build_cnn_forward_batch_shape(self):
        net = build_q_network("cnn", ACTION_COUNT, obs_size=567,
                               hidden_layers=[256, 256], in_channels=CNN_CHANNELS,
                               cnn_channels=[32, 64, 64], dense_width=256)
        obs = torch.zeros(4, CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
        out = net(obs)
        assert out.shape == torch.Size([4, ACTION_COUNT])

    def test_build_cnn_custom_channels(self):
        net = build_q_network("cnn", ACTION_COUNT, obs_size=567,
                               hidden_layers=[256, 256], in_channels=CNN_CHANNELS,
                               cnn_channels=[16, 32], dense_width=256)
        assert isinstance(net, CNNQNetwork)

    # --- Invalid arch ---

    def test_invalid_arch_raises_value_error(self):
        with pytest.raises(ValueError, match="model_arch"):
            build_q_network("transformer", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                             hidden_layers=[256, 256], in_channels=7,
                             cnn_channels=[32, 64, 64], dense_width=256)

    def test_empty_arch_raises_value_error(self):
        with pytest.raises(ValueError):
            build_q_network("", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                             hidden_layers=[256, 256], in_channels=7,
                             cnn_channels=[32, 64, 64], dense_width=256)

    # --- MLP backward compat: existing tests still pass with build_q_network ---

    def test_mlp_parameter_count_matches_direct_construction(self):
        net_factory = build_q_network("mlp", ACTION_COUNT, obs_size=OBSERVATION_SIZE,
                                       hidden_layers=[256, 256], in_channels=7,
                                       cnn_channels=[32, 64, 64], dense_width=256)
        net_direct = QNetwork(hidden_layers=[256, 256])
        assert net_factory.parameter_count() == net_direct.parameter_count()

