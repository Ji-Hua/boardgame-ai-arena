"""Unit tests for agent_system/training/dqn/observation_cnn.py — Phase 18A.

Test groups
-----------
TestCNNObservationConstants     — version string, shape, size constants
TestEncodeObservationCNN        — output shape, dtypes, value ranges
TestCNNObservationSemantics     — current-player-centric encoding correctness
TestCNNObservationImmutability  — state not mutated by encoding
TestCNNObservationSensitivity   — encoding changes when state changes
TestCNNObservationV1Preserved   — original dqn_obs_v1 still works unchanged
"""

from __future__ import annotations

import math

import pytest
import torch

from agent_system.training.dqn.observation_cnn import (
    CNN_BOARD_SIZE,
    CNN_CHANNELS,
    CNN_OBSERVATION_SHAPE,
    CNN_OBSERVATION_SIZE,
    CNN_OBSERVATION_VERSION,
    encode_observation_cnn,
    observation_shape,
    observation_size,
)
from agent_system.training.dqn.observation import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    encode_observation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_state():
    """Return the initial Quoridor engine state."""
    import quoridor_engine as qe
    engine = qe.RuleEngine.standard()
    return engine.initial_state()


# ---------------------------------------------------------------------------
# TestCNNObservationConstants
# ---------------------------------------------------------------------------

class TestCNNObservationConstants:
    def test_version_string(self):
        assert CNN_OBSERVATION_VERSION == "dqn_obs_cnn_v1"

    def test_channels(self):
        assert CNN_CHANNELS == 7

    def test_board_size(self):
        assert CNN_BOARD_SIZE == 9

    def test_shape_tuple(self):
        assert CNN_OBSERVATION_SHAPE == (7, 9, 9)

    def test_size_equals_product(self):
        assert CNN_OBSERVATION_SIZE == 7 * 9 * 9  # 567

    def test_observation_shape_helper(self):
        assert observation_shape() == CNN_OBSERVATION_SHAPE

    def test_observation_size_helper(self):
        assert observation_size() == CNN_OBSERVATION_SIZE

    def test_distinct_from_mlp_version(self):
        assert CNN_OBSERVATION_VERSION != OBSERVATION_VERSION

    def test_distinct_from_mlp_size(self):
        assert CNN_OBSERVATION_SIZE != OBSERVATION_SIZE


# ---------------------------------------------------------------------------
# TestEncodeObservationCNN
# ---------------------------------------------------------------------------

class TestEncodeObservationCNN:
    def test_output_is_list(self):
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        assert isinstance(obs, list)

    def test_output_outer_length(self):
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        assert len(obs) == CNN_CHANNELS  # 7

    def test_output_inner_length(self):
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        for ch_idx, plane in enumerate(obs):
            assert len(plane) == CNN_BOARD_SIZE, (
                f"Channel {ch_idx}: outer (x) length {len(plane)} != {CNN_BOARD_SIZE}"
            )
            for x_idx, row in enumerate(plane):
                assert len(row) == CNN_BOARD_SIZE, (
                    f"Channel {ch_idx}[{x_idx}]: inner (y) length {len(row)} != {CNN_BOARD_SIZE}"
                )

    def test_torch_tensor_shape(self):
        """torch.tensor on the nested list must give shape [7, 9, 9]."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert t.shape == torch.Size([7, 9, 9])

    def test_all_values_finite(self):
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert torch.all(torch.isfinite(t))

    def test_all_values_in_unit_range(self):
        """All values should be in [0, 1] (one-hot or normalized counts)."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_batch_stacking(self):
        """A list of two observations must stack to [2, 7, 9, 9]."""
        state = _make_initial_state()
        obs1 = encode_observation_cnn(state)
        obs2 = encode_observation_cnn(state)
        batch = torch.tensor([obs1, obs2], dtype=torch.float32)
        assert batch.shape == torch.Size([2, 7, 9, 9])

    def test_dtype_float32(self):
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestCNNObservationSemantics
# ---------------------------------------------------------------------------

class TestCNNObservationSemantics:
    def test_channel0_current_pawn_one_hot(self):
        """Channel 0 must be a one-hot for the current player's pawn."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        ch0 = t[0]
        # Exactly one 1.0 in channel 0
        assert float(ch0.sum()) == 1.0
        # Max must be 1.0
        assert float(ch0.max()) == 1.0

    def test_channel1_opponent_pawn_one_hot(self):
        """Channel 1 must be a one-hot for the opponent's pawn."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        ch1 = t[1]
        assert float(ch1.sum()) == 1.0
        assert float(ch1.max()) == 1.0

    def test_channel0_and_1_different_cells(self):
        """Current and opponent pawns start at different positions."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        # The sum of channel 0 + channel 1 should have exactly 2 non-zero cells
        combined = t[0] + t[1]
        assert float((combined > 0).float().sum()) == 2.0

    def test_channel2_h_walls_empty_at_start(self):
        """No horizontal walls at game start — channel 2 should be all zero."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert float(t[2].sum()) == 0.0

    def test_channel3_v_walls_empty_at_start(self):
        """No vertical walls at game start — channel 3 should be all zero."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert float(t[3].sum()) == 0.0

    def test_channel4_current_player_walls_initial(self):
        """At start both players have 10 walls → normalized value = 1.0 broadcast."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        ch4 = t[4]
        # All cells should be 1.0 (10 walls / 10 = 1.0)
        assert float(ch4.min()) == pytest.approx(1.0)
        assert float(ch4.max()) == pytest.approx(1.0)

    def test_channel5_opponent_walls_initial(self):
        """Channel 5 opponent walls: also 1.0 at start."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        ch5 = t[5]
        assert float(ch5.min()) == pytest.approx(1.0)
        assert float(ch5.max()) == pytest.approx(1.0)

    def test_channel6_goal_row_indicator(self):
        """Channel 6 must mark exactly one column (9 cells) with 1.0."""
        state = _make_initial_state()
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        ch6 = t[6]
        # Exactly CNN_BOARD_SIZE cells lit (one full row in y-dimension)
        assert float(ch6.sum()) == pytest.approx(float(CNN_BOARD_SIZE))
        # All lit cells should be 1.0
        assert float(ch6.max()) == pytest.approx(1.0)

    def test_channel6_p1_goal_is_y8(self):
        """P1 (first to move) goal row is y=8 (index 8 in the y-axis)."""
        import quoridor_engine as qe
        state = _make_initial_state()
        # Initial state: current player is P1
        assert state.current_player == qe.Player.P1
        obs = encode_observation_cnn(state)
        t = torch.tensor(obs, dtype=torch.float32)
        ch6 = t[6]  # shape [9, 9]
        # Column y=8: ch6[x][8] == 1.0 for all x
        for x in range(CNN_BOARD_SIZE):
            assert float(ch6[x][8]) == pytest.approx(1.0), f"Expected ch6[{x}][8]=1.0"
        # Column y=0 should be zero for P1
        for x in range(CNN_BOARD_SIZE):
            assert float(ch6[x][0]) == pytest.approx(0.0), f"Expected ch6[{x}][0]=0.0"


# ---------------------------------------------------------------------------
# TestCNNObservationImmutability
# ---------------------------------------------------------------------------

class TestCNNObservationImmutability:
    def test_encode_does_not_mutate_state(self):
        """Encoding should not change the state object in any way."""
        import quoridor_engine as qe
        state = _make_initial_state()
        # Record state attributes before encoding
        cp_before = state.current_player
        p1_pos_before = state.pawn_pos(qe.Player.P1)
        p2_pos_before = state.pawn_pos(qe.Player.P2)

        encode_observation_cnn(state)

        assert state.current_player == cp_before
        assert state.pawn_pos(qe.Player.P1) == p1_pos_before
        assert state.pawn_pos(qe.Player.P2) == p2_pos_before


# ---------------------------------------------------------------------------
# TestCNNObservationSensitivity
# ---------------------------------------------------------------------------

class TestCNNObservationSensitivity:
    def test_encoding_changes_after_pawn_move(self):
        """Observation must differ after a pawn move."""
        import quoridor_engine as qe
        from agent_system.training.dqn.action_space import legal_action_ids, decode_action_id

        engine = qe.RuleEngine.standard()
        state = engine.initial_state()
        obs_before = torch.tensor(encode_observation_cnn(state), dtype=torch.float32)

        # Make one pawn move
        ids = legal_action_ids(engine, state)
        # Find a move action (not a wall)
        from agent_system.training.dqn.action_space import decode_action_id
        action_id = ids[0]
        action = decode_action_id(action_id, state.current_player)
        next_state = engine.apply_action(state, action)

        # Encode from next state's perspective (next player)
        obs_after = torch.tensor(encode_observation_cnn(next_state), dtype=torch.float32)

        # The two observations should differ
        assert not torch.equal(obs_before, obs_after)

    def test_two_identical_states_give_identical_obs(self):
        """Same state must always produce the same observation."""
        state = _make_initial_state()
        obs1 = torch.tensor(encode_observation_cnn(state), dtype=torch.float32)
        obs2 = torch.tensor(encode_observation_cnn(state), dtype=torch.float32)
        assert torch.equal(obs1, obs2)


# ---------------------------------------------------------------------------
# TestCNNObservationV1Preserved
# ---------------------------------------------------------------------------

class TestCNNObservationV1Preserved:
    def test_v1_version_unchanged(self):
        """dqn_obs_v1 version constant must remain 'dqn_obs_v1'."""
        assert OBSERVATION_VERSION == "dqn_obs_v1"

    def test_v1_size_unchanged(self):
        """dqn_obs_v1 OBSERVATION_SIZE must remain 292."""
        assert OBSERVATION_SIZE == 292

    def test_v1_encode_still_returns_292_floats(self):
        state = _make_initial_state()
        obs = encode_observation(state)
        assert len(obs) == 292
        assert isinstance(obs, list)

    def test_v1_tensor_shape_unchanged(self):
        state = _make_initial_state()
        obs = encode_observation(state)
        t = torch.tensor(obs, dtype=torch.float32)
        assert t.shape == torch.Size([292])

    def test_v1_and_cnn_have_different_flat_sizes(self):
        state = _make_initial_state()
        mlp_obs = encode_observation(state)
        cnn_obs = encode_observation_cnn(state)
        # MLP: 292 floats; CNN: 7*9*9=567 floats (nested)
        assert len(mlp_obs) == 292
        assert len(cnn_obs) == 7  # 7 channels
