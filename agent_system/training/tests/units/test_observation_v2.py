# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for agent_system/training/dqn/observation_v2.py.

Covers:
  TestV2Constants         — OBSERVATION_SIZE==292, OBSERVATION_VERSION=="dqn_obs_v2"
  TestP1Unchanged         — P1 obs identical to v1
  TestP2PawnNormalized    — P2 pawn at (4,8) → normalized slot 36
  TestP2OpponentNormalized— P1 pawn at (4,0) seen from P2 → normalized slot 125
  TestV2CurrentPawnOneHot — exactly one 1.0 in obs[0:81]
  TestV2OpponentPawnOneHot— exactly one 1.0 in obs[81:162]
  TestP2HWallFlipped      — H-wall (wx,wy) from P2 → bit idx wx*8+(7-wy)
  TestP2VWallFlipped      — V-wall (wx,wy) from P2 → bit idx wx*8+(7-wy)
  TestV1V2Distinguishable — P2 initial state: v1 ≠ v2
  TestCheckpointV2RejectsV1Expected — save v2, load with v1 expected → ValueError
  TestCheckpointV1RejectsV2Expected — save v1, load with v2 expected → ValueError
  TestV2DoesNotMutateState— state unchanged after encoding
  TestWallCountsUnchanged — wall count scalars same in v1 and v2
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch

from quoridor_engine import Action, Orientation, Player, RuleEngine

from agent_system.training.dqn.action_space import decode_action_id
from agent_system.training.dqn.checkpoint import load_checkpoint, save_checkpoint
from agent_system.training.dqn.model import QNetwork
from agent_system.training.dqn.observation import (
    OBSERVATION_SIZE as V1_SIZE,
    OBSERVATION_VERSION as V1_VERSION,
    encode_observation,
)
from agent_system.training.dqn.observation_v2 import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    OBS_OFFSET_CURRENT_PAWN,
    OBS_OFFSET_CURRENT_WALLS,
    OBS_OFFSET_H_WALLS,
    OBS_OFFSET_OPPONENT_PAWN,
    OBS_OFFSET_OPPONENT_WALLS,
    OBS_OFFSET_V_WALLS,
    encode_observation_v2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


@pytest.fixture(scope="module")
def p1_initial(engine) -> object:
    """Initial state where it is P1's turn."""
    return engine.initial_state()


@pytest.fixture(scope="module")
def p2_initial(engine) -> object:
    """State where it is P2's turn (after one P1 move)."""
    state = engine.initial_state()
    # P1 moves forward (y+1 relative to P1 goal direction)
    mask_ids = [i for i, v in enumerate(
        __import__("agent_system.training.dqn.action_space", fromlist=["legal_action_mask"])
        .legal_action_mask(engine, state)
    ) if v]
    action = decode_action_id(mask_ids[0], Player.P1)
    return engine.apply_action(state, action)


# ---------------------------------------------------------------------------
# TestV2Constants
# ---------------------------------------------------------------------------

class TestV2Constants:
    def test_observation_size_is_292(self):
        assert OBSERVATION_SIZE == 292

    def test_observation_version_is_dqn_obs_v2(self):
        assert OBSERVATION_VERSION == "dqn_obs_v2"

    def test_v2_size_matches_v1_size(self):
        assert OBSERVATION_SIZE == V1_SIZE

    def test_v2_version_differs_from_v1_version(self):
        assert OBSERVATION_VERSION != V1_VERSION


# ---------------------------------------------------------------------------
# TestP1Unchanged
# ---------------------------------------------------------------------------

class TestP1Unchanged:
    def test_p1_v2_equals_v1(self, p1_initial):
        """For P1, encode_observation_v2 must produce identical output to v1."""
        v1 = encode_observation(p1_initial)
        v2 = encode_observation_v2(p1_initial)
        assert v1 == v2

    def test_p1_v2_length(self, p1_initial):
        v2 = encode_observation_v2(p1_initial)
        assert len(v2) == OBSERVATION_SIZE


# ---------------------------------------------------------------------------
# TestP2PawnNormalized
# ---------------------------------------------------------------------------

class TestP2PawnNormalized:
    def test_p2_pawn_normalized_slot(self, engine):
        """P2 starting pawn at (4,8) should appear at normalized slot 36 (4*9+0)."""
        state = engine.initial_state()
        # Advance to P2's turn
        action = decode_action_id(81 + 0, Player.P1)  # H-wall at (0,0) — any P1 action
        # Actually use a safe pawn move: P1 moves up to (4,1)
        p1_move_action = decode_action_id(4 * 9 + 1, Player.P1)  # x=4, y=1
        state = engine.apply_action(state, p1_move_action)

        # Now it's P2's turn; P2 pawn is at (4,8)
        assert state.current_player == Player.P2
        v2 = encode_observation_v2(state)

        # In v2 for P2: pawn (4,8) → (4, 8-8) = (4,0) → slot 4*9+0 = 36
        expected_slot = 4 * 9 + 0
        assert v2[OBS_OFFSET_CURRENT_PAWN + expected_slot] == 1.0

    def test_p2_pawn_exactly_one_hot(self, engine):
        """Normalized P2 pawn plane has exactly one 1.0."""
        state = engine.initial_state()
        p1_move_action = decode_action_id(4 * 9 + 1, Player.P1)
        state = engine.apply_action(state, p1_move_action)
        assert state.current_player == Player.P2

        v2 = encode_observation_v2(state)
        pawn_plane = v2[OBS_OFFSET_CURRENT_PAWN: OBS_OFFSET_CURRENT_PAWN + 81]
        assert sum(pawn_plane) == pytest.approx(1.0)
        assert pawn_plane[4 * 9 + 0] == 1.0


# ---------------------------------------------------------------------------
# TestP2OpponentNormalized
# ---------------------------------------------------------------------------

class TestP2OpponentNormalized:
    def test_p1_pawn_seen_from_p2_normalized(self, engine):
        """From P2's view, P1 at (4,1) should appear at normalized opponent slot 4*9+(8-1)=4*9+7=43 → offset 81+43=124."""
        state = engine.initial_state()
        p1_move_action = decode_action_id(4 * 9 + 1, Player.P1)  # P1 moves to (4,1)
        state = engine.apply_action(state, p1_move_action)
        assert state.current_player == Player.P2

        v2 = encode_observation_v2(state)
        # P1 is at (4,1). From P2's y-flip view: y → 8-y = 7. Slot: 4*9+7 = 43
        expected_slot = 4 * 9 + 7
        assert v2[OBS_OFFSET_OPPONENT_PAWN + expected_slot] == 1.0

    def test_p2_opponent_plane_exactly_one_hot(self, engine):
        """Opponent pawn plane (from P2 view) has exactly one 1.0."""
        state = engine.initial_state()
        p1_move_action = decode_action_id(4 * 9 + 1, Player.P1)
        state = engine.apply_action(state, p1_move_action)
        assert state.current_player == Player.P2

        v2 = encode_observation_v2(state)
        opp_plane = v2[OBS_OFFSET_OPPONENT_PAWN: OBS_OFFSET_OPPONENT_PAWN + 81]
        assert sum(opp_plane) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestV2CurrentPawnOneHot
# ---------------------------------------------------------------------------

class TestV2CurrentPawnOneHot:
    def test_current_pawn_one_hot_p1(self, p1_initial):
        v2 = encode_observation_v2(p1_initial)
        pawn = v2[OBS_OFFSET_CURRENT_PAWN: OBS_OFFSET_CURRENT_PAWN + 81]
        assert sum(pawn) == pytest.approx(1.0)
        assert max(pawn) == 1.0

    def test_current_pawn_one_hot_p2(self, engine):
        state = engine.initial_state()
        state = engine.apply_action(state, decode_action_id(4 * 9 + 1, Player.P1))
        v2 = encode_observation_v2(state)
        pawn = v2[OBS_OFFSET_CURRENT_PAWN: OBS_OFFSET_CURRENT_PAWN + 81]
        assert sum(pawn) == pytest.approx(1.0)
        assert max(pawn) == 1.0


# ---------------------------------------------------------------------------
# TestV2OpponentPawnOneHot
# ---------------------------------------------------------------------------

class TestV2OpponentPawnOneHot:
    def test_opponent_pawn_one_hot_p1(self, p1_initial):
        v2 = encode_observation_v2(p1_initial)
        opp = v2[OBS_OFFSET_OPPONENT_PAWN: OBS_OFFSET_OPPONENT_PAWN + 81]
        assert sum(opp) == pytest.approx(1.0)
        assert max(opp) == 1.0

    def test_opponent_pawn_one_hot_p2(self, engine):
        state = engine.initial_state()
        state = engine.apply_action(state, decode_action_id(4 * 9 + 1, Player.P1))
        v2 = encode_observation_v2(state)
        opp = v2[OBS_OFFSET_OPPONENT_PAWN: OBS_OFFSET_OPPONENT_PAWN + 81]
        assert sum(opp) == pytest.approx(1.0)
        assert max(opp) == 1.0


# ---------------------------------------------------------------------------
# TestP2HWallFlipped
# ---------------------------------------------------------------------------

class TestP2HWallFlipped:
    def test_h_wall_flipped_for_p2(self, engine):
        """H-wall placed at (wx, wy) should appear at bit wx*8+(7-wy) from P2's view."""
        # Place a P1 H-wall at (2, 3) then verify P2 sees it at (2, 4)
        # H-wall action id: 81 + wx*8 + wy = 81 + 2*8 + 3 = 81 + 19 = 100
        state = engine.initial_state()
        wall_action = decode_action_id(81 + 2 * 8 + 3, Player.P1)
        state = engine.apply_action(state, wall_action)
        assert state.current_player == Player.P2

        v2 = encode_observation_v2(state)
        # wall at (wx=2, wy=3) should appear at bit 2*8+(7-3)=2*8+4=20
        expected_bit = 2 * 8 + (7 - 3)
        assert v2[OBS_OFFSET_H_WALLS + expected_bit] == 1.0
        # Original slot (bit=19) should be zero
        original_bit = 2 * 8 + 3
        assert v2[OBS_OFFSET_H_WALLS + original_bit] == 0.0

    def test_h_wall_count_preserved(self, engine):
        """Total number of H-walls in v2 equals total in v1."""
        state = engine.initial_state()
        wall_action = decode_action_id(81 + 2 * 8 + 3, Player.P1)
        state = engine.apply_action(state, wall_action)
        v1 = encode_observation(state)
        v2 = encode_observation_v2(state)
        h_count_v1 = sum(v1[OBS_OFFSET_H_WALLS: OBS_OFFSET_H_WALLS + 64])
        h_count_v2 = sum(v2[OBS_OFFSET_H_WALLS: OBS_OFFSET_H_WALLS + 64])
        assert h_count_v1 == pytest.approx(h_count_v2)


# ---------------------------------------------------------------------------
# TestP2VWallFlipped
# ---------------------------------------------------------------------------

class TestP2VWallFlipped:
    def test_v_wall_flipped_for_p2(self, engine):
        """V-wall placed at (wx, wy) should appear at bit wx*8+(7-wy) from P2's view."""
        # V-wall action id: 145 + wx*8 + wy = 145 + 1*8 + 2 = 155
        state = engine.initial_state()
        wall_action = decode_action_id(145 + 1 * 8 + 2, Player.P1)
        state = engine.apply_action(state, wall_action)
        assert state.current_player == Player.P2

        v2 = encode_observation_v2(state)
        # wall at (wx=1, wy=2) → bit 1*8+(7-2) = 1*8+5 = 13
        expected_bit = 1 * 8 + (7 - 2)
        assert v2[OBS_OFFSET_V_WALLS + expected_bit] == 1.0
        original_bit = 1 * 8 + 2
        assert v2[OBS_OFFSET_V_WALLS + original_bit] == 0.0

    def test_v_wall_count_preserved(self, engine):
        """Total number of V-walls in v2 equals total in v1."""
        state = engine.initial_state()
        wall_action = decode_action_id(145 + 1 * 8 + 2, Player.P1)
        state = engine.apply_action(state, wall_action)
        v1 = encode_observation(state)
        v2 = encode_observation_v2(state)
        v_count_v1 = sum(v1[OBS_OFFSET_V_WALLS: OBS_OFFSET_V_WALLS + 64])
        v_count_v2 = sum(v2[OBS_OFFSET_V_WALLS: OBS_OFFSET_V_WALLS + 64])
        assert v_count_v1 == pytest.approx(v_count_v2)


# ---------------------------------------------------------------------------
# TestV1V2Distinguishable
# ---------------------------------------------------------------------------

class TestV1V2Distinguishable:
    def test_p2_v1_neq_v2(self, engine):
        """For P2's initial turn, v1 and v2 must produce different encodings."""
        state = engine.initial_state()
        state = engine.apply_action(state, decode_action_id(4 * 9 + 1, Player.P1))
        v1 = encode_observation(state)
        v2 = encode_observation_v2(state)
        assert v1 != v2

    def test_p2_pawn_slot_differs_v1_v2(self, engine):
        """P2 pawn occupies different slots in v1 vs v2."""
        state = engine.initial_state()
        state = engine.apply_action(state, decode_action_id(4 * 9 + 1, Player.P1))
        v1 = encode_observation(state)
        v2 = encode_observation_v2(state)
        # v1: P2 pawn at (4,8) → slot 4*9+8 = 44
        assert v1[OBS_OFFSET_CURRENT_PAWN + 44] == 1.0
        # v2: P2 pawn at (4,8) → normalized (4,0) → slot 36
        assert v2[OBS_OFFSET_CURRENT_PAWN + 36] == 1.0


# ---------------------------------------------------------------------------
# TestCheckpointV2RejectsV1Expected / TestCheckpointV1RejectsV2Expected
# ---------------------------------------------------------------------------

class TestCheckpointVersionEnforcement:
    def test_v2_checkpoint_rejected_by_v1_expected(self, tmp_path):
        """A checkpoint saved with v2 must raise ValueError when loaded with v1 expected."""
        net = QNetwork()
        path = tmp_path / "ckpt_v2.pt"
        save_checkpoint(path, net, obs_version="dqn_obs_v2")
        with pytest.raises(ValueError, match="observation_version"):
            load_checkpoint(path, expected_obs_version="dqn_obs_v1")

    def test_v1_checkpoint_rejected_by_v2_expected(self, tmp_path):
        """A checkpoint saved with v1 must raise ValueError when loaded with v2 expected."""
        net = QNetwork()
        path = tmp_path / "ckpt_v1.pt"
        save_checkpoint(path, net, obs_version="dqn_obs_v1")
        with pytest.raises(ValueError, match="observation_version"):
            load_checkpoint(path, expected_obs_version="dqn_obs_v2")

    def test_v2_checkpoint_loads_with_v2_expected(self, tmp_path):
        """A checkpoint saved with v2 loads successfully when v2 is expected."""
        net = QNetwork()
        path = tmp_path / "ckpt_v2.pt"
        save_checkpoint(path, net, obs_version="dqn_obs_v2")
        ckpt = load_checkpoint(path, expected_obs_version="dqn_obs_v2")
        assert ckpt.observation_version == "dqn_obs_v2"


# ---------------------------------------------------------------------------
# TestV2DoesNotMutateState
# ---------------------------------------------------------------------------

class TestV2DoesNotMutateState:
    def test_p1_state_unchanged(self, p1_initial):
        px, py = p1_initial.pawn_pos(Player.P1)
        _ = encode_observation_v2(p1_initial)
        px2, py2 = p1_initial.pawn_pos(Player.P1)
        assert (px, py) == (px2, py2)

    def test_p2_state_unchanged(self, engine):
        state = engine.initial_state()
        state = engine.apply_action(state, decode_action_id(4 * 9 + 1, Player.P1))
        p2x, p2y = state.pawn_pos(Player.P2)
        _ = encode_observation_v2(state)
        p2x2, p2y2 = state.pawn_pos(Player.P2)
        assert (p2x, p2y) == (p2x2, p2y2)


# ---------------------------------------------------------------------------
# TestWallCountsUnchanged
# ---------------------------------------------------------------------------

class TestWallCountsUnchanged:
    def test_p1_wall_counts_equal_v1_v2(self, p1_initial):
        v1 = encode_observation(p1_initial)
        v2 = encode_observation_v2(p1_initial)
        assert v1[OBS_OFFSET_CURRENT_WALLS] == pytest.approx(v2[OBS_OFFSET_CURRENT_WALLS])
        assert v1[OBS_OFFSET_OPPONENT_WALLS] == pytest.approx(v2[OBS_OFFSET_OPPONENT_WALLS])

    def test_p2_wall_counts_equal_v1_v2(self, engine):
        state = engine.initial_state()
        state = engine.apply_action(state, decode_action_id(4 * 9 + 1, Player.P1))
        v1 = encode_observation(state)
        v2 = encode_observation_v2(state)
        assert v1[OBS_OFFSET_CURRENT_WALLS] == pytest.approx(v2[OBS_OFFSET_CURRENT_WALLS])
        assert v1[OBS_OFFSET_OPPONENT_WALLS] == pytest.approx(v2[OBS_OFFSET_OPPONENT_WALLS])

    def test_p1_initial_walls_normalized(self, p1_initial):
        v2 = encode_observation_v2(p1_initial)
        # P1 starts with 10 walls → 10/10 = 1.0
        assert v2[OBS_OFFSET_CURRENT_WALLS] == pytest.approx(1.0)
        assert v2[OBS_OFFSET_OPPONENT_WALLS] == pytest.approx(1.0)
