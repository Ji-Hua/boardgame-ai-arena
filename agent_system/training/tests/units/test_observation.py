# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for agent_system/training/dqn/observation.py.

Covers:
    A. Constants — OBSERVATION_SIZE matches actual encoded length
    B. Segment lengths — pawn planes, wall planes, scalar slots
    C. Determinism — same state produces same vector
    D. Current-player pawn one-hot has exactly one active entry
    E. Opponent pawn one-hot has exactly one active entry
    F. Remaining-wall scalars are in valid normalized range
    G. Current-player-centric encoding: pawn planes switch after a turn
    H. Wall occupancy changes after a legal wall placement
    I. Encoder does not mutate the game state
    J. Integration: env.reset() and env.step() return encoded observations
"""

from __future__ import annotations

import pytest

from quoridor_engine import Action, Orientation, Player, RuleEngine

from agent_system.training.dqn.observation import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    OBS_OFFSET_CURRENT_PAWN,
    OBS_OFFSET_CURRENT_WALLS,
    OBS_OFFSET_H_WALLS,
    OBS_OFFSET_OPPONENT_PAWN,
    OBS_OFFSET_OPPONENT_WALLS,
    OBS_OFFSET_V_WALLS,
    _WALL_HEADS,
    encode_observation,
)
from agent_system.training.dqn.env import QuoridorEnv
from agent_system.training.dqn.action_space import legal_action_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


@pytest.fixture(scope="module")
def initial_state(engine: RuleEngine):
    return engine.initial_state()


@pytest.fixture(scope="module")
def initial_obs(initial_state) -> list[float]:
    return encode_observation(initial_state)


# ===========================================================================
# A. Constants
# ===========================================================================

class TestConstants:
    def test_observation_size_constant_is_292(self) -> None:
        assert OBSERVATION_SIZE == 292

    def test_observation_size_matches_encoded_length(self, initial_obs) -> None:
        assert len(initial_obs) == OBSERVATION_SIZE

    def test_version_string(self) -> None:
        assert OBSERVATION_VERSION == "dqn_obs_v1"

    def test_segment_offsets_partition_vector(self) -> None:
        """Offsets must form a contiguous non-overlapping partition."""
        # pawn planes: 81 + 81 = 162
        assert OBS_OFFSET_CURRENT_PAWN == 0
        assert OBS_OFFSET_OPPONENT_PAWN == OBS_OFFSET_CURRENT_PAWN + 81
        # wall planes: 64 + 64 = 128
        assert OBS_OFFSET_H_WALLS == OBS_OFFSET_OPPONENT_PAWN + 81
        assert OBS_OFFSET_V_WALLS == OBS_OFFSET_H_WALLS + 64
        # scalars: 1 + 1 = 2
        assert OBS_OFFSET_CURRENT_WALLS == OBS_OFFSET_V_WALLS + 64
        assert OBS_OFFSET_OPPONENT_WALLS == OBS_OFFSET_CURRENT_WALLS + 1
        assert OBS_OFFSET_OPPONENT_WALLS + 1 == OBSERVATION_SIZE


# ===========================================================================
# B. Segment lengths
# ===========================================================================

class TestSegmentLengths:
    def test_current_pawn_segment_length(self, initial_obs) -> None:
        segment = initial_obs[OBS_OFFSET_CURRENT_PAWN:OBS_OFFSET_OPPONENT_PAWN]
        assert len(segment) == 81

    def test_opponent_pawn_segment_length(self, initial_obs) -> None:
        segment = initial_obs[OBS_OFFSET_OPPONENT_PAWN:OBS_OFFSET_H_WALLS]
        assert len(segment) == 81

    def test_h_wall_segment_length(self, initial_obs) -> None:
        segment = initial_obs[OBS_OFFSET_H_WALLS:OBS_OFFSET_V_WALLS]
        assert len(segment) == 64

    def test_v_wall_segment_length(self, initial_obs) -> None:
        segment = initial_obs[OBS_OFFSET_V_WALLS:OBS_OFFSET_CURRENT_WALLS]
        assert len(segment) == 64

    def test_wall_heads_constant(self) -> None:
        assert _WALL_HEADS == 64


# ===========================================================================
# C. Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_state_produces_same_vector(self, initial_state) -> None:
        obs1 = encode_observation(initial_state)
        obs2 = encode_observation(initial_state)
        assert obs1 == obs2

    def test_encoding_is_all_floats(self, initial_obs) -> None:
        assert all(isinstance(v, float) for v in initial_obs)


# ===========================================================================
# D. Current-player pawn one-hot
# ===========================================================================

class TestCurrentPlayerPawnPlane:
    def test_exactly_one_active_entry(self, initial_obs) -> None:
        segment = initial_obs[OBS_OFFSET_CURRENT_PAWN:OBS_OFFSET_OPPONENT_PAWN]
        assert segment.count(1.0) == 1
        assert segment.count(0.0) == 80

    def test_active_entry_matches_p1_start(self, engine: RuleEngine, initial_state) -> None:
        """Initial state has P1 to move; current player is P1 at start position."""
        topo = engine.topology
        p1_start_x, p1_start_y = topo.start_pos(Player.P1)
        expected_idx = OBS_OFFSET_CURRENT_PAWN + p1_start_x * 9 + p1_start_y
        obs = encode_observation(initial_state)
        assert obs[expected_idx] == 1.0


# ===========================================================================
# E. Opponent pawn one-hot
# ===========================================================================

class TestOpponentPawnPlane:
    def test_exactly_one_active_entry(self, initial_obs) -> None:
        segment = initial_obs[OBS_OFFSET_OPPONENT_PAWN:OBS_OFFSET_H_WALLS]
        assert segment.count(1.0) == 1
        assert segment.count(0.0) == 80

    def test_active_entry_matches_p2_start(self, engine: RuleEngine, initial_state) -> None:
        """Initial state has P1 to move; opponent is P2 at start position."""
        topo = engine.topology
        p2_start_x, p2_start_y = topo.start_pos(Player.P2)
        expected_idx = OBS_OFFSET_OPPONENT_PAWN + p2_start_x * 9 + p2_start_y
        obs = encode_observation(initial_state)
        assert obs[expected_idx] == 1.0


# ===========================================================================
# F. Remaining-wall scalars
# ===========================================================================

class TestRemainingWallScalars:
    def test_scalars_are_in_unit_range(self, initial_obs) -> None:
        assert 0.0 <= initial_obs[OBS_OFFSET_CURRENT_WALLS] <= 1.0
        assert 0.0 <= initial_obs[OBS_OFFSET_OPPONENT_WALLS] <= 1.0

    def test_initial_walls_are_normalized_to_one(self, initial_obs) -> None:
        """Both players start with 10 walls; MAX_WALLS=10, so normalized = 1.0."""
        assert initial_obs[OBS_OFFSET_CURRENT_WALLS] == pytest.approx(1.0)
        assert initial_obs[OBS_OFFSET_OPPONENT_WALLS] == pytest.approx(1.0)


# ===========================================================================
# G. Current-player-centric: pawn planes switch after a turn
# ===========================================================================

class TestCurrentPlayerCentricEncoding:
    def test_current_pawn_plane_switches_after_move(
        self, engine: RuleEngine, initial_state
    ) -> None:
        """After a pawn move, P2 is current. The 'current pawn' plane should
        now reflect P2's position, and 'opponent pawn' should reflect P1."""
        topo = engine.topology

        # Pick a legal pawn move for P1.
        for action in engine.legal_actions(initial_state):
            if str(action.kind) == "MovePawn":
                next_state = engine.apply_action(initial_state, action)
                break
        else:
            pytest.skip("No pawn move available in initial state.")

        # After P1's move, current player is P2.
        assert next_state.current_player == Player.P2
        obs_after = encode_observation(next_state)

        # Current pawn should now show P2's start position (unchanged since P2
        # has not moved yet).
        p2_start_x, p2_start_y = topo.start_pos(Player.P2)
        current_pawn_idx = OBS_OFFSET_CURRENT_PAWN + p2_start_x * 9 + p2_start_y
        assert obs_after[current_pawn_idx] == 1.0

        # Opponent pawn should show P1's new position.
        p1_new_x, p1_new_y = next_state.pawn_pos(Player.P1)
        opponent_pawn_idx = OBS_OFFSET_OPPONENT_PAWN + p1_new_x * 9 + p1_new_y
        assert obs_after[opponent_pawn_idx] == 1.0

    def test_pawn_planes_have_one_active_each_after_move(
        self, engine: RuleEngine, initial_state
    ) -> None:
        """Both pawn planes must always have exactly one active entry."""
        for action in engine.legal_actions(initial_state):
            if str(action.kind) == "MovePawn":
                next_state = engine.apply_action(initial_state, action)
                break
        else:
            pytest.skip("No pawn move available in initial state.")

        obs = encode_observation(next_state)
        cur_seg = obs[OBS_OFFSET_CURRENT_PAWN:OBS_OFFSET_OPPONENT_PAWN]
        opp_seg = obs[OBS_OFFSET_OPPONENT_PAWN:OBS_OFFSET_H_WALLS]
        assert cur_seg.count(1.0) == 1
        assert opp_seg.count(1.0) == 1


# ===========================================================================
# H. Wall occupancy changes after a legal wall placement
# ===========================================================================

class TestWallOccupancy:
    def test_h_wall_occupancy_zero_at_initial(self, initial_obs) -> None:
        """No walls placed at game start."""
        h_seg = initial_obs[OBS_OFFSET_H_WALLS:OBS_OFFSET_V_WALLS]
        assert all(v == 0.0 for v in h_seg)

    def test_v_wall_occupancy_zero_at_initial(self, initial_obs) -> None:
        """No walls placed at game start."""
        v_seg = initial_obs[OBS_OFFSET_V_WALLS:OBS_OFFSET_CURRENT_WALLS]
        assert all(v == 0.0 for v in v_seg)

    def test_h_wall_entry_set_after_placement(
        self, engine: RuleEngine, initial_state
    ) -> None:
        """After placing a horizontal wall at (x, y), bit x*8+y in h_wall
        segment must become 1.0."""
        # Find a legal horizontal wall placement.
        for action in engine.legal_actions(initial_state):
            if str(action.kind) == "PlaceWall" and str(action.coordinate_kind) == "Horizontal":
                wx, wy = action.target_x, action.target_y
                next_state = engine.apply_action(initial_state, action)
                break
        else:
            pytest.skip("No legal horizontal wall in initial state.")

        obs = encode_observation(next_state)
        bit_idx = wx * 8 + wy
        assert obs[OBS_OFFSET_H_WALLS + bit_idx] == 1.0

    def test_v_wall_entry_set_after_placement(
        self, engine: RuleEngine, initial_state
    ) -> None:
        """After placing a vertical wall at (x, y), bit x*8+y in v_wall
        segment must become 1.0."""
        for action in engine.legal_actions(initial_state):
            if str(action.kind) == "PlaceWall" and str(action.coordinate_kind) == "Vertical":
                wx, wy = action.target_x, action.target_y
                next_state = engine.apply_action(initial_state, action)
                break
        else:
            pytest.skip("No legal vertical wall in initial state.")

        obs = encode_observation(next_state)
        bit_idx = wx * 8 + wy
        assert obs[OBS_OFFSET_V_WALLS + bit_idx] == 1.0

    def test_wall_count_decreases_after_placement(
        self, engine: RuleEngine, initial_state
    ) -> None:
        """After current player places a wall, their normalized wall count drops."""
        for action in engine.legal_actions(initial_state):
            if str(action.kind) == "PlaceWall":
                next_state = engine.apply_action(initial_state, action)
                break
        else:
            pytest.skip("No legal wall in initial state.")

        # After a wall placement, the turn changes. The player who placed the wall
        # is now the *opponent* in next_state's encoding.
        obs_before = encode_observation(initial_state)
        obs_after = encode_observation(next_state)

        # Before: current player (P1) has walls = 1.0; after: P1 is now opponent.
        opp_walls_after = obs_after[OBS_OFFSET_OPPONENT_WALLS]
        assert opp_walls_after < 1.0


# ===========================================================================
# I. Encoder does not mutate game state
# ===========================================================================

class TestNoStateMutation:
    def test_encode_does_not_mutate_pawn_positions(
        self, engine: RuleEngine, initial_state
    ) -> None:
        p1_before = initial_state.pawn_pos(Player.P1)
        p2_before = initial_state.pawn_pos(Player.P2)
        encode_observation(initial_state)
        assert initial_state.pawn_pos(Player.P1) == p1_before
        assert initial_state.pawn_pos(Player.P2) == p2_before

    def test_encode_does_not_mutate_wall_bitsets(
        self, engine: RuleEngine, initial_state
    ) -> None:
        h_before = initial_state.horizontal_heads
        v_before = initial_state.vertical_heads
        encode_observation(initial_state)
        assert initial_state.horizontal_heads == h_before
        assert initial_state.vertical_heads == v_before


# ===========================================================================
# J. Integration: env returns encoded observations
# ===========================================================================

class TestEnvIntegration:
    @pytest.fixture
    def env(self, engine: RuleEngine) -> QuoridorEnv:
        return QuoridorEnv(engine)

    def test_reset_returns_list_of_floats(self, env: QuoridorEnv) -> None:
        obs = env.reset()
        assert isinstance(obs, list)
        assert all(isinstance(v, float) for v in obs)

    def test_reset_obs_length_matches_observation_size(self, env: QuoridorEnv) -> None:
        obs = env.reset()
        assert len(obs) == OBSERVATION_SIZE

    def test_step_returns_obs_of_correct_length(self, env: QuoridorEnv) -> None:
        env.reset()
        ids = env.legal_action_ids()
        next_obs, _, _, _ = env.step(ids[0])
        assert len(next_obs) == OBSERVATION_SIZE

    def test_step_obs_is_list_of_floats(self, env: QuoridorEnv) -> None:
        env.reset()
        ids = env.legal_action_ids()
        next_obs, _, _, _ = env.step(ids[0])
        assert isinstance(next_obs, list)
        assert all(isinstance(v, float) for v in next_obs)

    def test_reset_obs_consistent_with_encode_observation(
        self, env: QuoridorEnv, engine: RuleEngine
    ) -> None:
        """env.reset() must return the same vector as encode_observation(initial_state)."""
        obs_from_env = env.reset()
        obs_direct = encode_observation(engine.initial_state())
        assert obs_from_env == obs_direct

    def test_step_obs_consistent_with_encode_observation(
        self, env: QuoridorEnv, engine: RuleEngine
    ) -> None:
        """env.step() obs must match encode_observation(raw_state()) after the step."""
        env.reset()
        ids = env.legal_action_ids()
        obs_from_step, _, _, _ = env.step(ids[0])
        obs_direct = encode_observation(env.raw_state())
        assert obs_from_step == obs_direct

    def test_info_contains_observation_version(self, env: QuoridorEnv) -> None:
        env.reset()
        ids = env.legal_action_ids()
        _, _, _, info = env.step(ids[0])
        assert info.get("observation_version") == OBSERVATION_VERSION

    def test_raw_state_still_accessible(self, env: QuoridorEnv) -> None:
        env.reset()
        raw = env.raw_state()
        assert raw is not None
        # Should have pawn_pos method.
        assert hasattr(raw, "pawn_pos")
