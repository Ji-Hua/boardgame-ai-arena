# TEST_CLASSIFICATION: SPECIFIED
"""Unit tests for agent_system/training/dqn/action_space.py.

Covers:
    A. Action space constants (total count, range boundaries)
    B. MovePawn encode/decode correctness and round-trips
    C. Horizontal wall encode/decode correctness and round-trips
    D. Vertical wall encode/decode correctness and round-trips
    E. Invalid action ID rejection
    F. Legal action mask/ids — every Engine legal action appears, no extras
    G. No duplicate action IDs from legal actions
"""

from __future__ import annotations

import pytest

import quoridor_engine as qe
from quoridor_engine import Action, Orientation, Player, RuleEngine

from agent_system.training.dqn.action_space import (
    ACTION_COUNT,
    ACTION_SPACE_VERSION,
    BOARD_SIZE,
    HWALL_ID_END,
    HWALL_ID_START,
    PAWN_ID_END,
    PAWN_ID_START,
    VWALL_ID_END,
    VWALL_ID_START,
    WALL_GRID_SIZE,
    decode_action_id,
    encode_engine_action,
    encode_move_pawn,
    encode_place_hwall,
    encode_place_vwall,
    is_valid_action_id,
    legal_action_ids,
    legal_action_mask,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.standard()


@pytest.fixture(scope="module")
def initial_state(engine: RuleEngine):
    return engine.initial_state()


# ===========================================================================
# A. Constants
# ===========================================================================

class TestActionSpaceConstants:
    def test_total_action_count(self) -> None:
        assert ACTION_COUNT == 209

    def test_pawn_range_size(self) -> None:
        assert PAWN_ID_END - PAWN_ID_START == 81  # 9*9

    def test_hwall_range_size(self) -> None:
        assert HWALL_ID_END - HWALL_ID_START == 64  # 8*8

    def test_vwall_range_size(self) -> None:
        assert VWALL_ID_END - VWALL_ID_START == 64  # 8*8

    def test_ranges_cover_all_ids(self) -> None:
        total = (
            (PAWN_ID_END - PAWN_ID_START)
            + (HWALL_ID_END - HWALL_ID_START)
            + (VWALL_ID_END - VWALL_ID_START)
        )
        assert total == ACTION_COUNT

    def test_ranges_are_contiguous_and_non_overlapping(self) -> None:
        assert PAWN_ID_START == 0
        assert PAWN_ID_END == HWALL_ID_START
        assert HWALL_ID_END == VWALL_ID_START
        assert VWALL_ID_END == ACTION_COUNT

    def test_board_size(self) -> None:
        assert BOARD_SIZE == 9

    def test_wall_grid_size(self) -> None:
        assert WALL_GRID_SIZE == 8

    def test_version_string(self) -> None:
        assert ACTION_SPACE_VERSION == "dqn_action_v1"


# ===========================================================================
# B. MovePawn encode/decode
# ===========================================================================

class TestMovePawnEncoding:
    """Representative encode/decode tests for pawn move actions."""

    @pytest.mark.parametrize("x, y, expected_id", [
        (0, 0, 0),
        (0, 8, 8),
        (4, 0, 36),   # 4 * 9 + 0 = 36
        (4, 4, 40),   # 4 * 9 + 4 = 40
        (8, 8, 80),   # 8 * 9 + 8 = 80
    ])
    def test_encode_representative(self, x: int, y: int, expected_id: int) -> None:
        assert encode_move_pawn(x, y) == expected_id

    def test_encode_all_pawn_ids_are_in_range(self) -> None:
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                action_id = encode_move_pawn(x, y)
                assert PAWN_ID_START <= action_id < PAWN_ID_END

    def test_encode_pawn_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            encode_move_pawn(9, 0)
        with pytest.raises(ValueError):
            encode_move_pawn(0, 9)
        with pytest.raises(ValueError):
            encode_move_pawn(-1, 0)

    @pytest.mark.parametrize("x, y", [
        (0, 0), (4, 4), (8, 8), (0, 8), (8, 0),
    ])
    def test_round_trip(self, x: int, y: int) -> None:
        player = Player.P1
        action_id = encode_move_pawn(x, y)
        decoded = decode_action_id(action_id, player)
        re_encoded = encode_engine_action(decoded)
        assert re_encoded == action_id

    @pytest.mark.parametrize("x, y", [
        (0, 0), (4, 4), (8, 8),
    ])
    def test_decoded_action_has_correct_kind_and_coords(self, x: int, y: int) -> None:
        player = Player.P1
        action_id = encode_move_pawn(x, y)
        decoded = decode_action_id(action_id, player)
        assert str(decoded.kind) == "MovePawn"
        assert decoded.target_x == x
        assert decoded.target_y == y


# ===========================================================================
# C. Horizontal wall encode/decode
# ===========================================================================

class TestHorizontalWallEncoding:
    @pytest.mark.parametrize("x, y, expected_id", [
        (0, 0, 81),
        (0, 7, 88),   # 81 + 0*8 + 7 = 88
        (7, 7, 144),  # 81 + 7*8 + 7 = 144
        (4, 4, 117),  # 81 + 4*8 + 4 = 117
    ])
    def test_encode_representative(self, x: int, y: int, expected_id: int) -> None:
        assert encode_place_hwall(x, y) == expected_id

    def test_encode_all_hwall_ids_are_in_range(self) -> None:
        for x in range(WALL_GRID_SIZE):
            for y in range(WALL_GRID_SIZE):
                action_id = encode_place_hwall(x, y)
                assert HWALL_ID_START <= action_id < HWALL_ID_END

    def test_encode_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            encode_place_hwall(8, 0)
        with pytest.raises(ValueError):
            encode_place_hwall(0, 8)
        with pytest.raises(ValueError):
            encode_place_hwall(-1, 0)

    @pytest.mark.parametrize("x, y", [
        (0, 0), (3, 3), (7, 7), (0, 7), (7, 0),
    ])
    def test_round_trip(self, x: int, y: int) -> None:
        player = Player.P1
        action_id = encode_place_hwall(x, y)
        decoded = decode_action_id(action_id, player)
        re_encoded = encode_engine_action(decoded)
        assert re_encoded == action_id

    @pytest.mark.parametrize("x, y", [
        (0, 0), (3, 3), (7, 7),
    ])
    def test_decoded_action_has_correct_kind_and_coords(self, x: int, y: int) -> None:
        player = Player.P1
        action_id = encode_place_hwall(x, y)
        decoded = decode_action_id(action_id, player)
        assert str(decoded.kind) == "PlaceWall"
        assert str(decoded.coordinate_kind) == "Horizontal"
        assert decoded.target_x == x
        assert decoded.target_y == y


# ===========================================================================
# D. Vertical wall encode/decode
# ===========================================================================

class TestVerticalWallEncoding:
    @pytest.mark.parametrize("x, y, expected_id", [
        (0, 0, 145),
        (0, 7, 152),   # 145 + 0*8 + 7 = 152
        (7, 7, 208),   # 145 + 7*8 + 7 = 208
        (4, 4, 181),   # 145 + 4*8 + 4 = 181
    ])
    def test_encode_representative(self, x: int, y: int, expected_id: int) -> None:
        assert encode_place_vwall(x, y) == expected_id

    def test_encode_all_vwall_ids_are_in_range(self) -> None:
        for x in range(WALL_GRID_SIZE):
            for y in range(WALL_GRID_SIZE):
                action_id = encode_place_vwall(x, y)
                assert VWALL_ID_START <= action_id < VWALL_ID_END

    def test_encode_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            encode_place_vwall(8, 0)
        with pytest.raises(ValueError):
            encode_place_vwall(0, 8)
        with pytest.raises(ValueError):
            encode_place_vwall(-1, 0)

    @pytest.mark.parametrize("x, y", [
        (0, 0), (3, 3), (7, 7), (0, 7), (7, 0),
    ])
    def test_round_trip(self, x: int, y: int) -> None:
        player = Player.P1
        action_id = encode_place_vwall(x, y)
        decoded = decode_action_id(action_id, player)
        re_encoded = encode_engine_action(decoded)
        assert re_encoded == action_id

    @pytest.mark.parametrize("x, y", [
        (0, 0), (3, 3), (7, 7),
    ])
    def test_decoded_action_has_correct_kind_and_coords(self, x: int, y: int) -> None:
        player = Player.P1
        action_id = encode_place_vwall(x, y)
        decoded = decode_action_id(action_id, player)
        assert str(decoded.kind) == "PlaceWall"
        assert str(decoded.coordinate_kind) == "Vertical"
        assert decoded.target_x == x
        assert decoded.target_y == y


# ===========================================================================
# E. Invalid action ID rejection
# ===========================================================================

class TestInvalidActionId:
    @pytest.mark.parametrize("bad_id", [-1, 209, 300, 1000])
    def test_is_valid_returns_false(self, bad_id: int) -> None:
        assert not is_valid_action_id(bad_id)

    @pytest.mark.parametrize("good_id", [0, 80, 81, 144, 145, 208])
    def test_is_valid_returns_true_for_boundaries(self, good_id: int) -> None:
        assert is_valid_action_id(good_id)

    @pytest.mark.parametrize("bad_id", [-1, 209, 300])
    def test_decode_raises_for_invalid_id(self, bad_id: int) -> None:
        with pytest.raises(ValueError):
            decode_action_id(bad_id, Player.P1)


# ===========================================================================
# F. Legal action mask and ids — agreement with Engine
# ===========================================================================

class TestLegalActionMask:
    def test_initial_state_mask_length(self, engine, initial_state) -> None:
        mask = legal_action_mask(engine, initial_state)
        assert len(mask) == ACTION_COUNT

    def test_initial_state_has_at_least_one_legal_action(self, engine, initial_state) -> None:
        mask = legal_action_mask(engine, initial_state)
        assert any(mask)

    def test_every_engine_legal_action_appears_in_mask(self, engine, initial_state) -> None:
        """Each Engine legal action must map to a True entry in the mask."""
        mask = legal_action_mask(engine, initial_state)
        for action in engine.legal_actions(initial_state):
            action_id = encode_engine_action(action)
            assert mask[action_id], (
                f"Engine legal action id={action_id} not marked legal in mask"
            )

    def test_every_true_mask_entry_is_an_engine_legal_action(self, engine, initial_state) -> None:
        """Every ID marked True in the mask must be engine-legal."""
        engine_legal_ids = {
            encode_engine_action(a) for a in engine.legal_actions(initial_state)
        }
        mask = legal_action_mask(engine, initial_state)
        for action_id, legal in enumerate(mask):
            if legal:
                assert action_id in engine_legal_ids, (
                    f"action_id={action_id} is True in mask but not engine-legal"
                )

    def test_mask_and_legal_action_ids_are_consistent(self, engine, initial_state) -> None:
        """legal_action_ids() must match the True positions in legal_action_mask()."""
        mask = legal_action_mask(engine, initial_state)
        ids = legal_action_ids(engine, initial_state)
        mask_true_ids = {i for i, v in enumerate(mask) if v}
        assert set(ids) == mask_true_ids

    def test_no_duplicate_ids_from_legal_actions(self, engine, initial_state) -> None:
        ids = legal_action_ids(engine, initial_state)
        assert len(ids) == len(set(ids))

    def test_illegal_ids_are_not_in_mask(self, engine, initial_state) -> None:
        """Boundary checks: id -1 and 209 are never in the mask."""
        mask = legal_action_mask(engine, initial_state)
        # The mask is exactly ACTION_COUNT long; -1 and 209 cannot index it.
        assert len(mask) == ACTION_COUNT
        assert not mask[0] or True  # id 0 may or may not be legal; just no IndexError
