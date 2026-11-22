# TEST_CLASSIFICATION: SPECIFIED
# Spec: documents/agent/analysis/decision-analysis-interface.md
import random

import pytest
from pydantic import ValidationError
from quoridor_engine import Player

from agent_system.analysis.state import (
    ENGINE, Move, Position, build_state, snapshot, to_position,
)


def test_legal_play_roundtrips_and_path_witnesses_match_engine():
    rng = random.Random(194)
    state = ENGINE.initial_state()
    for _ in range(65):
        position = to_position(state)
        reconstructed = build_state(Position.model_validate(position))
        assert to_position(reconstructed) == position
        view = snapshot(reconstructed)
        for i, player in enumerate((Player.P1, Player.P2)):
            assert len(view["paths"][i])-1 == view["distances"][i]
            assert tuple(view["paths"][i][0]) == tuple(state.pawn_pos(player))
            assert view["paths"][i][-1][1] == (8 if i == 0 else 0)
        if view["winner"]:
            break
        state = ENGINE.apply_action(state, rng.choice(ENGINE.legal_actions(state)))


@pytest.mark.parametrize("kwargs", [
    {"pawns": ((9, 0), (4, 8))}, {"pawns": ((4, 8), (4, 8))},
    {"pawns": ((4, 8), (4, 0))}, {"walls_remaining": (9, 10)},
    {"walls_remaining": (11, 9)}, {"pawns": ((True, 0), (4, 8))},
    {"turn": 3}, {"unknown": 0},
])
def test_malformed_positions_rejected_before_engine(kwargs):
    with pytest.raises(ValidationError):
        Position(**kwargs)


@pytest.mark.parametrize("walls", [
    [{"x": 3, "y": 3, "orientation": "horizontal"}]*2,
    [{"x": 3, "y": 3, "orientation": "horizontal"},
     {"x": 3, "y": 3, "orientation": "vertical"}],
    [{"x": 3, "y": 3, "orientation": "horizontal"},
     {"x": 4, "y": 3, "orientation": "horizontal"}],
    [{"x": 3, "y": 3, "orientation": "vertical"},
     {"x": 3, "y": 4, "orientation": "vertical"}],
])
def test_crossing_and_overlapping_walls_rejected(walls):
    with pytest.raises(ValueError):
        build_state(Position(walls=walls, walls_remaining=(8, 10)))


def test_wall_sealing_pawn_at_board_edge_is_rejected():
    # P1 at a1 is trapped by one wall above and one to its right.
    position = Position(pawns=((0, 0), (4, 8)), walls_remaining=(8, 10), walls=[
        {"x": 0, "y": 0, "orientation": "horizontal"},
        {"x": 0, "y": 0, "orientation": "vertical"},
    ])
    # This case also crosses: rejection must occur before unsafe FFI state use.
    with pytest.raises(ValueError):
        build_state(position)
    # Three noncrossing walls surround a pawn; path preservation is the rejection.
    position = Position(pawns=((1, 0), (4, 8)), walls_remaining=(7, 10), walls=[
        {"x": 0, "y": 1, "orientation": "horizontal"},
        {"x": 0, "y": 0, "orientation": "vertical"},
        {"x": 1, "y": 0, "orientation": "vertical"},
    ])
    with pytest.raises(ValueError, match="路径"):
        build_state(position)


def test_wall_coordinate_eight_rejected_before_constructor():
    with pytest.raises(ValueError):
        Move(type="horizontal", target=(8, 0), player=1).engine_action()


def test_terminal_position_has_no_actions_and_zero_winner_distance():
    view = snapshot(build_state(Position(pawns=((4, 8), (3, 3)))))
    assert view["winner"] == 1
    assert view["distances"][0] == 0
    assert view["legal_actions"] == []
