# TEST_CLASSIFICATION: SPECIFIED
import math
import pytest
from agent_system.evaluation.arena.ratings import score_interval, fit_elo, update_elo_pair


def test_perfect_record_is_not_certainty():
    lo, hi = score_interval([1.0] * 25)
    assert 0.70 < lo < 0.75 and hi == 1
    assert score_interval([0.0] * 25) == pytest.approx([0, 1-lo])


def test_no_data_and_invalid_score():
    assert score_interval([]) == [0.0, 1.0]
    with pytest.raises(ValueError):
        score_interval([1.1])


def test_ratings_are_symmetric_and_order_independent():
    even = fit_elo(["a", "b", "c"], [("a", "b", 25, 50), ("b", "c", 25, 50)], "a")
    assert all(v == pytest.approx(1500) for v in even.values())
    rows = [("a", "b", 50, 50), ("a", "c", 40, 50), ("b", "c", 10, 50)]
    forward = fit_elo(["a", "b", "c"], rows, "c")
    backward = fit_elo(["a", "b", "c"], rows[::-1], "c")
    assert forward == pytest.approx(backward)
    assert forward["a"] > forward["c"] > forward["b"]
    assert all(math.isfinite(v) for v in forward.values())


def test_pair_update_preserves_total_rating():
    a, b = update_elo_pair(1500, 1500, [1, 0])
    assert (a, b) == (1500, 1500)
    a, b = update_elo_pair(1500, 1500, [1, 1])
    assert a > b and a+b == 3000
