# TEST_CLASSIFICATION: SPECIFIED
import pytest
from agent_system.evaluation.arena.matrix import make_openings, play_block, summarize

A = {"id": "a", "algo": {"type": "greedy"}, "policy": {"type": "top_k", "k": 1}}
B = {"id": "b", "algo": {"type": "random"}, "policy": {"type": "top_k", "k": 209}, "deterministic": False}


def test_opening_suite_is_reproducible_unique_and_balanced():
    book = make_openings(5, 42)
    assert book == make_openings(5, 42)
    assert len({o["state_hash"] for o in book}) == 5
    assert all(len(o["actions"]) == 4 for o in book)
    assert all(o["state"]["current_player"] == 1 for o in book)


def test_seats_accounting_and_control_exclusion():
    opening = make_openings(1, 7)[0]
    block = play_block(A, B, opening, max_plies=4, seed=8)
    assert [g["p1"] for g in block["games"]] == ["a", "b"]
    assert len(block["games"]) == 2
    assert all(g["termination"] == "ply_limit" for g in block["games"])
    report = summarize([A, B], [block], expected_blocks=1)
    assert report["ranked_games"] == 2
    assert report["matrix"]["a"]["b"]["score"] == 0.5
    assert report["matrix"]["b"]["a"]["score"] == 0.5
    assert report["matrix"]["a"]["b"]["p1_games"] == 1
    assert report["matrix"]["a"]["b"]["p2_games"] == 1
    assert [r["rank"] for r in report["ranking"]] == [1, 1]
    block["control"] = True
    report = summarize([A, B], [block], expected_blocks=1)
    assert report["ranked_games"] == 0 and report["control_games"] == 2
