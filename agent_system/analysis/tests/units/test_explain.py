# TEST_CLASSIFICATION: SPECIFIED
# Spec: documents/agent/analysis/decision-analysis-interface.md
import json
import random

import pytest

from agent_system.analysis.explain import AnalysisLimit, analyze, branch
from agent_system.analysis.state import (
    ENGINE, Move, Position, action_id, build_state, to_position, wire_action,
)
from agent_system.evaluation.arena.agents.scorers import GreedyScorer, MinimaxScorer
from agent_system.runtime.service.agents.greedy_agent import GreedyAgent
from agent_system.runtime.service.agents.minimax_agent import MinimaxAgent, score_position
from agent_system.runtime.service.policy import TopKPolicy


def wire(state):
    p = to_position(state)
    return {
        "pawns": {str(i+1): {"row": xy[0], "col": xy[1]} for i, xy in enumerate(p["pawns"])},
        "walls_remaining": {"1": p["walls_remaining"][0], "2": p["walls_remaining"][1]},
        "current_player": p["turn"],
        "wall_state": {field: getattr(state, field) for field in
                       ("horizontal_edges", "vertical_edges", "horizontal_heads", "vertical_heads")},
    }


def numeric(value):
    return float(value) if value in ("+inf", "-inf") else value


def positions():
    state = ENGINE.initial_state()
    rng = random.Random(872)
    result = []
    for step in range(32):
        if step in (0, 3, 8, 13, 21, 30):
            result.append(Position.model_validate(to_position(state)))
        state = ENGINE.apply_action(state, rng.choice(ENGINE.legal_actions(state)))
    result.extend([
        Position(pawns=((4, 2), (4, 1))), Position(pawns=((4, 7), (3, 3))),
        Position(pawns=((3, 5), (4, 1)), turn=2),
    ])
    return result


@pytest.mark.parametrize("position", positions())
@pytest.mark.parametrize("algorithm,depth", [("greedy", 1), ("minimax", 1), ("minimax", 2)])
def test_explanations_match_arena_scores_and_serving_choice(position, algorithm, depth):
    state = build_state(position)
    result = analyze(position, algorithm, depth)
    scorer = GreedyScorer() if algorithm == "greedy" else MinimaxScorer(depth)
    original = scorer.score(wire(state))
    assert [(c["action"], numeric(c["value"])) for c in result["candidates"]] == original
    agent = GreedyAgent() if algorithm == "greedy" else MinimaxAgent(depth)
    legal = [wire_action(a) for a in ENGINE.legal_actions(state)]
    assert result["selected_id"] == action_id(agent.make_action(wire(state), legal))
    assert result["agent_spec"]["algo_type"] == algorithm
    assert result["agent_spec"]["policy_params"] == {"k": 1}
    if algorithm == "minimax":
        assert result["agent_spec"]["algo_params"] == {"depth": depth}


def test_selected_action_is_optimal_and_root_bounds_are_honest():
    position = Position()
    state = build_state(position)
    result = analyze(position, "minimax", 2)
    exact = {}
    for candidate in result["candidates"]:
        after = ENGINE.apply_action(state, Move.model_validate(candidate["action"]).engine_action())
        # Independent depth-2 exhaustive oracle, no alpha-beta implementation.
        values = []
        for reply in ENGINE.legal_actions(after):
            leaf = ENGINE.apply_action(after, reply)
            from agent_system.analysis.state import distances
            d1, d2 = distances(leaf)
            values.append(d2-d1)
        true = min(values)
        exact[candidate["id"]] = true
        value = numeric(candidate["value"])
        if candidate["bound"] == "exact":
            assert value == true
        elif candidate["bound"] == "upper":
            assert true <= value
        else:
            assert true >= value
    assert exact[result["selected_id"]] == max(exact.values())
    # A regression guard against presenting pruned root returns as exact.
    assert any(numeric(c["value"]) != exact[c["id"]] and c["bound"] == "upper"
               for c in result["candidates"])


def test_threat_demonstrates_opponent_reply_and_terminal_semantics():
    position = Position(pawns=((4, 2), (4, 1)))
    greedy = analyze(position, "greedy")
    minimax = analyze(position, "minimax", 2)
    assert greedy["selected_id"] != minimax["selected_id"]
    chosen = next(c for c in greedy["candidates"] if c["id"] == greedy["selected_id"])
    proof = branch(position, Move.model_validate(chosen["action"]), "minimax", 2)
    assert proof["value"] == "-inf"
    assert proof["principal_variation"][-1]["winner"] == 2
    json.dumps(proof, allow_nan=False)


def test_full_branch_search_and_trace_cap_preserve_value_and_legal_pv():
    position = Position()
    move = Move(type="pawn", target=(5, 0), player=1)
    full = branch(position, move, "minimax", 3, trace_limit=1200)
    limited = branch(position, move, "minimax", 3, trace_limit=2)
    assert full["value"] == limited["value"]
    assert full["principal_variation"] == limited["principal_variation"]
    assert limited["trace"]["recorded"] == 2
    assert limited["trace"]["omitted"] > 0
    assert limited["trace"]["nodes"][0]["parent"] is None
    state = build_state(position)
    for step in full["principal_variation"]:
        state = ENGINE.apply_action(state, Move.model_validate(step["action"]).engine_action())
        assert to_position(state) == step["position"]
    assert len(full["principal_variation"]) == 3
    d1, d2 = full["principal_variation"][-1]["distances"]
    assert full["value"] == d2-d1
    # Returning an observed search must not change even the original pruned scores.
    unobserved, context = score_position(build_state(position), 3)
    observed = analyze(position, "minimax", 3)
    assert [r["score"] for r in unobserved] == [numeric(c["value"]) for c in observed["candidates"]]


@pytest.mark.parametrize("kwargs", [{"node_limit": 1}, {"seconds": 0}])
def test_incomplete_search_never_emits_a_recommendation(kwargs):
    with pytest.raises(AnalysisLimit):
        analyze(Position(), "minimax", 3, **kwargs)


def test_seeded_top_k_behavior_preserved_against_original_scores():
    position = Position()
    state = build_state(position)
    legal = [wire_action(a) for a in ENGINE.legal_actions(state)]
    expected_scores = MinimaxScorer(2).score(wire(state))
    expected = TopKPolicy(3).select(expected_scores, random.Random(51))
    agent = MinimaxAgent(2, policy=TopKPolicy(3), seed=51)
    assert agent.make_action(wire(state), legal) == expected


def test_terminal_and_immediate_win_serialize_as_valid_json():
    won = analyze(Position(pawns=((4, 8), (3, 3))), "minimax", 2)
    assert won["complete"] and won["selected_id"] is None and not won["candidates"]
    finish = analyze(Position(pawns=((4, 7), (3, 3))), "minimax", 2)
    best = next(c for c in finish["candidates"] if c["id"] == finish["selected_id"])
    assert best["immediate_win"] and best["value"] == "+inf"
    json.dumps(finish, allow_nan=False)
