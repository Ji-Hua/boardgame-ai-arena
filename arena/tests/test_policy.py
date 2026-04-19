# TEST_CLASSIFICATION: SPECIFIED
"""Tests for the top-k policy system.

Covers:
  A. Determinism — same seed → identical action sequence
  B. Non-determinism — different seeds → divergent action sequences (k > 1)
  C. Regression — k=1 → identical to original deterministic behavior
  D. Integration — short agent-vs-agent game completes successfully

Tests cover both:
  - Arena agents (arena/agents/core.py abstractions)
  - Agent service agents (agents/agent_service/ BaseAgent subclasses)
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Arena imports
# ---------------------------------------------------------------------------
from arena.agents.core import Agent, AgentInstance, Policy, TopKPolicy
from arena.agents.loader import load_agent
from arena.runner import play_single_game

# ---------------------------------------------------------------------------
# Agent service imports
# ---------------------------------------------------------------------------
from agents.agent_service.policy import (
    TopKPolicy as ServiceTopKPolicy,
    build_policy,
)
from agents.agent_service.agents.greedy_agent import GreedyAgent
from agents.agent_service.agents.minimax_agent import MinimaxAgent
from agents.agent_service.specs.yaml_agent_spec import YamlAgentMaterializer
from agents.agent_service.yaml_loader import load_definition

_AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "agents" / "agent_defs"

# ---------------------------------------------------------------------------
# Shared game state fixtures
# ---------------------------------------------------------------------------

_INITIAL_STATE = {
    "current_player": 1,
    "pawns": {
        "1": {"row": 4, "col": 0},
        "2": {"row": 4, "col": 8},
    },
    "walls_remaining": {"1": 10, "2": 10},
    "wall_state": {
        "horizontal_edges": 0,
        "vertical_edges": 0,
        "horizontal_heads": 0,
        "vertical_heads": 0,
    },
    "game_over": False,
    "winner": None,
}

_LEGAL_ACTIONS = [
    {"player": 1, "type": "pawn", "target": [3, 0]},
    {"player": 1, "type": "pawn", "target": [4, 1]},
    {"player": 1, "type": "pawn", "target": [5, 0]},
]


# ===================================================================
# Unit tests: TopKPolicy
# ===================================================================

class TestTopKPolicyUnit:
    """Direct tests for the TopKPolicy select() method."""

    def test_k1_returns_best(self):
        """k=1 always selects the highest-scoring action."""
        policy = TopKPolicy(k=1)
        scored = [
            ({"type": "pawn", "target": [0, 0]}, 1.0),
            ({"type": "pawn", "target": [0, 1]}, 3.0),
            ({"type": "pawn", "target": [0, 2]}, 2.0),
        ]
        rng = random.Random(42)
        result = policy.select(scored, rng)
        assert result == {"type": "pawn", "target": [0, 1]}

    def test_k1_deterministic_across_rngs(self):
        """k=1 returns the same action regardless of RNG seed."""
        policy = TopKPolicy(k=1)
        scored = [
            ({"type": "pawn", "target": [0, 0]}, 5.0),
            ({"type": "pawn", "target": [0, 1]}, 3.0),
        ]
        results = set()
        for seed in range(100):
            r = policy.select(scored, random.Random(seed))
            results.add(r["target"][0] * 10 + r["target"][1])
        assert len(results) == 1, "k=1 must always pick the same action"

    def test_topk_selects_from_top(self):
        """k=3 should only select from top-3 scoring actions."""
        policy = TopKPolicy(k=3)
        scored = [
            ({"id": "a"}, 10.0),
            ({"id": "b"}, 8.0),
            ({"id": "c"}, 6.0),
            ({"id": "d"}, 4.0),
            ({"id": "e"}, 2.0),
        ]
        seen_ids = set()
        for seed in range(200):
            result = policy.select(scored, random.Random(seed))
            seen_ids.add(result["id"])
        # Should only ever see top-3
        assert seen_ids <= {"a", "b", "c"}
        # With enough seeds, should see at least 2 distinct
        assert len(seen_ids) >= 2

    def test_stable_sort_ties(self):
        """Tied scores maintain original order (stable sort)."""
        policy = TopKPolicy(k=1)
        scored = [
            ({"id": "first"}, 5.0),
            ({"id": "second"}, 5.0),
        ]
        # k=1 with stable sort: first item (original order) wins
        result = policy.select(scored, random.Random(0))
        assert result["id"] == "first"

    def test_k_larger_than_list(self):
        """k > len(actions) selects from all actions."""
        policy = TopKPolicy(k=100)
        scored = [
            ({"id": "a"}, 1.0),
            ({"id": "b"}, 2.0),
        ]
        seen = set()
        for seed in range(100):
            r = policy.select(scored, random.Random(seed))
            seen.add(r["id"])
        assert seen == {"a", "b"}

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            TopKPolicy(k=0)
        with pytest.raises(ValueError):
            TopKPolicy(k=-1)


class TestServiceTopKPolicyUnit:
    """Verify agent-service TopKPolicy matches arena TopKPolicy behavior."""

    def test_k1_returns_best(self):
        policy = ServiceTopKPolicy(k=1)
        scored = [
            ({"id": "low"}, 1.0),
            ({"id": "high"}, 5.0),
        ]
        result = policy.select(scored, random.Random(0))
        assert result["id"] == "high"

    def test_k_property(self):
        assert ServiceTopKPolicy(k=3).k == 3


class TestBuildPolicy:
    """Test the build_policy factory function."""

    def test_none_config_returns_none(self):
        assert build_policy(None) is None

    def test_top_k_config(self):
        p = build_policy({"type": "top_k", "k": 5})
        assert isinstance(p, ServiceTopKPolicy)
        assert p.k == 5

    def test_default_k(self):
        p = build_policy({"type": "top_k"})
        assert isinstance(p, ServiceTopKPolicy)
        assert p.k == 1

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown policy type"):
            build_policy({"type": "mcts"})


# ===================================================================
# A. Determinism tests — same seed → identical actions
# ===================================================================

class TestDeterminism:
    """Same seed must produce identical action sequences."""

    def test_greedy_determinism_no_policy(self):
        """GreedyAgent without policy: identical calls → identical results."""
        a = GreedyAgent()
        b = GreedyAgent()
        assert a.make_action(_INITIAL_STATE, _LEGAL_ACTIONS) == \
               b.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)

    def test_greedy_determinism_with_policy(self):
        """GreedyAgent + TopK(3) + same seed → identical results."""
        policy = ServiceTopKPolicy(k=3)
        a = GreedyAgent(policy=policy, seed=42)
        b = GreedyAgent(policy=policy, seed=42)
        assert a.make_action(_INITIAL_STATE, _LEGAL_ACTIONS) == \
               b.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)

    def test_minimax_determinism_no_policy(self):
        """MinimaxAgent without policy: identical calls → identical results."""
        a = MinimaxAgent(depth=1)
        b = MinimaxAgent(depth=1)
        assert a.make_action(_INITIAL_STATE, _LEGAL_ACTIONS) == \
               b.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)

    def test_minimax_determinism_with_policy(self):
        """MinimaxAgent + TopK(3) + same seed → identical results."""
        policy = ServiceTopKPolicy(k=3)
        a = MinimaxAgent(depth=1, policy=policy, seed=42)
        b = MinimaxAgent(depth=1, policy=policy, seed=42)
        assert a.make_action(_INITIAL_STATE, _LEGAL_ACTIONS) == \
               b.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)

    def test_arena_determinism_topk3(self):
        """Arena agent with k=3 and same seed → identical game results."""
        agent = load_agent(_AGENT_DIR / "greedy_topk3.yaml")
        r1 = play_single_game(
            AgentInstance(agent, seed=10),
            AgentInstance(agent, seed=11),
            seed=99,
        )
        r2 = play_single_game(
            AgentInstance(agent, seed=10),
            AgentInstance(agent, seed=11),
            seed=99,
        )
        assert r1.winner == r2.winner
        assert r1.num_steps == r2.num_steps


# ===================================================================
# B. Non-determinism tests — different seeds → divergence
# ===================================================================

class TestNonDeterminism:
    """Different seeds with k > 1 should produce different action sequences."""

    def test_greedy_different_seeds_diverge(self):
        """GreedyAgent + TopK(3) + different seeds → at least some divergence."""
        policy = ServiceTopKPolicy(k=3)
        actions = set()
        for seed in range(50):
            agent = GreedyAgent(policy=policy, seed=seed)
            action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
            actions.add(str(action))
        # With k=3 and 50 different seeds, we should see > 1 distinct action
        assert len(actions) > 1, \
            "k=3 with different seeds must produce at least some variation"

    def test_minimax_different_seeds_diverge(self):
        """MinimaxAgent + TopK(3) + different seeds → at least some divergence."""
        policy = ServiceTopKPolicy(k=3)
        actions = set()
        for seed in range(50):
            agent = MinimaxAgent(depth=1, policy=policy, seed=seed)
            action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
            actions.add(str(action))
        assert len(actions) > 1, \
            "k=3 with different seeds must produce at least some variation"

    def test_arena_topk3_different_seeds_diverge(self):
        """Arena greedy_topk3 with different seeds → different game outcomes."""
        agent = load_agent(_AGENT_DIR / "greedy_topk3.yaml")
        outcomes = set()
        for base in range(20):
            r = play_single_game(
                AgentInstance(agent, seed=base * 2),
                AgentInstance(agent, seed=base * 2 + 1),
                seed=base,
            )
            outcomes.add((r.winner, r.num_steps))
        # With enough seed variation and k=3, outcomes should vary
        assert len(outcomes) > 1, \
            "Different seeds with k=3 should produce different game outcomes"


# ===================================================================
# C. Regression tests — k=1 → identical to old implementation
# ===================================================================

class TestRegressionK1:
    """k=1 must reproduce exactly the same behavior as no policy (old impl)."""

    def test_greedy_k1_matches_no_policy(self):
        """GreedyAgent with TopK(k=1) == GreedyAgent without policy."""
        baseline = GreedyAgent()
        with_k1 = GreedyAgent(policy=ServiceTopKPolicy(k=1), seed=42)

        baseline_action = baseline.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        k1_action = with_k1.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert baseline_action == k1_action

    def test_minimax_k1_matches_no_policy(self):
        """MinimaxAgent with TopK(k=1) == MinimaxAgent without policy."""
        baseline = MinimaxAgent(depth=1)
        with_k1 = MinimaxAgent(depth=1, policy=ServiceTopKPolicy(k=1), seed=42)

        baseline_action = baseline.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        k1_action = with_k1.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert baseline_action == k1_action

    def test_arena_k1_matches_k1_yaml(self):
        """Arena greedy.yaml (k=1) → same result as before."""
        agent_k1 = load_agent(_AGENT_DIR / "greedy.yaml")
        r1 = play_single_game(
            AgentInstance(agent_k1, seed=10),
            AgentInstance(agent_k1, seed=11),
            seed=99,
        )
        r2 = play_single_game(
            AgentInstance(agent_k1, seed=10),
            AgentInstance(agent_k1, seed=11),
            seed=99,
        )
        assert r1.winner == r2.winner
        assert r1.num_steps == r2.num_steps


# ===================================================================
# D. Integration tests — full games complete successfully
# ===================================================================

class TestIntegration:
    """Short agent-vs-agent games must complete without errors."""

    def test_arena_greedy_topk3_game(self):
        """A game between greedy_topk3 agents completes."""
        agent = load_agent(_AGENT_DIR / "greedy_topk3.yaml")
        r = play_single_game(
            AgentInstance(agent, seed=100),
            AgentInstance(agent, seed=101),
            seed=42,
        )
        assert r.num_steps > 0
        assert r.winner is not None or r.num_steps > 0

    def test_arena_minimax_topk3_game(self):
        """A game between minimax_d2_topk3 agents completes."""
        agent = load_agent(_AGENT_DIR / "minimax_d2_topk3.yaml")
        r = play_single_game(
            AgentInstance(agent, seed=200),
            AgentInstance(agent, seed=201),
            seed=42,
        )
        assert r.num_steps > 0

    def test_arena_mixed_topk_game(self):
        """greedy_topk3 vs greedy (k=1) game completes."""
        topk3 = load_agent(_AGENT_DIR / "greedy_topk3.yaml")
        det = load_agent(_AGENT_DIR / "greedy.yaml")
        r = play_single_game(
            AgentInstance(topk3, seed=300),
            AgentInstance(det, seed=301),
            seed=42,
        )
        assert r.num_steps > 0

    def test_service_greedy_with_policy_makes_valid_action(self):
        """GreedyAgent with policy produces a valid action dict."""
        agent = GreedyAgent(policy=ServiceTopKPolicy(k=3), seed=42)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action
        assert "type" in action
        assert "target" in action

    def test_service_minimax_with_policy_makes_valid_action(self):
        """MinimaxAgent with policy produces a valid action dict."""
        agent = MinimaxAgent(depth=1, policy=ServiceTopKPolicy(k=3), seed=42)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action
        assert "type" in action
        assert "target" in action


# ===================================================================
# Spec integration tests
# ===================================================================

class TestSpecPolicySupport:
    """Materializers correctly pass policy config to agents."""

    def test_greedy_spec_with_policy(self):
        defn = load_definition(_AGENT_DIR / "greedy.yaml")
        mat = YamlAgentMaterializer(defn)
        config = {"policy": {"type": "top_k", "k": 3}}
        context = {"seed": 42}
        agent = mat.create_instance(config, context)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action

    def test_greedy_spec_without_policy(self):
        """No policy config → original deterministic behavior."""
        defn = load_definition(_AGENT_DIR / "greedy.yaml")
        mat = YamlAgentMaterializer(defn)
        agent = mat.create_instance({})
        a1 = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        agent2 = mat.create_instance({})
        a2 = agent2.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert a1 == a2

    def test_minimax_spec_with_policy(self):
        defn = load_definition(_AGENT_DIR / "minimax.yaml")
        mat = YamlAgentMaterializer(defn)
        config = {"depth": 1, "policy": {"type": "top_k", "k": 3}}
        context = {"seed": 42}
        agent = mat.create_instance(config, context)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action

    def test_minimax_spec_without_policy(self):
        """No policy config → original deterministic behavior."""
        defn = load_definition(_AGENT_DIR / "minimax.yaml")
        mat = YamlAgentMaterializer(defn)
        agent = mat.create_instance({"depth": 1})
        a1 = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        agent2 = mat.create_instance({"depth": 1})
        a2 = agent2.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert a1 == a2
