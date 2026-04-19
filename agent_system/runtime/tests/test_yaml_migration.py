# TEST_CLASSIFICATION: SPECIFIED
"""Tests for the Agent Service YAML migration.

Covers:
  A. YAML loading — definitions parse correctly
  B. Construction — agents can be instantiated from YAML defs
  C. Policy support — policy flows through YAML path correctly
  D. Service integration — AgentService registers YAML agents correctly
  E. Regression — behavior matches old builtin spec path
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_system.runtime.service.yaml_loader import (
    AgentDefinition,
    load_definition,
    load_definitions_from_dir,
    create_agent_from_definition,
    parse_agent_definition,
)
from agent_system.runtime.service.specs.yaml_agent_spec import YamlAgentMaterializer
from agent_system.runtime.service.service import AgentService
from agent_system.runtime.service.agents.greedy_agent import GreedyAgent
from agent_system.runtime.service.agents.minimax_agent import MinimaxAgent
from agent_system.runtime.service.agents.random_agent import RandomAgentV2

_DEFS_DIR = Path(__file__).resolve().parent.parent.parent / "definition" / "agent_defs"

# Shared game state for agent action tests
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
# A. YAML Loading
# ===================================================================

class TestYamlLoading:
    def test_load_greedy_definition(self):
        defn = load_definition(_DEFS_DIR / "greedy.yaml")
        assert defn.id == "greedy"
        assert defn.algo_type == "greedy"
        assert defn.category == "ai"
        assert defn.deterministic is True
        assert defn.policy_k == 1

    def test_load_minimax_definition(self):
        defn = load_definition(_DEFS_DIR / "minimax.yaml")
        assert defn.id == "minimax"
        assert defn.algo_type == "minimax"
        assert defn.algo_params["depth"] == 2
        assert defn.policy_k == 1

    def test_load_random_v2_definition(self):
        defn = load_definition(_DEFS_DIR / "random_v2.yaml")
        assert defn.id == "random_v2"
        assert defn.algo_type == "random_v2"
        assert defn.algo_params["threshold"] == 0.8
        assert defn.deterministic is False

    def test_load_minimax_presets(self):
        simple = load_definition(_DEFS_DIR / "minimax_simple.yaml")
        medium = load_definition(_DEFS_DIR / "minimax_medium.yaml")
        hard = load_definition(_DEFS_DIR / "minimax_hard.yaml")
        assert simple.algo_params["depth"] == 2
        assert medium.algo_params["depth"] == 3
        assert hard.algo_params["depth"] == 5

    def test_load_all_from_dir(self):
        defs = load_definitions_from_dir(_DEFS_DIR)
        ids = {d.id for d in defs}
        assert "greedy" in ids
        assert "minimax" in ids
        assert "random_v2" in ids

    def test_parse_definition_defaults(self):
        data = {"id": "test", "algo": {"type": "greedy"}}
        defn = parse_agent_definition(data)
        assert defn.policy_config == {"type": "top_k", "k": 1}
        assert defn.category == "ai"
        assert defn.deterministic is True


# ===================================================================
# B. Construction
# ===================================================================

class TestConstruction:
    def test_create_greedy_from_definition(self):
        defn = load_definition(_DEFS_DIR / "greedy.yaml")
        agent = create_agent_from_definition(defn)
        assert isinstance(agent, GreedyAgent)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action

    def test_create_minimax_from_definition(self):
        defn = load_definition(_DEFS_DIR / "minimax.yaml")
        agent = create_agent_from_definition(defn)
        assert isinstance(agent, MinimaxAgent)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action

    def test_create_random_v2_from_definition(self):
        defn = load_definition(_DEFS_DIR / "random_v2.yaml")
        agent = create_agent_from_definition(defn)
        assert isinstance(agent, RandomAgentV2)
        action = agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action

    def test_minimax_depth_from_yaml(self):
        defn = load_definition(_DEFS_DIR / "minimax_hard.yaml")
        agent = create_agent_from_definition(defn)
        assert isinstance(agent, MinimaxAgent)
        assert agent._depth == 5

    def test_config_overrides_depth(self):
        defn = load_definition(_DEFS_DIR / "minimax.yaml")
        agent = create_agent_from_definition(defn, config_overrides={"depth": 3})
        assert isinstance(agent, MinimaxAgent)
        assert agent._depth == 3


# ===================================================================
# C. Policy Support
# ===================================================================

class TestPolicySupport:
    def test_k1_no_policy_injected(self):
        """k=1 in YAML → no policy injected (deterministic default path)."""
        defn = load_definition(_DEFS_DIR / "greedy.yaml")
        agent = create_agent_from_definition(defn)
        assert agent._policy is None

    def test_topk3_override_injects_policy(self):
        """Policy override with k=3 → policy is injected."""
        defn = load_definition(_DEFS_DIR / "greedy.yaml")
        agent = create_agent_from_definition(
            defn,
            config_overrides={"policy": {"type": "top_k", "k": 3}},
            context={"seed": 42},
        )
        assert agent._policy is not None
        assert agent._policy.k == 3

    def test_yaml_spec_with_policy_override(self):
        """YamlAgentMaterializer correctly passes policy override to agent."""
        defn = load_definition(_DEFS_DIR / "minimax.yaml")
        spec = YamlAgentMaterializer(defn)
        config = {"depth": 1, "policy": {"type": "top_k", "k": 3}}
        context = {"seed": 42}
        agent = spec.create_instance(config, context)
        assert isinstance(agent, MinimaxAgent)
        assert agent._policy is not None

    def test_k1_deterministic_regression(self):
        """k=1 via YAML → identical action as direct construction."""
        defn = load_definition(_DEFS_DIR / "greedy.yaml")
        yaml_agent = create_agent_from_definition(defn)
        direct_agent = GreedyAgent()
        assert yaml_agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS) == \
               direct_agent.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)


# ===================================================================
# D. Service Integration
# ===================================================================

class TestServiceIntegration:
    def test_service_lists_yaml_agents(self):
        """AgentService has YAML-loaded types in its registry."""
        svc = AgentService()
        types = svc.list_types()
        type_ids = {t["type_id"] for t in types}
        assert "greedy" in type_ids
        assert "minimax" in type_ids
        assert "random_v2" in type_ids
        # Presets
        assert "minimax_simple" in type_ids
        assert "minimax_medium" in type_ids
        assert "minimax_hard" in type_ids
        # Non-YAML agents still registered
        assert "dummy" in type_ids
        assert "random" in type_ids
        assert "replay" in type_ids

    def test_service_creates_greedy(self):
        svc = AgentService()
        iid = svc.create_agent("greedy", "test_room", 1)
        assert isinstance(iid, str)

    def test_service_creates_minimax(self):
        svc = AgentService()
        iid = svc.create_agent("minimax", "test_room", 1)
        assert isinstance(iid, str)

    def test_service_creates_minimax_with_depth(self):
        svc = AgentService()
        iid = svc.create_agent("minimax", "test_room", 1, config={"depth": 3})
        assert isinstance(iid, str)

    def test_service_get_action(self):
        svc = AgentService()
        svc.create_agent("greedy", "test_room", 1)
        svc.start_room_agents("test_room")
        action = svc.get_action("test_room", 1, _INITIAL_STATE, _LEGAL_ACTIONS)
        assert "player" in action
        assert "type" in action
        assert "target" in action

    def test_service_display_names(self):
        svc = AgentService()
        types = {t["type_id"]: t["display_name"] for t in svc.list_types()}
        assert types["greedy"] == "Greedy Agent"
        assert types["minimax"] == "Minimax Agent"
        assert types["minimax_simple"] == "Minimax Agent (Simple)"
        assert types["minimax_medium"] == "Minimax Agent (Medium)"

    def test_service_agent_categories(self):
        svc = AgentService()
        types = {t["type_id"]: t["category"] for t in svc.list_types()}
        assert types["greedy"] == "ai"
        assert types["minimax"] == "ai"
        assert types["random_v2"] == "scripted"


# ===================================================================
# E. Regression — YAML path matches old builtin spec behavior
# ===================================================================

class TestRegression:
    def test_greedy_yaml_matches_direct(self):
        """Greedy from YAML produces same action as direct GreedyAgent()."""
        svc = AgentService()
        svc.create_agent("greedy", "reg_room", 1)
        svc.start_room_agents("reg_room")
        yaml_action = svc.get_action("reg_room", 1, _INITIAL_STATE, _LEGAL_ACTIONS)

        direct = GreedyAgent()
        direct_action = direct.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)

        assert yaml_action == direct_action

    def test_minimax_yaml_matches_direct(self):
        """Minimax from YAML produces same action as direct MinimaxAgent(depth=2)."""
        svc = AgentService()
        svc.create_agent("minimax", "reg_room2", 1)
        svc.start_room_agents("reg_room2")
        yaml_action = svc.get_action("reg_room2", 1, _INITIAL_STATE, _LEGAL_ACTIONS)

        direct = MinimaxAgent(depth=2)
        direct_action = direct.make_action(_INITIAL_STATE, _LEGAL_ACTIONS)

        assert yaml_action == direct_action
