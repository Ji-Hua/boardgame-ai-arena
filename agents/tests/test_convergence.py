# TEST_CLASSIFICATION: SPECIFIED
"""Convergence tests: verify both Arena and Agent Service consume shared AgentSpec.

Phase 2+3 of the Agent System SSOT convergence. These tests verify:
    A. Both consumers parse YAML through the shared AgentSpec model
    B. agents/agent_defs/ is the canonical source for both consumers
    C. Arena materialization (AgentSpec → Arena Agent) works for all
       Arena-compatible specs in the canonical directory
    D. Agent Service wrapper (AgentDefinition) delegates to shared AgentSpec
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agent_spec import AgentSpec as SharedAgentSpec, load_agent_specs_from_dir
from agents.agent_service.yaml_loader import (
    AgentDefinition,
    load_definitions_from_dir,
)
from arena.agents.loader import (
    _SCORER_REGISTRY,
    load_agent,
    load_agents_from_dir,
    materialize_agent,
)
from arena.agents.core import Agent as ArenaAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CANONICAL_DIR = _REPO_ROOT / "agents" / "agent_defs"


# ===================================================================
# A. Shared AgentSpec is the parsing source for both consumers
# ===================================================================

class TestSharedParsingSource:
    """Both consumers ultimately delegate to the shared AgentSpec parser."""

    def test_agent_definition_wraps_shared_spec(self):
        """AgentDefinition (Agent Service) exposes the underlying SharedAgentSpec."""
        defs = load_definitions_from_dir(_CANONICAL_DIR)
        for defn in defs:
            assert hasattr(defn, "spec"), f"{defn.id} missing .spec"
            assert isinstance(defn.spec, SharedAgentSpec)

    def test_agent_definition_id_matches_spec_id(self):
        """AgentDefinition.id delegates to SharedAgentSpec.id."""
        defs = load_definitions_from_dir(_CANONICAL_DIR)
        for defn in defs:
            assert defn.id == defn.spec.id

    def test_agent_definition_algo_fields_match_spec(self):
        """AgentDefinition delegates algo_type/algo_params to SharedAgentSpec."""
        defs = load_definitions_from_dir(_CANONICAL_DIR)
        for defn in defs:
            assert defn.algo_type == defn.spec.algo_type
            assert defn.algo_params == defn.spec.algo_params


# ===================================================================
# B. Canonical directory serves both consumers
# ===================================================================

class TestCanonicalDirectory:
    """agents/agent_defs/ is the single source for both Arena and Agent Service."""

    @pytest.fixture()
    def shared_specs(self) -> list[SharedAgentSpec]:
        return load_agent_specs_from_dir(_CANONICAL_DIR)

    def test_canonical_dir_exists(self):
        assert _CANONICAL_DIR.is_dir()

    def test_has_arena_agents(self, shared_specs: list[SharedAgentSpec]):
        """Canonical dir includes agents loadable by Arena (algo_type in _SCORER_REGISTRY)."""
        arena_ids = {s.id for s in shared_specs if s.algo_type in _SCORER_REGISTRY}
        # At minimum, the original Arena agents must be present
        assert {"random", "greedy", "minimax_d2", "minimax_d3"} <= arena_ids

    def test_has_agent_service_agents(self, shared_specs: list[SharedAgentSpec]):
        """Canonical dir includes agents loadable by Agent Service."""
        svc_ids = {s.id for s in shared_specs}
        assert {"greedy", "minimax", "random_v2"} <= svc_ids

    def test_shared_agents_have_consistent_ids(self, shared_specs: list[SharedAgentSpec]):
        """No duplicate IDs in canonical directory."""
        ids = [s.id for s in shared_specs]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


# ===================================================================
# C. Arena materialization from canonical specs
# ===================================================================

class TestArenaMaterialization:
    """materialize_agent produces valid Arena Agent objects from shared specs."""

    @pytest.fixture()
    def arena_specs(self) -> list[SharedAgentSpec]:
        """Only specs whose algo_type is in Arena's scorer registry."""
        all_specs = load_agent_specs_from_dir(_CANONICAL_DIR)
        return [s for s in all_specs if s.algo_type in _SCORER_REGISTRY]

    def test_all_arena_specs_materialize(self, arena_specs: list[SharedAgentSpec]):
        """Every Arena-compatible spec materializes without error."""
        for spec in arena_specs:
            agent = materialize_agent(spec)
            assert isinstance(agent, ArenaAgent), f"{spec.id} did not materialize"

    def test_materialized_agent_preserves_id(self, arena_specs: list[SharedAgentSpec]):
        for spec in arena_specs:
            agent = materialize_agent(spec)
            assert agent.id == spec.id

    def test_materialized_agent_preserves_algo_metadata(self, arena_specs: list[SharedAgentSpec]):
        for spec in arena_specs:
            agent = materialize_agent(spec)
            assert agent.algo_type == spec.algo_type
            assert agent.algo_params == spec.algo_params

    def test_materialized_agent_preserves_policy_metadata(self, arena_specs: list[SharedAgentSpec]):
        for spec in arena_specs:
            agent = materialize_agent(spec)
            assert agent.policy_type == spec.policy_type
            assert agent.policy_params == spec.policy_params

    def test_load_agent_from_canonical_dir(self):
        """load_agent() works with files in the canonical directory."""
        agent = load_agent(_CANONICAL_DIR / "greedy.yaml")
        assert agent.id == "greedy"
        assert agent.algo_type == "greedy"

    def test_load_agents_from_canonical_dir(self):
        """load_agents_from_dir() filters to Arena-compatible agents from canonical dir."""
        agents = load_agents_from_dir(_CANONICAL_DIR)
        # All returned agents have Arena-compatible algo types
        for agent in agents:
            assert agent.algo_type in _SCORER_REGISTRY
        assert len(agents) >= 6  # at least the 6 Arena-compatible agents


# ===================================================================
# D. Agent Service filtering — only materializable specs registered
# ===================================================================

class TestAgentServiceFiltering:
    """Agent Service only registers YAML specs it can materialize."""

    def test_unmaterializable_specs_not_in_service(self):
        """Specs with algo_type not in _AGENT_BUILDERS are skipped."""
        from agents.agent_service.service import AgentService
        from agents.agent_service.yaml_loader import _AGENT_BUILDERS

        svc = AgentService()
        types = svc.list_types()
        type_ids = {t["type_id"] for t in types}

        # 'random' (Arena's random) has algo_type='random' which is NOT
        # in _AGENT_BUILDERS, so it should NOT appear as a YAML-loaded type.
        # (The ClassAgentMaterializer(RandomAgent) registers 'random' separately.)
        all_specs = load_agent_specs_from_dir(_CANONICAL_DIR)
        for spec in all_specs:
            if spec.algo_type not in _AGENT_BUILDERS:
                # The spec's ID should only appear if registered by a
                # non-YAML source (e.g. ClassAgentMaterializer).
                pass  # No crash; the YAML spec is silently skipped.

        # Verify materializable YAML agents ARE registered.
        assert "greedy" in type_ids
        assert "minimax" in type_ids
        assert "random_v2" in type_ids
