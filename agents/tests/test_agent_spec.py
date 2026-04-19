# TEST_CLASSIFICATION: SPECIFIED
"""Tests for the shared Agent Spec model (agents/agent_spec.py).

Covers:
    A. Unit tests for parse_agent_spec()
       - full 6-field input (Agent Service style)
       - legacy 3-field input (Arena style)
       - defaulting behavior for missing fields
    B. Directory loading tests
       - agents/agent_defs/ (canonical YAML source, 11 files)
    C. Structural assertions for every loaded spec
    D. Snapshot assertions for known files
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agent_spec import AgentSpec, load_agent_spec, load_agent_specs_from_dir, parse_agent_spec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CANONICAL_DEFS = _REPO_ROOT / "agents" / "agent_defs"


# ===================================================================
# A. Unit tests for parse_agent_spec()
# ===================================================================

class TestParseAgentSpec:
    """Unit tests for parse_agent_spec()."""

    def test_full_agent_service_style(self) -> None:
        """Full 6-field Agent Service YAML shape parses correctly."""
        data = {
            "id": "minimax_hard",
            "display_name": "Minimax Agent (Hard)",
            "category": "ai",
            "deterministic": True,
            "algo": {"type": "minimax", "params": {"depth": 5}},
            "policy": {"type": "top_k", "k": 1},
        }
        spec = parse_agent_spec(data)

        assert spec.id == "minimax_hard"
        assert spec.algo_type == "minimax"
        assert spec.algo_params == {"depth": 5}
        assert spec.policy_type == "top_k"
        assert spec.policy_params == {"k": 1}
        assert spec.display_name == "Minimax Agent (Hard)"
        assert spec.category == "ai"
        assert spec.deterministic is True

    def test_legacy_arena_style(self) -> None:
        """Legacy 3-field Arena YAML shape parses with defaults."""
        data = {
            "id": "greedy",
            "algo": {"type": "greedy"},
            "policy": {"type": "top_k", "k": 1},
        }
        spec = parse_agent_spec(data)

        assert spec.id == "greedy"
        assert spec.algo_type == "greedy"
        assert spec.algo_params == {}
        assert spec.policy_type == "top_k"
        assert spec.policy_params == {"k": 1}
        # Defaults
        assert spec.display_name == "greedy"  # defaults to id
        assert spec.category == "ai"
        assert spec.deterministic is True

    def test_defaults_for_missing_optional_fields(self) -> None:
        """All optional fields default correctly when absent."""
        data = {
            "id": "test_agent",
            "algo": {"type": "some_algo"},
        }
        spec = parse_agent_spec(data)

        assert spec.id == "test_agent"
        assert spec.algo_type == "some_algo"
        assert spec.algo_params == {}
        assert spec.policy_type == "top_k"
        assert spec.policy_params == {"k": 1}
        assert spec.display_name == "test_agent"
        assert spec.category == "ai"
        assert spec.deterministic is True

    def test_scripted_category(self) -> None:
        """Non-default category is preserved."""
        data = {
            "id": "random_v2",
            "display_name": "Random Agent V2",
            "category": "scripted",
            "deterministic": False,
            "algo": {"type": "random_v2", "params": {"threshold": 0.8}},
            "policy": {"type": "top_k", "k": 1},
        }
        spec = parse_agent_spec(data)

        assert spec.category == "scripted"
        assert spec.deterministic is False
        assert spec.algo_params == {"threshold": 0.8}

    def test_policy_params_exclude_type_key(self) -> None:
        """The 'type' key is excluded from policy_params."""
        data = {
            "id": "topk3_agent",
            "algo": {"type": "greedy"},
            "policy": {"type": "top_k", "k": 3},
        }
        spec = parse_agent_spec(data)

        assert spec.policy_type == "top_k"
        assert spec.policy_params == {"k": 3}
        assert "type" not in spec.policy_params

    def test_missing_algo_params_defaults_to_empty_dict(self) -> None:
        """algo.params absent → algo_params is empty dict."""
        data = {
            "id": "test",
            "algo": {"type": "greedy"},
            "policy": {"type": "top_k", "k": 1},
        }
        spec = parse_agent_spec(data)
        assert spec.algo_params == {}

    def test_empty_display_name_defaults_to_id(self) -> None:
        """Explicit empty display_name defaults to id."""
        data = {
            "id": "my_agent",
            "display_name": "",
            "algo": {"type": "greedy"},
        }
        spec = parse_agent_spec(data)
        assert spec.display_name == "my_agent"


# ===================================================================
# A2. Frozen / immutability tests
# ===================================================================

class TestAgentSpecImmutability:
    """AgentSpec should be frozen."""

    def test_frozen_attribute_raises(self) -> None:
        spec = parse_agent_spec({"id": "x", "algo": {"type": "greedy"}})
        with pytest.raises(AttributeError):
            spec.id = "changed"  # type: ignore[misc]

    def test_defensive_copy_algo_params(self) -> None:
        """Caller's dict mutation must not affect the spec."""
        params = {"depth": 2}
        spec = AgentSpec(id="x", algo_type="minimax", algo_params=params)
        params["depth"] = 99
        assert spec.algo_params == {"depth": 2}

    def test_defensive_copy_policy_params(self) -> None:
        """Caller's dict mutation must not affect the spec."""
        params = {"k": 3}
        spec = AgentSpec(id="x", algo_type="minimax", policy_params=params)
        params["k"] = 99
        assert spec.policy_params == {"k": 3}


# ===================================================================
# B. Directory loading tests
# ===================================================================

class TestLoadCanonicalDir:
    """Load all YAML files from agents/agent_defs/ (canonical source)."""

    @pytest.fixture()
    def specs(self) -> list[AgentSpec]:
        assert _CANONICAL_DEFS.is_dir(), f"Missing: {_CANONICAL_DEFS}"
        return load_agent_specs_from_dir(_CANONICAL_DEFS)

    def test_count(self, specs: list[AgentSpec]) -> None:
        assert len(specs) == 11

    def test_all_ids_present(self, specs: list[AgentSpec]) -> None:
        ids = {s.id for s in specs}
        assert ids == {
            "greedy",
            "greedy_topk3",
            "minimax",
            "minimax_d2",
            "minimax_d2_topk3",
            "minimax_d3",
            "minimax_simple",
            "minimax_medium",
            "minimax_hard",
            "random",
            "random_v2",
        }

    def test_structural_validity(self, specs: list[AgentSpec]) -> None:
        """Every loaded spec satisfies structural assertions."""
        for spec in specs:
            assert isinstance(spec.id, str) and spec.id
            assert isinstance(spec.algo_type, str) and spec.algo_type
            assert isinstance(spec.algo_params, dict)
            assert isinstance(spec.policy_type, str) and spec.policy_type
            assert isinstance(spec.policy_params, dict)
            assert isinstance(spec.display_name, str) and spec.display_name
            assert isinstance(spec.category, str) and spec.category
            assert isinstance(spec.deterministic, bool)


# ===================================================================
# C. Single-file loading
# ===================================================================

class TestLoadAgentSpec:
    """Test load_agent_spec() for individual files."""

    def test_load_single_agent_service_file(self) -> None:
        path = _CANONICAL_DEFS / "greedy.yaml"
        assert path.exists()
        spec = load_agent_spec(path)
        assert spec.id == "greedy"
        assert spec.algo_type == "greedy"

    def test_load_arena_compatible_file(self) -> None:
        """Enriched specs in canonical dir parse with explicit fields."""
        path = _CANONICAL_DEFS / "minimax_d3.yaml"
        assert path.exists()
        spec = load_agent_spec(path)
        assert spec.id == "minimax_d3"
        assert spec.algo_type == "minimax"
        assert spec.display_name == "Minimax Agent (Depth 3)"
        assert spec.category == "ai"


# ===================================================================
# D. Snapshot assertions for known files
# ===================================================================

class TestSnapshotAgentService:
    """Exact snapshot for known Agent Service-style YAML files."""

    def test_minimax_hard(self) -> None:
        spec = load_agent_spec(_CANONICAL_DEFS / "minimax_hard.yaml")
        assert spec == AgentSpec(
            id="minimax_hard",
            algo_type="minimax",
            algo_params={"depth": 5},
            policy_type="top_k",
            policy_params={"k": 1},
            display_name="Minimax Agent (Hard)",
            category="ai",
            deterministic=True,
        )

    def test_random_v2(self) -> None:
        spec = load_agent_spec(_CANONICAL_DEFS / "random_v2.yaml")
        assert spec == AgentSpec(
            id="random_v2",
            algo_type="random_v2",
            algo_params={"threshold": 0.8},
            policy_type="top_k",
            policy_params={"k": 1},
            display_name="Random Agent V2",
            category="scripted",
            deterministic=False,
        )


class TestSnapshotArenaCompatible:
    """Exact snapshot for Arena-origin YAML files now enriched in canonical dir."""

    def test_greedy_topk3(self) -> None:
        spec = load_agent_spec(_CANONICAL_DEFS / "greedy_topk3.yaml")
        assert spec == AgentSpec(
            id="greedy_topk3",
            algo_type="greedy",
            algo_params={},
            policy_type="top_k",
            policy_params={"k": 3},
            display_name="Greedy Agent (Top-K 3)",
            category="ai",
            deterministic=False,
        )

    def test_minimax_d2_topk3(self) -> None:
        spec = load_agent_spec(_CANONICAL_DEFS / "minimax_d2_topk3.yaml")
        assert spec == AgentSpec(
            id="minimax_d2_topk3",
            algo_type="minimax",
            algo_params={"depth": 2},
            policy_type="top_k",
            policy_params={"k": 3},
            display_name="Minimax Agent (Depth 2, Top-K 3)",
            category="ai",
            deterministic=False,
        )

    def test_random(self) -> None:
        spec = load_agent_spec(_CANONICAL_DEFS / "random.yaml")
        assert spec == AgentSpec(
            id="random",
            algo_type="random",
            algo_params={},
            policy_type="top_k",
            policy_params={"k": 1},
            display_name="Random Agent",
            category="scripted",
            deterministic=False,
        )
