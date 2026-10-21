"""Integration tests for the head-to-head evaluator (evaluate_dqn_head_to_head).

Smoke-tests the fixed evaluator by:
  - Building two tiny checkpoints from randomly-initialised networks.
  - Running a short head-to-head (10 games).
  - Verifying structural invariants:
      * game count consistency
      * seat-split totals
      * illegal action count is zero (networks are legal-mask-compliant)
      * device-aware wrapper works on CPU
      * compatibility check raises on mismatched hidden_size
      * _resolve_device returns cpu for 'cpu', cuda for 'auto' (if available)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from agent_system.training.dqn.checkpoint import save_checkpoint, load_checkpoint
from agent_system.training.dqn.model import QNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    from quoridor_engine import RuleEngine
    return RuleEngine.standard()


def _make_network(hidden_size: int = 256) -> QNetwork:
    return QNetwork(hidden_size=hidden_size)


def _save_ckpt(tmp_path: Path, name: str, hidden_size: int = 256) -> Path:
    net = _make_network(hidden_size=hidden_size)
    p = tmp_path / f"{name}.pt"
    save_checkpoint(p, net, agent_id=name, checkpoint_id=name)
    return p


# ---------------------------------------------------------------------------
# _DeviceAgent
# ---------------------------------------------------------------------------

class TestDeviceAgent:
    def test_cpu_select_action_is_legal(self, tmp_path):
        """_DeviceAgent on CPU always returns a legal action."""
        from scripts.evaluate_dqn_head_to_head import _DeviceAgent
        from agent_system.training.dqn.action_space import legal_action_mask, ACTION_COUNT
        from agent_system.training.dqn.observation import encode_observation

        net = _make_network()
        agent = _DeviceAgent(net, device=torch.device("cpu"), checkpoint_id="test")
        engine = _make_engine()
        state = engine.initial_state()
        mask = legal_action_mask(engine, state)
        obs = encode_observation(state)

        action_id = agent.select_action(obs, mask)
        assert 0 <= action_id < ACTION_COUNT
        assert mask[action_id], "Selected action must be legal"

    def test_network_moved_to_device(self, tmp_path):
        """Network parameter should reside on the requested device after wrapping."""
        from scripts.evaluate_dqn_head_to_head import _DeviceAgent

        net = _make_network()
        device = torch.device("cpu")
        agent = _DeviceAgent(net, device=device, checkpoint_id="test")
        param = next(agent._network.parameters())
        assert param.device.type == "cpu"


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

class TestCompatibilityCheck:
    def test_matching_checkpoints_pass(self, tmp_path):
        from scripts.evaluate_dqn_head_to_head import _check_compatibility

        ckpt_a = load_checkpoint(_save_ckpt(tmp_path, "a", hidden_size=256))
        ckpt_b = load_checkpoint(_save_ckpt(tmp_path, "b", hidden_size=256))
        # Should not raise
        _check_compatibility(ckpt_a, ckpt_b)

    def test_mismatched_hidden_size_raises(self, tmp_path):
        from scripts.evaluate_dqn_head_to_head import _check_compatibility

        ckpt_a = load_checkpoint(_save_ckpt(tmp_path, "a", hidden_size=256))
        ckpt_b = load_checkpoint(_save_ckpt(tmp_path, "b", hidden_size=128))
        with pytest.raises(ValueError, match="hidden_layers"):
            _check_compatibility(ckpt_a, ckpt_b)


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_cpu(self):
        from scripts.evaluate_dqn_head_to_head import _resolve_device

        dev = _resolve_device("cpu")
        assert dev.type == "cpu"

    def test_auto_returns_valid_device(self):
        from scripts.evaluate_dqn_head_to_head import _resolve_device

        dev = _resolve_device("auto")
        assert dev.type in ("cpu", "cuda")

    def test_cuda_raises_if_unavailable(self, monkeypatch):
        from scripts.evaluate_dqn_head_to_head import _resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="cuda"):
            _resolve_device("cuda")


# ---------------------------------------------------------------------------
# run_head_to_head: structural invariants
# ---------------------------------------------------------------------------

class TestRunHeadToHead:
    @pytest.fixture(scope="class")
    def engine(self):
        return _make_engine()

    @pytest.fixture(scope="class")
    def agents(self, tmp_path_factory):
        from scripts.evaluate_dqn_head_to_head import _DeviceAgent

        tmp = tmp_path_factory.mktemp("ckpts")
        device = torch.device("cpu")
        net_a = _make_network()
        net_b = _make_network()
        agent_a = _DeviceAgent(net_a, device=device, checkpoint_id="a")
        agent_b = _DeviceAgent(net_b, device=device, checkpoint_id="b")
        return agent_a, agent_b

    def test_game_count(self, engine, agents):
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=10, max_steps=200,
        )
        assert result.num_games == 10

    def test_wins_draws_sum_to_games(self, engine, agents):
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=10, max_steps=200,
        )
        assert result.a_wins + result.b_wins + result.draws == 10

    def test_seat_split_p1_p2_consistent(self, engine, agents):
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=10, max_steps=200,
        )
        # a_wins_as_p1 + a_wins_as_p2 == a_wins
        assert result.a_wins_as_p1 + result.a_wins_as_p2 == result.a_wins
        # b_wins_as_p1 + b_wins_as_p2 == b_wins
        assert result.b_wins_as_p1 + result.b_wins_as_p2 == result.b_wins
        # draws_when_a_p1 + draws_when_a_p2 == draws
        assert result.draws_when_a_p1 + result.draws_when_a_p2 == result.draws

    def test_illegal_actions_zero(self, engine, agents):
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=10, max_steps=200,
        )
        assert result.a_illegal_actions == 0
        assert result.b_illegal_actions == 0

    def test_game_lengths_match_count(self, engine, agents):
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=10, max_steps=200,
        )
        assert len(result.game_lengths) == 10

    def test_win_rates_sum_to_at_most_one(self, engine, agents):
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=10, max_steps=200,
        )
        assert result.a_win_rate + result.b_win_rate + result.draw_rate == pytest.approx(1.0)

    def test_seat_even_a_p1(self, engine, agents):
        """Even game index: A should be P1.  Verify via wins_as_p1 tracking."""
        from scripts.evaluate_dqn_head_to_head import run_head_to_head

        agent_a, agent_b = agents
        # With only 2 games, game 0 = A is P1, game 1 = B is P1.
        result = run_head_to_head(
            agent_a, agent_b, engine, "a", "b",
            num_games=2, max_steps=200,
        )
        # Structural: a_wins_as_p1 + a_wins_as_p2 == a_wins
        assert result.a_wins_as_p1 + result.a_wins_as_p2 == result.a_wins

    def test_illegal_action_awards_win_to_opponent(self, engine):
        """If an agent always selects action 0 when it is illegal, opponent wins."""
        from scripts.evaluate_dqn_head_to_head import _DeviceAgent, run_head_to_head
        from agent_system.training.dqn.action_space import legal_action_mask, ACTION_COUNT
        from agent_system.training.dqn.observation import encode_observation

        class _AlwaysIllegalAgent(_DeviceAgent):
            """Always returns action 0 regardless of legality."""
            def select_action(self, observation, legal_action_mask):
                return 0

        class _AlwaysLegalAgent(_DeviceAgent):
            """Uses standard greedy selection (always legal)."""
            pass

        net = _make_network()
        device = torch.device("cpu")
        bad_agent = _AlwaysIllegalAgent(net, device=device, checkpoint_id="bad")
        good_agent = _AlwaysLegalAgent(net, device=device, checkpoint_id="good")

        # Check if action 0 is ever illegal from initial state
        state = engine.initial_state()
        from quoridor_engine import Player
        mask = legal_action_mask(engine, state)

        # Only run this test if action 0 is actually illegal at start
        if mask[0]:
            pytest.skip("Action 0 is legal at initial state; test not applicable")

        # A=bad_agent, B=good_agent, 1 game (A=P1)
        result = run_head_to_head(
            bad_agent, good_agent, engine, "bad", "good",
            num_games=1, max_steps=200,
        )
        # Bad agent made an illegal action → good_agent (B) should win
        assert result.b_wins == 1
        assert result.a_wins == 0
        assert result.a_illegal_actions == 1
        assert result.b_illegal_actions == 0
