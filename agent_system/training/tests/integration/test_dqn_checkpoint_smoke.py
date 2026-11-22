"""Integration tests for DQN checkpoint agent and evaluator.

These tests do a mini end-to-end flow:
    tiny training → save checkpoint → load checkpoint → agent rollout/evaluation

Tests are grouped as:
    TestDQNCheckpointAgentSmoke  — agent creation, action selection, legal guarantees
    TestEvaluatorSmoke           — EvalResult shape, game count, illegal action count
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from agent_system.training.dqn.action_space import ACTION_COUNT, legal_action_mask
from agent_system.training.dqn.checkpoint import load_checkpoint, save_checkpoint
from agent_system.training.dqn.evaluator import (
    DQNCheckpointAgent,
    EvalResult,
    evaluate_vs_random,
)
from agent_system.training.dqn.model import QNetwork
from agent_system.training.dqn.observation import OBSERVATION_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    from quoridor_engine import RuleEngine
    return RuleEngine.standard()


def _make_network():
    return QNetwork()


def _tiny_training_step(network, optimizer):
    """Apply a single random-gradient update so weights are non-zero."""
    fake_input = torch.randn(4, OBSERVATION_SIZE)
    fake_target = torch.randn(4, ACTION_COUNT)
    loss = (network(fake_input) - fake_target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------------------
# DQNCheckpointAgent — unit-level (smoke, no full game)
# ---------------------------------------------------------------------------

class TestDQNCheckpointAgentSmoke:
    """Tests that do NOT run full games."""

    def test_from_network_creates_agent(self):
        net = _make_network()
        agent = DQNCheckpointAgent(net, checkpoint_id="test-id")
        assert agent.checkpoint_id == "test-id"

    def test_from_checkpoint_factory(self, tmp_path):
        net = _make_network()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net, checkpoint_id="ckpt-001")
        ckpt = load_checkpoint(path)
        agent = DQNCheckpointAgent.from_checkpoint(ckpt)
        assert agent.checkpoint_id == "ckpt-001"

    def test_from_path_factory(self, tmp_path):
        net = _make_network()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net, checkpoint_id="path-ckpt")
        agent = DQNCheckpointAgent.from_path(str(path))
        assert agent.checkpoint_id == "path-ckpt"

    def test_select_action_returns_int(self):
        net = _make_network()
        agent = DQNCheckpointAgent(net)
        engine = _make_engine()
        state = engine.initial_state()
        mask = legal_action_mask(engine, state)
        obs = [0.0] * OBSERVATION_SIZE
        action = agent.select_action(obs, mask)
        assert isinstance(action, int)

    def test_select_action_is_in_legal_range(self):
        net = _make_network()
        agent = DQNCheckpointAgent(net)
        engine = _make_engine()
        state = engine.initial_state()
        mask = legal_action_mask(engine, state)
        obs = [0.0] * OBSERVATION_SIZE
        action = agent.select_action(obs, mask)
        assert 0 <= action < ACTION_COUNT

    def test_select_action_is_legal(self):
        net = _make_network()
        agent = DQNCheckpointAgent(net)
        engine = _make_engine()
        state = engine.initial_state()
        mask = legal_action_mask(engine, state)
        obs = [0.0] * OBSERVATION_SIZE
        action = agent.select_action(obs, mask)
        assert mask[action] is True

    def test_network_attribute_accessible(self):
        net = _make_network()
        agent = DQNCheckpointAgent(net)
        assert agent.network is net

    def test_agent_trained_checkpoint_selects_legal_action(self, tmp_path):
        """Save after mini training, load, select action — must be legal."""
        net = _make_network()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        _tiny_training_step(net, optimizer)

        path = tmp_path / "trained.pt"
        save_checkpoint(path, net)
        agent = DQNCheckpointAgent.from_path(str(path))

        engine = _make_engine()
        state = engine.initial_state()
        mask = legal_action_mask(engine, state)
        obs = [0.0] * OBSERVATION_SIZE
        action = agent.select_action(obs, mask)
        assert mask[action] is True


# ---------------------------------------------------------------------------
# Evaluator — full game runs
# ---------------------------------------------------------------------------

class TestEvaluatorSmoke:
    """Tests that run complete games via evaluate_vs_random."""

    @pytest.fixture()
    def agent(self):
        net = _make_network()
        return DQNCheckpointAgent(net, checkpoint_id="eval-test")

    @pytest.fixture()
    def engine(self):
        return _make_engine()

    def test_eval_returns_eval_result(self, agent, engine):
        rng = random.Random(0)
        result = evaluate_vs_random(agent, engine, num_games=2, rng=rng)
        assert isinstance(result, EvalResult)

    def test_eval_game_count_correct(self, agent, engine):
        rng = random.Random(42)
        result = evaluate_vs_random(agent, engine, num_games=4, rng=rng)
        assert result.num_games == 4

    def test_eval_wins_losses_draws_sum_to_num_games(self, agent, engine):
        rng = random.Random(1)
        result = evaluate_vs_random(agent, engine, num_games=4, rng=rng)
        assert result.wins + result.losses + result.draws == result.num_games

    def test_eval_illegal_action_count_is_zero(self, agent, engine):
        """A properly-masked agent should produce zero illegal actions."""
        rng = random.Random(7)
        result = evaluate_vs_random(agent, engine, num_games=4, rng=rng)
        assert result.illegal_action_count == 0

    def test_eval_win_rate_is_ratio(self, agent, engine):
        rng = random.Random(3)
        result = evaluate_vs_random(agent, engine, num_games=4, rng=rng)
        assert 0.0 <= result.win_rate <= 1.0

    def test_eval_win_rate_equals_wins_over_games(self, agent, engine):
        rng = random.Random(5)
        result = evaluate_vs_random(agent, engine, num_games=4, rng=rng)
        expected = result.wins / result.num_games
        assert abs(result.win_rate - expected) < 1e-9

    def test_eval_game_lengths_count_matches_num_games(self, agent, engine):
        rng = random.Random(9)
        result = evaluate_vs_random(agent, engine, num_games=4, rng=rng)
        assert len(result.game_lengths) == 4

    def test_eval_game_lengths_positive(self, agent, engine):
        rng = random.Random(11)
        result = evaluate_vs_random(agent, engine, num_games=2, rng=rng)
        assert all(g > 0 for g in result.game_lengths)

    def test_eval_avg_game_length_positive(self, agent, engine):
        rng = random.Random(13)
        result = evaluate_vs_random(agent, engine, num_games=2, rng=rng)
        assert result.avg_game_length > 0

    def test_eval_checkpoint_id_preserved(self, agent, engine):
        rng = random.Random(17)
        result = evaluate_vs_random(agent, engine, num_games=2, rng=rng)
        assert result.checkpoint_id == "eval-test"

    def test_eval_opponent_id_preserved(self, agent, engine):
        rng = random.Random(19)
        result = evaluate_vs_random(
            agent, engine, num_games=2, rng=rng, opponent_id="rng_v1"
        )
        assert result.opponent_id == "rng_v1"

    def test_eval_terminates_via_terminal_state(self, agent, engine):
        """Games must end — either terminal or max_steps guard."""
        rng = random.Random(23)
        # Use tiny max_steps to force draw path as fallback
        result = evaluate_vs_random(
            agent, engine, num_games=2, max_steps=5000, rng=rng
        )
        # All games accounted for
        assert result.wins + result.losses + result.draws == result.num_games

    def test_eval_max_steps_guard_produces_draws(self, agent, engine):
        """With max_steps=1, every game must be a draw."""
        rng = random.Random(29)
        result = evaluate_vs_random(
            agent, engine, num_games=4, max_steps=1, rng=rng
        )
        assert result.draws == 4
        assert result.wins == 0
        assert result.losses == 0

    def test_full_flow_save_train_load_evaluate(self, tmp_path, engine):
        """Mini training → save → load → evaluate completes without error."""
        net = _make_network()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        _tiny_training_step(net, optimizer)

        path = tmp_path / "flow_test.pt"
        save_checkpoint(path, net, training_step=10, episode_count=5)

        agent = DQNCheckpointAgent.from_path(str(path))
        rng = random.Random(31)
        result = evaluate_vs_random(agent, engine, num_games=2, rng=rng)

        assert result.num_games == 2
        assert result.illegal_action_count == 0
        assert result.wins + result.losses + result.draws == 2
