"""Tests for the Arena evaluation system."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from arena.models import GameRecord, MatchResult, TournamentResult
from arena.db import init_db, insert_game, fetch_all_games
from arena.aggregator import compute_win_rate_matrix, format_matrix_text
from arena.runner import play_single_game, ArenaRunner
from arena.agents.core import Agent, AgentInstance
from arena.agents.loader import load_agent


# ---------------------------------------------------------------------------
# Agent YAML directory
# ---------------------------------------------------------------------------

_AGENT_DIR = Path(__file__).resolve().parent.parent / "agent_defs"


# ---------------------------------------------------------------------------
# Helpers — load agents from YAML definitions
# ---------------------------------------------------------------------------

def _load_random_agent() -> Agent:
    return load_agent(_AGENT_DIR / "random.yaml")


def _load_greedy_agent() -> Agent:
    return load_agent(_AGENT_DIR / "greedy.yaml")


def _load_minimax_agent(depth: int = 2) -> Agent:
    return load_agent(_AGENT_DIR / f"minimax_d{depth}.yaml")


def _make_instance(agent: Agent, seed: int = 0) -> AgentInstance:
    return AgentInstance(agent, seed=seed)


# ---------------------------------------------------------------------------
# Test: Single game runs to completion
# ---------------------------------------------------------------------------

class TestSingleGame:
    def test_single_game_runs(self):
        """A game between two random agents completes without crashing."""
        agent = _load_random_agent()
        a = _make_instance(agent, seed=100)
        b = _make_instance(agent, seed=101)
        record = play_single_game(a, b, seed=42)

        assert isinstance(record, GameRecord)
        assert record.num_steps > 0
        assert record.winner in (agent.id, None)
        assert record.seed == 42

    def test_single_game_greedy_vs_random(self):
        """Greedy vs Random completes and typically greedy wins."""
        greedy = _load_greedy_agent()
        rand = _load_random_agent()
        a = _make_instance(greedy, seed=200)
        b = _make_instance(rand, seed=201)
        record = play_single_game(a, b, seed=100)

        assert isinstance(record, GameRecord)
        assert record.num_steps > 0


# ---------------------------------------------------------------------------
# Test: Match result counts are consistent
# ---------------------------------------------------------------------------

class TestMatchResultCounts:
    def test_match_result_counts(self):
        """wins_a + wins_b + draws == num_games for a match."""
        runner = ArenaRunner()
        a = _load_random_agent()
        b = _load_random_agent()
        # Give b a distinct id so results can be attributed
        b = Agent(id="random_b", scorer=b.scorer, policy=b.policy,
                  algo_type=b.algo_type, algo_params=b.algo_params,
                  policy_type=b.policy_type, policy_params=b.policy_params)
        num_games = 10

        result = runner.run_match(a, b, num_games, base_seed=42)

        assert isinstance(result, MatchResult)
        assert result.wins_a + result.wins_b + result.draws == num_games


# ---------------------------------------------------------------------------
# Test: Determinism — same seed produces same results
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_determinism_single_game(self):
        """Same seed → same game outcome."""
        agent = _load_random_agent()

        r1 = play_single_game(_make_instance(agent, 10), _make_instance(agent, 11), seed=99)
        r2 = play_single_game(_make_instance(agent, 10), _make_instance(agent, 11), seed=99)

        assert r1.winner == r2.winner
        assert r1.num_steps == r2.num_steps

    def test_determinism_match(self):
        """Same seed → same match result."""
        runner = ArenaRunner()
        a = _load_random_agent()
        b = Agent(id="random_b", scorer=a.scorer, policy=a.policy,
                  algo_type=a.algo_type, algo_params=a.algo_params,
                  policy_type=a.policy_type, policy_params=a.policy_params)

        m1 = runner.run_match(a, b, 10, base_seed=42)
        m2 = runner.run_match(a, b, 10, base_seed=42)

        assert m1.wins_a == m2.wins_a
        assert m1.wins_b == m2.wins_b
        assert m1.draws == m2.draws


# ---------------------------------------------------------------------------
# Test: SQLite storage
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_and_insert(self):
        """Database initializes and records can be inserted and fetched."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = init_db(db_path)

            record = GameRecord(
                agent_a="random", agent_b="greedy",
                winner="greedy", num_steps=30, seed=1,
            )
            insert_game(conn, record)

            games = fetch_all_games(conn)
            assert len(games) == 1
            assert games[0].agent_a == "random"
            assert games[0].winner == "greedy"

            conn.close()


# ---------------------------------------------------------------------------
# Test: Aggregation matrix
# ---------------------------------------------------------------------------

class TestAggregationMatrix:
    def test_aggregation_matrix_shape(self):
        """Win-rate matrix contains one entry per unique pair."""
        games = [
            GameRecord("a", "b", "a", 10, 1),
            GameRecord("a", "b", "b", 12, 2),
            GameRecord("a", "b", "a", 11, 3),
            GameRecord("b", "c", "c", 15, 4),
        ]
        matrix = compute_win_rate_matrix(games)

        assert ("a", "b") in matrix
        assert ("b", "c") in matrix
        assert len(matrix) == 2

    def test_aggregation_values(self):
        """Win rate is computed correctly."""
        games = [
            GameRecord("a", "b", "a", 10, 1),
            GameRecord("a", "b", "a", 10, 2),
            GameRecord("a", "b", "b", 10, 3),
            GameRecord("a", "b", "a", 10, 4),
        ]
        matrix = compute_win_rate_matrix(games)
        assert matrix[("a", "b")] == pytest.approx(0.75)

    def test_format_matrix_text(self):
        """Matrix text output contains agent names."""
        matrix = {("a", "b"): 0.6}
        text = format_matrix_text(matrix, ["a", "b"])
        assert "a" in text
        assert "b" in text
        assert "0.60" in text


# ---------------------------------------------------------------------------
# Test: Agent strength ordering (soft check)
# ---------------------------------------------------------------------------

class TestAgentStrengthOrder:
    @pytest.mark.slow
    def test_greedy_beats_random(self):
        """Greedy should win more than random over many games."""
        runner = ArenaRunner()
        greedy = _load_greedy_agent()
        rand = _load_random_agent()

        result = runner.run_match(greedy, rand, num_games=20, base_seed=42)

        # Greedy should win at least 60% (allow tolerance)
        total = result.wins_a + result.wins_b + result.draws
        win_rate = result.wins_a / total if total > 0 else 0
        assert win_rate >= 0.5, f"Greedy win rate {win_rate:.0%} < 50% vs Random"

    @pytest.mark.slow
    def test_minimax_d2_beats_random(self):
        """Minimax(d=2) should beat random."""
        runner = ArenaRunner()
        mm2 = _load_minimax_agent(depth=2)
        rand = _load_random_agent()

        result = runner.run_match(mm2, rand, num_games=10, base_seed=42)

        total = result.wins_a + result.wins_b + result.draws
        win_rate = result.wins_a / total if total > 0 else 0
        assert win_rate >= 0.5, f"Minimax(d=2) win rate {win_rate:.0%} < 50% vs Random"
