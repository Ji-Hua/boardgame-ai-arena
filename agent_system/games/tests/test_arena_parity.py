"""Frozen evidence captured from unmodified matrix.py at 772e88f, seed 42."""
import unittest
from agent_system.evaluation.arena.matrix import play_block, standard_opening

class ArenaParityTests(unittest.TestCase):
    def test_existing_arena_keeps_trajectory_and_protocol(self):
        a = dict(id="a", algo=dict(type="random"), policy=dict(type="top_k", k=1))
        b = dict(a, id="b")
        block = play_block(a, b, standard_opening(), max_plies=20, seed=42)
        for game in block["games"]:
            self.assertEqual(game["trajectory_hash"], "efb3b025845d3c334301bb75073cf11bb9651662edfe767e500b0651b3bd139c")
            self.assertEqual((game["termination"], game["steps"]), ("ply_limit", 20))
            self.assertIsNone(game["offender"])
