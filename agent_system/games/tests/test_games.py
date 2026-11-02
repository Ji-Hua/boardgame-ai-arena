import copy
import random
import unittest
from agent_system.games.contracts import QUORIDOR, KUHN_POKER, stable_hash
from agent_system.games.registry import create_game, validate_ref
from agent_system.games.match import RandomAgent, play, replay

class GamesTests(unittest.TestCase):
    def test_unknown_rules_are_rejected(self):
        for ref in ({}, dict(QUORIDOR.wire(), version=2), dict(QUORIDOR.wire(), version=True)):
            with self.assertRaises(ValueError):
                validate_ref(ref)

    def test_quoridor_matches_rust_for_legal_trajectory(self):
        from quoridor_engine import RuleEngine
        from agent_system.training.dqn.action_space import decode_action_id, legal_action_ids
        from backend.adapters.engine_adapter import _serialize_state
        game, engine, rng = create_game(QUORIDOR.wire()), RuleEngine.standard(), random.Random(87)
        state, direct = game.initial_state(), engine.initial_state()
        for _ in range(100):
            self.assertEqual(game.serialize(state), _serialize_state(direct))
            self.assertEqual(game.serialize(game.restore(game.serialize(state))), game.serialize(state))
            legal = tuple(sorted(legal_action_ids(engine, direct))) if not engine.is_game_over(direct) else ()
            self.assertEqual(game.legal_actions(state), legal)
            if not legal:
                break
            action = rng.choice(legal)
            state = game.apply(state, action)
            direct = engine.apply_action(direct, decode_action_id(action, direct.current_player))

    def test_quoridor_terminal_and_replay(self):
        from quoridor_engine import Player
        game = create_game(QUORIDOR.wire())
        state = game.initial_state()
        # Engine-produced terminal trajectory, no reimplementation of rules.
        for _ in range(80):
            if game.actor(state) is None: break
            target = 8 if state.current_player == Player.P1 else 0
            moves = [a for a in game.engine.legal_actions(state) if a.kind == "MovePawn"]
            action = min(moves, key=lambda a: abs(a.target_y-target))
            state = game.engine.apply_action(state, action)
        self.assertIsNone(game.actor(state))
        self.assertEqual(sorted(game.returns(state).values()), [-1.0, 1.0])
        record = play(game, {r: RandomAgent() for r in game.roles}, max_decisions=10)
        self.assertEqual(record["termination"], "truncated")
        self.assertEqual(replay(game, record, "p1"), game.observe(game.restore(record["final"]), "p1"))
        broken = copy.deepcopy(record); broken["events"][0]["action"] = -1
        with self.assertRaises(ValueError): replay(game, broken, "p1")

    def test_kuhn_private_observations_and_replay(self):
        game = create_game(KUHN_POKER.wire())
        a = game.apply(game.apply(game.initial_state(), 0), 1)
        b = game.apply(game.apply(game.initial_state(), 0), 2)
        self.assertEqual(game.observe(a, "p1"), game.observe(b, "p1"))
        self.assertEqual(game.legal_actions(a), game.legal_actions(b))
        self.assertNotEqual(game.observe(a, "p2"), game.observe(b, "p2"))
        agents = lambda: {r: RandomAgent() for r in game.roles}
        first = play(game, agents(), environment_seed=93)
        self.assertEqual(first, play(game, agents(), environment_seed=93))
        self.assertEqual(first["termination"], "engine_terminal")
        self.assertEqual(sum(first["returns"].values()), 0)
        for role in game.roles:
            self.assertEqual(replay(game, first, role), game.observe(game.restore(first["final"]), role))
        self.assertNotIn("environment_seed", replay(game, first, "p1"))

    def test_agent_failures_are_not_game_outcomes(self):
        game = create_game(KUHN_POKER.wire())
        class Invalid(RandomAgent):
            def act(self, decision): return -10
        result = play(game, {"p1": Invalid(), "p2": RandomAgent()})
        self.assertEqual(result["termination"], "illegal_action")
        self.assertEqual(result["returns"], {})
        self.assertEqual(result["offender"], "p1")

    def test_repeated_actor_and_agent_rng_do_not_control_chance(self):
        class Sequence:
            ref, roles = KUHN_POKER, ("p1", "p2")
            def initial_state(self): return 0
            def actor(self, s): return ("chance", "p1", "p1", "p2", None)[s]
            def observe(self, s, r): return {"stage": s}
            def legal_actions(self, s): return (0,) if s < 4 else ()
            def chance_outcomes(self, s): return ((0, 1.0),)
            def apply(self, s, a): return s+1
            def returns(self, s): return {"p1": 1, "p2": -1} if s == 4 else {}
            def serialize(self, s): return s
            def restore(self, s): return s
        result = play(Sequence(), {r: RandomAgent() for r in Sequence.roles})
        self.assertEqual([e["actor"] for e in result["events"]], ["chance", "p1", "p1", "p2"])
        self.assertEqual(result["decisions"], 3)
