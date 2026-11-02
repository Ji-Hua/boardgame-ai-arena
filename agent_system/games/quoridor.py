"""Thin adapter over the existing Rust rules and versioned Quoridor codecs."""
from .contracts import QUORIDOR

class Quoridor:
    ref = QUORIDOR
    roles = ("p1", "p2")

    def __init__(self):
        from quoridor_engine import RuleEngine
        self.engine = RuleEngine.standard()

    def initial_state(self):
        return self.engine.initial_state()

    def actor(self, state):
        if self.engine.is_game_over(state):
            return None
        from quoridor_engine import Player
        return "p1" if state.current_player == Player.P1 else "p2"

    def observe(self, state, role):
        if role not in self.roles:
            raise ValueError("Unknown role")
        return self.serialize(state)

    def legal_actions(self, state):
        from agent_system.training.dqn.action_space import legal_action_ids
        return tuple(sorted(legal_action_ids(self.engine, state))) if self.actor(state) else ()

    def apply(self, state, action):
        from agent_system.training.dqn.action_space import decode_action_id
        if type(action) is not int or action not in self.legal_actions(state):
            raise ValueError("Illegal action")
        return self.engine.apply_action(state, decode_action_id(action, state.current_player))

    def chance_outcomes(self, state):
        return ()

    def returns(self, state):
        if not self.engine.is_game_over(state):
            return {}
        from quoridor_engine import Player
        winner = self.engine.winner(state)
        return {"p1": 1.0 if winner == Player.P1 else -1.0,
                "p2": 1.0 if winner == Player.P2 else -1.0}

    def serialize(self, state):
        from backend.adapters.engine_adapter import _serialize_state
        return _serialize_state(state)

    def restore(self, value):
        from agent_system.evaluation.arena.agents.scorers import _build_state
        return _build_state(value)
