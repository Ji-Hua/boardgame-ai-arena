"""Optional OpenSpiel adapter. Only explicitly registered two-player games."""
from .contracts import KUHN_POKER

class KuhnPoker:
    ref = KUHN_POKER
    roles = ("p1", "p2")

    def __init__(self):
        try:
            import pyspiel
        except ImportError as exc:
            raise RuntimeError("Kuhn Poker requires the optional open_spiel package") from exc
        self.game = pyspiel.load_game("kuhn_poker")

    def initial_state(self):
        return self.game.new_initial_state()

    def actor(self, state):
        if state.is_terminal():
            return None
        return "chance" if state.is_chance_node() else self.roles[state.current_player()]

    def observe(self, state, role):
        if role not in self.roles:
            raise ValueError("Unknown role")
        player = self.roles.index(role)
        # Information state includes the player's visible history, never the other card.
        return {"information_state": state.information_state_string(player),
                "tensor": list(state.information_state_tensor(player))}

    def legal_actions(self, state):
        return tuple(state.legal_actions()) if not state.is_terminal() else ()

    def apply(self, state, action):
        if type(action) is not int or action not in self.legal_actions(state):
            raise ValueError("Illegal action")
        result = state.clone()
        result.apply_action(action)
        return result

    def chance_outcomes(self, state):
        return tuple(state.chance_outcomes()) if state.is_chance_node() else ()

    def returns(self, state):
        return dict(zip(self.roles, state.returns())) if state.is_terminal() else {}

    def serialize(self, state):
        return state.serialize()

    def restore(self, value):
        return self.game.deserialize_state(value)

    def public_observe(self, state):
        return {"betting_history": list(state.history()[2:]),
                "actor": self.actor(state), "terminal": state.is_terminal()}

    def action_label(self, state, action):
        return state.action_to_string(action)
