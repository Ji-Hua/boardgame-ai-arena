"""Runtime counterpart of Arena RandomScorer plus its declared policy."""
import random
from quoridor_engine import RuleEngine
from agent_system.runtime.service.base_agent import BaseAgent
from agent_system.runtime.service.agents.greedy_agent import _build_state, _engine_action_to_wire


class ConstantScorerAgent(BaseAgent):
    category = "scripted"

    def __init__(self, policy=None, seed=None):
        self.policy = policy
        self.rng = random.Random(seed)
        self.engine = RuleEngine.standard()

    def make_action(self, game_state, legal_actions):
        actions = self.engine.legal_actions(_build_state(game_state))
        scores = [(_engine_action_to_wire(a, game_state["current_player"]), 1.0) for a in actions]
        if self.policy is None:
            return scores[0][0]
        return self.policy.select(scores, self.rng)
