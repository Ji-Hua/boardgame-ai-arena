"""Online realization of an immutable DQN checkpoint."""
import random
from agent_system.runtime.service.base_agent import BaseAgent
from agent_system.runtime.inference.dqn import DQNInference


class DQNAgent(BaseAgent):
    type_id = "dqn"
    display_name = "DQN Checkpoint"
    category = "ai"

    def __init__(self, checkpoint, sha256, device="cpu", policy=None, seed=None):
        self.inference = DQNInference(checkpoint, sha256, device)
        self.policy = policy
        self.rng = random.Random(seed)

    def make_action(self, game_state, legal_actions):
        scores = self.inference.score(game_state)
        if self.policy is not None:
            return self.policy.select(scores, self.rng)
        return max(scores, key=lambda item: item[1])[0]
