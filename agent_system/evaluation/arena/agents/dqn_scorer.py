"""Arena scorer using the same learned inference as the Agent Service."""
from agent_system.evaluation.arena.agents.core import Scorer
from agent_system.runtime.inference.dqn import DQNInference


class DQNScorer(Scorer):
    def __init__(self, checkpoint, sha256, device="cpu"):
        self.inference = DQNInference(checkpoint, sha256, device)

    def score(self, state):
        return self.inference.score(state)
