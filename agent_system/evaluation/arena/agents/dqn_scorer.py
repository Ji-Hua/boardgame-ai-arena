"""Arena scorer using the same learned inference as the Agent Service."""
from agent_system.evaluation.arena.agents.core import Scorer
from agent_system.runtime.inference.dqn import DQNInference


class DQNScorer(Scorer):
    def __init__(self, checkpoint, sha256, device="cpu"):
        self.inference = DQNInference(checkpoint, sha256, device)
        self.last_stats = {}

    def score(self, state):
        result = self.inference.score(state)
        self.last_stats = {"model_evaluations": 1, "forward_batches": 1, "simulations": 0}
        return result
