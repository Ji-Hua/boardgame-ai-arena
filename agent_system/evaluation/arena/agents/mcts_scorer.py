"""Checkpoint-pinned PUCT adapter; evaluation is deterministic and memoryless."""
from agent_system.evaluation.arena.agents.core import Scorer
from agent_system.runtime.inference.dqn import DQNInference
from agent_system.runtime.search.puct import SearchConfig, TorchEvaluator, search_batch
from agent_system.runtime.service.agents.greedy_agent import _build_state, _engine_action_to_wire


class MCTSScorer(Scorer):
    def __init__(self, checkpoint, sha256, device="cpu", simulations=50, c_puct=1.5, max_depth=128):
        self.inference = DQNInference(checkpoint, sha256, device)
        if self.inference.metadata["eval_summary"].get("value_head_status") not in {"calibrated", "trained"}:
            raise ValueError("MCTS requires outcome calibration; random value heads are not supported")
        self.config = SearchConfig(simulations=simulations, c_puct=c_puct, max_depth=max_depth)
        self.evaluator = TorchEvaluator(self.inference.network, device)
        self.last_stats, self.last_result = {}, None

    def score(self, state):
        before = self.evaluator.stats.copy()
        result = search_batch([_build_state(state)], self.evaluator, self.config)[0]
        self.last_result = result
        self.last_stats = {**result.stats, "forward_batches": self.evaluator.stats["forward_batches"] - before["forward_batches"]}
        root = result.root
        self.last_stats["search_changed_policy_argmax"] = int(result.action_id != int(root.ids[root.prior.argmax()]))
        # TopKPolicy is stable: visit ties resolve by prior, then action index.
        return [(_engine_action_to_wire(root.actions[i], state["current_player"]), float(root.visits[i]))
                for i in result.ranked_indices]
