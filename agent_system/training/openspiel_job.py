"""Small OpenSpiel reference workloads; all rules and learning updates are upstream."""
import importlib.metadata
import json
from pathlib import Path
import random

from agent_system.games.contracts import KUHN_POKER, stable_hash

DEFAULT_CONFIG = dict(algorithm="cfr", iterations=200, eval_every=20, seed=42)

def validate_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("Expected algorithm, iterations, eval_every and seed only")
    result = {**DEFAULT_CONFIG, **config}
    if result["algorithm"] not in ("cfr", "tabular_q_learning"):
        raise ValueError("Supported algorithms: cfr, tabular_q_learning")
    for key, high in (("iterations", 100000), ("eval_every", 100000), ("seed", 2**32 - 1)):
        low = 0 if key == "seed" else 1
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    return result

def run_openspiel(job, run, events):
    import numpy as np
    from open_spiel.python import policy, rl_environment, rl_tools
    from open_spiel.python.algorithms import cfr, exploitability, tabular_qlearner
    from agent_system.games.openspiel import KuhnPoker

    if job.get("game_ref") != KUHN_POKER.wire():
        raise ValueError("OpenSpiel reference worker requires the exact Kuhn Poker GameRef")
    raw = json.loads(job["config"]) if isinstance(job["config"], str) else job["config"]
    config = validate_config(raw)
    adapter = KuhnPoker()
    game = adapter.game
    np.random.seed(config["seed"])
    if config["algorithm"] == "cfr":
        solver = cfr.CFRSolver(game)
        def update():
            solver.evaluate_and_update_policy()
        def current_policy():
            return solver.average_policy()
        unit = "solver_iteration"
    else:
        env = rl_environment.Environment(game,
            chance_event_sampler=rl_environment.ChanceEventSampler(seed=config["seed"]),
            observation_type=rl_environment.ObservationType.INFORMATION_STATE,
            enable_legality_check=True)
        learners = [tabular_qlearner.QLearner(i, game.num_distinct_actions(),
                    epsilon_schedule=rl_tools.ConstantSchedule(0.2)) for i in range(2)]
        def update():
            step = env.reset()
            while not step.last():
                role = step.observations["current_player"]
                step = env.step([learners[role].step(step).action])
            for learner in learners:
                learner.step(step)  # terminal rewards delivered to both roles
        def current_policy():
            # Upstream evaluation samples an action even when only probabilities are needed.
            # Preserve training RNG so evaluation frequency cannot change the learned policy.
            saved = np.random.get_state()
            def probabilities(state):
                role = state.current_player()
                infos, legal = [[], []], [[], []]
                infos[role] = state.information_state_tensor(role)
                legal[role] = state.legal_actions()
                step = rl_environment.TimeStep(dict(info_state=infos, legal_actions=legal),
                    None, None, rl_environment.StepType.MID)
                result = learners[role].step(step, is_evaluation=True)
                return enumerate(result.probs)
            try:
                return policy.tabular_policy_from_callable(game, probabilities)
            finally:
                np.random.set_state(saved)
        unit = "episode"
    provenance = dict(schema_version=1, game_ref=job["game_ref"],
        project_id=job["project_id"], source=job["source"], image=job.get("image"),
        config=config, config_sha256=stable_hash(config), progress_unit=unit,
        package_versions={"open_spiel": importlib.metadata.version("open_spiel"),
                          "numpy": np.__version__},
        output_semantics="action_probabilities",
        policy_kind="average_policy" if config["algorithm"] == "cfr" else "greedy_q_policy")
    (run / "research-config.json").write_text(json.dumps(provenance, indent=2))
    def evaluate(step, stream):
        current = current_policy()
        value = exploitability.exploitability(game, current)
        row = dict(type="train", step=step, exploitability=value,
                   nash_conv=2 * value, progress_unit=unit)
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        events.emit("evaluation", **{k:v for k,v in row.items() if k != "type"})
        return current
    with (run / "research-metrics.jsonl").open("x") as stream:
        current = evaluate(0, stream)
        for step in range(1, config["iterations"] + 1):
            update()
            if step % config["eval_every"] == 0 or step == config["iterations"]:
                current = evaluate(step, stream)
    # Information-state strings and legal action probabilities, never a true-state policy.
    rows = []
    for state in current.states:
        role = adapter.roles[state.current_player()]
        rows.append(dict(role=role, information_state=state.information_state_string(),
                         probabilities={str(a): float(p) for a,p in current.action_probabilities(state).items()}))
    bundle = dict(**provenance, policies=rows)
    bundle["digest"] = stable_hash(bundle)
    artifact = run / "policy.json"
    artifact.write_text(json.dumps(bundle, sort_keys=True, indent=2, allow_nan=False))
    # Reload the serialized artifact and exercise both seats through the common match runner.
    from agent_system.games.match import RandomAgent, play, replay
    results = []
    for seat in adapter.roles:
        agents = {role: PolicyAgent.load(artifact) if role == seat else RandomAgent() for role in adapter.roles}
        record = play(adapter, agents, environment_seed=config["seed"],
                      agent_seeds={"p1": config["seed"] + 1, "p2": config["seed"] + 2})
        if record["termination"] != "engine_terminal":
            raise RuntimeError("Exported policy failed adapter round-trip")
        for viewer in adapter.roles:
            replay(adapter, record, viewer)
        results.append(dict(policy_role=seat, returns=record["returns"], termination=record["termination"]))
    (run / "policy-smoke.json").write_text(json.dumps(results, indent=2))
    from agent_system.training.jobs import sha256
    manifest = dict(schema_version=1, artifacts=[dict(file="policy.json", sha256=sha256(artifact),
        kind="openspiel_tabular_policy", output_semantics="action_probabilities",
        game_ref=job["game_ref"])])
    (run / "research-artifacts.json").write_text(json.dumps(manifest, indent=2))
    events.emit("policy_exported", **manifest["artifacts"][0])
    return []  # Never register JSON as a legacy .pt checkpoint.

class PolicyAgent:
    def __init__(self, bundle):
        if bundle.get("game_ref") != KUHN_POKER.wire() or bundle.get("schema_version") != 1:
            raise ValueError("Incompatible policy game or schema")
        if bundle.get("output_semantics") != "action_probabilities":
            raise ValueError("Invalid policy semantics")
        if stable_hash({k:v for k,v in bundle.items() if k != "digest"}) != bundle.get("digest"):
            raise ValueError("Policy integrity mismatch")
        self.rows = {(r["role"], r["information_state"]): r["probabilities"] for r in bundle["policies"]}
        if len(self.rows) != len(bundle["policies"]):
            raise ValueError("Duplicate policy information state")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def reset(self, seed):
        self.rng = random.Random(seed)

    def act(self, decision):
        import math
        probs = self.rows[(decision.role, decision.observation["information_state"])]
        if set(probs) != {str(a) for a in decision.legal_actions}:
            raise ValueError("Policy action set mismatch")
        weights = [probs[str(a)] for a in decision.legal_actions]
        if any(not math.isfinite(p) or p < 0 for p in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("Invalid policy distribution")
        return self.rng.choices(decision.legal_actions, weights=weights)[0]
