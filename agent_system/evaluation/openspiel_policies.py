"""Versioned paired-seat evaluation of immutable Kuhn Poker policies."""
import hashlib
import json
from pathlib import Path
import re
import statistics
from agent_system.games.contracts import KUHN_POKER, Decision, stable_hash
from agent_system.games.match import RandomAgent, play, replay
from agent_system.training.openspiel_job import PolicyAgent

PROTOCOL = "kuhn-policy-paired-v1"

def validate_config(config):
    if not isinstance(config, dict) or set(config) - {"candidate_sha", "opponent_sha", "pairs", "seed"}:
        raise ValueError("Expected candidate_sha, opponent_sha, pairs and seed")
    value = dict(opponent_sha=None, pairs=100, seed=42)
    value.update(config)
    for key in ("candidate_sha", "opponent_sha"):
        sha = value.get(key)
        if key == "opponent_sha" and sha is None:
            continue
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(key + " must be a full policy SHA-256")
    for key, lo, hi in (("pairs", 1, 1000), ("seed", 0, 2**32-1)):
        if type(value[key]) is not int or not lo <= value[key] <= hi:
            raise ValueError(f"{key} must be within [{lo}, {hi}]")
    return value

def load_policy(root, sha, project_id, game_ref):
    path = Path(root)/"policies"/(sha+".json")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("Policy SHA mismatch")
    bundle = json.loads(data)
    if bundle.get("project_id") != project_id or bundle.get("game_ref") != game_ref:
        raise ValueError("Policy project or game mismatch")
    return PolicyAgent(bundle)

def run_evaluation(job, run, events):
    from agent_system.games.openspiel import KuhnPoker
    from open_spiel.python import policy
    from open_spiel.python.algorithms import exploitability
    if job.get("game_ref") != KUHN_POKER.wire():
        raise ValueError("Unsupported evaluation game")
    config = validate_config(json.loads(job["config"]) if isinstance(job["config"], str) else job["config"])
    game = KuhnPoker()
    candidate = load_policy(job["data_root"], config["candidate_sha"], job["project_id"], job["game_ref"])
    opponent = (load_policy(job["data_root"], config["opponent_sha"], job["project_id"], job["game_ref"])
                if config["opponent_sha"] else RandomAgent())
    # Traverse information states through OpenSpiel, checking complete policy coverage.
    tabular = policy.TabularPolicy(game.game)
    expected = set()
    for index, state in enumerate(tabular.states):
        role = game.roles[state.current_player()]
        request = Decision(role, game.observe(state, role), game.legal_actions(state), 0)
        probs = candidate.probabilities(request)
        expected.add((role, request.observation["information_state"]))
        for action, probability in zip(request.legal_actions, probs):
            tabular.action_probability_array[index, action] = probability
    if expected != set(candidate.rows):
        raise ValueError("Policy information-state coverage mismatch")
    exact = exploitability.exploitability(game.game, tabular)
    results, by_role = [], {role: [] for role in game.roles}
    with (run/"research-metrics.jsonl").open("x") as metrics, (run/"private-replays.jsonl").open("x") as replays:
        for pair in range(config["pairs"]):
            scores = []
            for seat in game.roles:
                agents = {r: candidate if r == seat else opponent for r in game.roles}
                record = play(game, agents, environment_seed=config["seed"]+pair,
                    agent_seeds={r: config["seed"]+2*pair+(1 if r==seat else 2) for r in game.roles})
                if record["termination"] != "engine_terminal":
                    raise ValueError("Policy failed evaluation: "+record["termination"])
                for viewer in game.roles:
                    replay(game, record, viewer)
                replays.write(json.dumps(dict(pair=pair, candidate_role=seat, replay=record))+"\n")
                score = record["returns"][seat]
                scores.append(score); by_role[seat].append(score)
            results.append(statistics.mean(scores))
            metrics.write(json.dumps(dict(type="train", step=pair+1, progress_unit="evaluation_pair",
                mean_return=statistics.mean(results), exploitability=exact))+"\n")
            metrics.flush()
    report = dict(schema_version=1, protocol=PROTOCOL, config=config, config_sha256=stable_hash(config),
        game_ref=job["game_ref"], project_id=job["project_id"], source=job["source"],
        image=job["image"], games=2*config["pairs"], mean_return=statistics.mean(results),
        standard_error=statistics.stdev(results)/(len(results)**0.5) if len(results)>1 else None,
        role_returns={r:statistics.mean(values) for r,values in by_role.items()},
        exploitability=exact, nash_conv=2*exact)
    report["digest"] = stable_hash(report)
    (run/"research-evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    events.emit("research_evaluation", protocol=PROTOCOL, games=report["games"], mean_return=report["mean_return"])
    return []
