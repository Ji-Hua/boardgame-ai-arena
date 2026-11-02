"""Local research match/replay. Trusted full-state records are NOT public DTOs."""
from __future__ import annotations
import copy
import math
import random
from .contracts import Decision, stable_hash

class RandomAgent:
    def reset(self, seed):
        self.rng = random.Random(seed)

    def act(self, decision):
        return self.rng.choice(decision.legal_actions)

def play(game, agents, *, environment_seed=0, agent_seeds=None, max_decisions=500, max_events=10000):
    if set(agents) != set(game.roles) or len({id(a) for a in agents.values()}) != len(agents):
        raise ValueError("Provide a separate agent instance for each role")
    if max_decisions < 1 or max_events < 1:
        raise ValueError("Positive decision and event limits required")
    rng = random.Random(environment_seed)
    seeds = agent_seeds if agent_seeds is not None else {role: i for i, role in enumerate(game.roles)}
    if set(seeds) != set(game.roles):
        raise ValueError("One agent seed per role required")
    for role, agent in agents.items():
        agent.reset(seeds[role])
    state = game.initial_state()
    initial = game.serialize(state)
    events, decisions = [], 0
    termination, offender = "engine_terminal", None
    while game.actor(state) is not None:
        if decisions >= max_decisions or len(events) >= max_events:
            termination = "truncated"
            break
        role = game.actor(state)
        if role == "chance":
            outcomes = game.chance_outcomes(state)
            if not outcomes or any(not math.isfinite(p) or p < 0 for _, p in outcomes) or not math.isclose(sum(p for _, p in outcomes), 1.0):
                raise ValueError("Invalid chance distribution")
            action = rng.choices([a for a, _ in outcomes], weights=[p for _, p in outcomes])[0]
        else:
            if role not in agents:
                raise ValueError("Engine returned unknown role")
            request = Decision(role, copy.deepcopy(game.observe(state, role)),
                               tuple(game.legal_actions(state)), decisions)
            try:
                action = agents[role].act(request)
            except Exception:
                termination, offender = "agent_error", role
                break
            if type(action) is not int or action not in request.legal_actions:
                termination, offender = "illegal_action", role
                break
            decisions += 1
        # Engine failures propagate; they are not misreported as agent forfeits.
        state = game.apply(state, action)
        events.append({"actor": role, "action": action})
    record = {"schema_version": 2, "visibility": "research_private", "game_ref": game.ref.wire(),
              "initial": initial, "events": events, "termination": termination, "offender": offender,
              "returns": game.returns(state), "decisions": decisions,
              "environment_seed": environment_seed, "agent_seeds": seeds,
              "final": game.serialize(state)}
    record["digest"] = stable_hash(record)
    return record

def replay(game, record, viewer):
    """Reconstruct one player's final view; never return internal initial state, seed or digest."""
    if viewer not in game.roles or record["game_ref"] != game.ref.wire():
        raise ValueError("Incompatible game or viewer")
    if record.get("schema_version") != 2 or record.get("visibility") != "research_private":
        raise ValueError("Unsupported replay")
    if stable_hash({k: v for k, v in record.items() if k != "digest"}) != record.get("digest"):
        raise ValueError("Replay integrity mismatch")
    state = game.restore(record["initial"])
    for event in record["events"]:
        if game.actor(state) != event["actor"]:
            raise ValueError("Replay actor mismatch")
        state = game.apply(state, event["action"])
    if game.serialize(state) != record["final"] or game.returns(state) != record["returns"]:
        raise ValueError("Replay result mismatch")
    return copy.deepcopy(game.observe(state, viewer))
