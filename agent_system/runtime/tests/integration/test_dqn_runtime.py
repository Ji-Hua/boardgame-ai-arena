# TEST_CLASSIFICATION: SPECIFIED
import hashlib
import pytest
from quoridor_engine import RuleEngine, Player
from agent_system.training.dqn.checkpoint import save_checkpoint
from agent_system.training.dqn.model import CNNQNetwork, QNetwork
from agent_system.training.dqn.action_space import encode_engine_action
from agent_system.runtime.inference.dqn import DQNInference
from agent_system.evaluation.arena.agents.scorers import _engine_action_to_wire


def wire_state(s):
    return {
        "current_player": 1 if s.current_player == Player.P1 else 2,
        "pawns": {str(i): dict(zip(("row", "col"), s.pawn_pos(p)))
                  for i, p in [(1, Player.P1), (2, Player.P2)]},
        "walls_remaining": {str(i): s.walls_remaining(p) for i, p in [(1, Player.P1), (2, Player.P2)]},
        "wall_state": {k: getattr(s, k) for k in
                       ["horizontal_edges", "vertical_edges", "horizontal_heads", "vertical_heads"]},
    }


@pytest.mark.parametrize("arch,model", [("cnn", CNNQNetwork), ("mlp", QNetwork)])
def test_both_seats_score_exactly_all_legal_actions(tmp_path, arch, model):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model(), model_arch=arch)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    inference = DQNInference(str(path), sha)
    engine = RuleEngine.standard()
    state = engine.initial_state()
    for _ in range(4):
        legal = sorted(engine.legal_actions(state), key=encode_engine_action)
        wire = wire_state(state)
        scores = inference.score(wire)
        assert [a for a, _ in scores] == [_engine_action_to_wire(a, wire["current_player"]) for a in legal]
        chosen = max(scores, key=lambda x: x[1])[0]
        matched = next(a for a in legal if _engine_action_to_wire(a, wire["current_player"]) == chosen)
        state = engine.apply_action(state, matched)


def test_checkpoint_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, QNetwork())
    with pytest.raises(ValueError, match="SHA256"):
        DQNInference(str(path), "0" * 64)


@pytest.mark.parametrize("arch,model", [("cnn", CNNQNetwork), ("mlp", QNetwork)])
def test_online_lifecycle_matches_arena_on_both_seats(tmp_path, arch, model):
    import yaml
    from agent_system.runtime.service.service import AgentService
    from agent_system.definition.agent_spec import parse_agent_spec
    from agent_system.evaluation.arena.agents.loader import materialize_agent
    from agent_system.evaluation.arena.agents.core import AgentInstance
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model(), model_arch=arch)
    spec = {"id": "test_model", "algo": {"type": "dqn", "params": {
        "checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }}, "policy": {"type": "top_k", "k": 1}}
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(spec))
    service = AgentService(agent_defs_dir=tmp_path)
    arena = AgentInstance(materialize_agent(parse_agent_spec(spec)), seed=42)
    for seat in [1, 2]:
        instance_id = service.create_agent("test_model", "room", seat)
        service.start_agent(instance_id)
    engine = RuleEngine.standard()
    state = engine.initial_state()
    for _ in range(6):
        wire = wire_state(state)
        online = service.get_action("room", wire["current_player"], wire, [])
        assert online == arena.act(wire)
        action = next(a for a in engine.legal_actions(state)
                      if _engine_action_to_wire(a, wire["current_player"]) == online)
        state = engine.apply_action(state, action)
    service.destroy_room_agents("room")
    assert not service.has_agent("room", 1)


def test_service_preserves_yaml_depth_before_explicit_overrides(tmp_path):
    import yaml
    from agent_system.runtime.service.service import AgentService
    spec = {"id": "depth_one", "algo": {"type": "minimax", "params": {"depth": 1}},
            "policy": {"type": "top_k", "k": 1}}
    (tmp_path / "depth.yaml").write_text(yaml.safe_dump(spec))
    service = AgentService(agent_defs_dir=tmp_path)
    first = service.create_agent("depth_one", "room", 1)
    second = service.create_agent("depth_one", "room", 2, config={"depth": 3})
    assert service._instance_mgr.get(first).agent._depth == 1
    assert service._instance_mgr.get(second).agent._depth == 3


def test_uniform_random_service_matches_canonical_seeded_policy():
    from agent_system.definition.agent_spec import parse_agent_spec
    from agent_system.evaluation.arena.agents.loader import materialize_agent
    from agent_system.evaluation.arena.agents.core import AgentInstance
    from agent_system.runtime.service.yaml_loader import parse_agent_definition, create_agent_from_definition
    spec = {"id": "random_legal", "algo": {"type": "random"},
            "policy": {"type": "top_k", "k": 209}, "deterministic": False}
    runtime = create_agent_from_definition(parse_agent_definition(spec), context={"seed": 42})
    arena = AgentInstance(materialize_agent(parse_agent_spec(spec)), seed=42)
    state = wire_state(RuleEngine.standard().initial_state())
    assert [runtime.make_action(state, []) for _ in range(20)] == [arena.act(state) for _ in range(20)]
