"""A hierarchy must change selection, masking, exploration and DDQN targets."""
import hashlib
import json
import random

import pytest
import torch

from agent_system.analysis import neural
from agent_system.analysis.state import ENGINE, Position, Move, build_state
from agent_system.runtime.inference.dqn import DQNInference
from agent_system.training.dqn.action_space import legal_action_mask
from agent_system.training.dqn.checkpoint import load_checkpoint, save_checkpoint
from agent_system.training.dqn.evaluator import DQNCheckpointAgent
from agent_system.training.dqn.model import CNNQNetwork, DQNPolicy, policy_argmax, select_policy_action
from agent_system.training.dqn.observation_cnn import encode_observation_cnn
from agent_system.training.dqn.trainer import train_step


def controlled_network():
    net = CNNQNetwork(cnn_channels=[2, 2], hierarchical=True, activation="leaky_relu")
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.head[-1].bias[37] = 1  # e2
        net.head[-1].bias[81] = 100  # a1h: flat argmax would choose this.
        net.action_type_head.bias.copy_(torch.tensor([2., 1.]))
    return net


def test_route_precedes_action_argmax_and_excludes_unavailable_types():
    q = torch.zeros(209); q[37] = 1; q[81] = 100
    mask = torch.zeros(209, dtype=torch.bool); mask[[27, 37, 45, 81]] = True
    assert policy_argmax(q, mask).item() == 81
    assert policy_argmax(q, mask, torch.tensor([2., 1.])).item() == 37
    mask[81] = False
    assert policy_argmax(q, mask, torch.tensor([-10., 1000.])).item() == 37


def test_exploration_is_balanced_by_type_not_action_count():
    mask = [i in (27, 37, 45) or i >= 81 for i in range(209)]
    rng = random.Random(91)
    q, kind = torch.zeros(209), torch.zeros(2)
    actions = [select_policy_action(q, mask, type_q=kind, epsilon=1., rng=rng) for _ in range(6000)]
    assert all(mask[a] for a in actions)
    assert 2800 < sum(a >= 81 for a in actions) < 3200


def test_checkpoint_runtime_evaluator_and_analysis_share_the_hierarchy(tmp_path, monkeypatch):
    net = controlled_network()
    path = tmp_path / "hierarchical.pt"
    save_checkpoint(path, net, model_arch="cnn")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {"id": "hierarchical", "checkpoint": str(path), "sha256": digest,
             "metadata": {"model_config": {"model_arch": "cnn"}}}
    catalog = tmp_path / "catalog.json"; catalog.write_text(json.dumps({"models": [entry]}))
    monkeypatch.setenv("LAB_MODEL_CATALOG", str(catalog))
    neural.cached_runner.cache_clear()
    restored = load_checkpoint(path)
    state = ENGINE.initial_state()
    obs, mask = encode_observation_cnn(state), legal_action_mask(ENGINE, state)
    assert restored.network.hierarchical is True
    assert DQNPolicy(restored.network).select_action(obs, mask) == 37
    assert DQNCheckpointAgent.from_checkpoint(restored).select_action(obs, mask) == 37
    wire = {"current_player": 1, "pawns": {"1": {"row": 4, "col": 0}, "2": {"row": 4, "col": 8}},
            "walls_remaining": {"1": 10, "2": 10}, "wall_state": {}}
    scores = DQNInference(str(path), digest).score(wire)
    assert len(scores) == 3 and max(scores, key=lambda r: r[1])[0]["target"] == [4, 1]
    result = neural.analyze(Position(), "hierarchical")
    assert result["selected_id"] == "pawn:4:1"
    assert result["neural"]["hierarchy"]["selected_type"] == "pawn"
    assert len(result["candidates"]) == 131
    assert sum(c["route_eligible"] for c in result["candidates"]) == 3
    detail = neural.branch(Position(), Move(type="pawn", target=(4, 1), player=1), "hierarchical")
    assert {s["objective"] for s in detail["neural"]["sensitivities"]} == {"q", "margin", "type_q", "type_margin"}
    json.dumps(result, allow_nan=False)
    neural.cached_runner.cache_clear()


def test_ddqn_bootstrap_follows_type_before_action():
    online, target = controlled_network(), controlled_network()
    with torch.no_grad():
        target.head[-1].bias[37] = .75
    mask = torch.tensor([legal_action_mask(ENGINE, ENGINE.initial_state())])
    obs = torch.tensor([encode_observation_cnn(ENGINE.initial_state())])
    batch = {"obs": obs, "next_obs": obs, "action": torch.tensor([37]), "reward": torch.tensor([0.]),
             "done": torch.tensor([0.]), "next_mask": mask}
    result = train_step(online, target, torch.optim.SGD(online.parameters(), lr=0.), batch,
                        gamma=.5, algorithm="double_dqn")
    assert result.mean_target_q == pytest.approx(.375)
    assert torch.isfinite(online.action_type_head.bias.grad).all()
    assert online.action_type_head.bias.grad[0] != 0


def test_flat_cnn_forward_remains_identical():
    net = CNNQNetwork(cnn_channels=[2, 3])
    obs = torch.randn(4, 7, 9, 9)
    reference = net.head(net.conv(obs).flatten(1))
    q, kind = net.forward_heads(obs)
    assert kind is None and torch.equal(reference, q)
