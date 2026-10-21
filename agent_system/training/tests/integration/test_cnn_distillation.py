"""Reflections preserve engine legality, teacher optimality and checkpoint meaning."""
import copy
import random

import numpy as np
import pytest
import torch

from agent_system.analysis.state import ENGINE, Position, build_state, to_position
from agent_system.runtime.service.agents.minimax_agent import inspect_branch, score_position
from agent_system.training.dqn.action_space import decode_action_id, encode_engine_action
from agent_system.training.dqn.checkpoint import load_checkpoint
from agent_system.training.dqn.distillation import (
    _collect_game, policy_loss, reflect_batch, reflection_permutation, train_distillation,
)
from agent_system.training.dqn.observation_cnn import encode_observation_cnn
from scripts.train_dqn import TrainConfig


def reflected_position(position, fx, fy):
    result = copy.deepcopy(position)
    result["pawns"] = [[8-x if fx else x, 8-y if fy else y] for x, y in position["pawns"]]
    for wall in result["walls"]:
        wall["x"] = 7 - wall["x"] if fx else wall["x"]
        wall["y"] = 7 - wall["y"] if fy else wall["y"]
    if fy:
        result["pawns"].reverse()
        result["walls_remaining"].reverse()
        result["turn"] = 3 - result["turn"]
    return result


@pytest.mark.parametrize("fx,fy", [(False, False), (True, False), (False, True), (True, True)])
def test_reflections_match_real_engine_with_wall_anchors_and_player_swap(fx, fy):
    state, rng = ENGINE.initial_state(), random.Random(914)
    for _ in range(11):
        state = ENGINE.apply_action(state, rng.choice(ENGINE.legal_actions(state)))
    candidates, _ = score_position(state, 2)
    expert = encode_engine_action(max(candidates, key=lambda c: c["score"])["action"])
    obs = torch.tensor(np.asarray(encode_observation_cnn(state)))[None]
    legal = torch.zeros((1, 209), dtype=torch.bool)
    legal[0, [encode_engine_action(c["action"]) for c in candidates]] = True
    actual = build_state(Position(**reflected_position(to_position(state), fx, fy)))
    transformed, mask, label = reflect_batch(obs, legal, torch.tensor([expert]), fx, fy)
    assert torch.equal(transformed[0], torch.tensor(np.asarray(encode_observation_cnn(actual))))
    assert set(mask[0].nonzero().flatten().tolist()) == {encode_engine_action(a) for a in ENGINE.legal_actions(actual)}
    chosen = decode_action_id(label.item(), actual.current_player)
    exact = inspect_branch(actual, chosen, 2)[0]
    assert exact == max(c["score"] for c in score_position(actual, 2)[0])
    restored, restored_mask, restored_label = reflect_batch(transformed, mask, label, fx, fy)
    assert torch.equal(restored, obs) and torch.equal(restored_mask, legal) and restored_label.item() == expert
    permutation = reflection_permutation(fx, fy)
    assert len(set(permutation.tolist())) == 209


def test_masked_policy_loss_has_no_illegal_gradient_and_rejects_bad_label():
    logits = torch.tensor([[0., 100., 1.]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    loss = policy_loss(logits, mask, torch.tensor([0]))
    loss.backward()
    assert logits.grad[0, 1] == 0 and logits.grad[0, 0] < 0
    with pytest.raises(ValueError, match="legal"):
        policy_loss(logits, mask, torch.tensor([1]))


def test_shared_collection_and_fresh_training_record_real_episode_and_logit_semantics(tmp_path):
    config = {"seed": 33, "max_plies": 8, "random_prefix_max": 4,
              "exploration": .2, "random_pawn_probability": .7}
    rows, trace = _collect_game((0, config))
    assert len(rows) == len(trace["actions"]) == 8
    assert all(row["legal"][row["expert_d1"]] and row["legal"][row["expert_d2"]] for row in rows)
    data = {key: torch.from_numpy(np.asarray([row[key] for row in rows])) for key in rows[0]}
    path = tmp_path / "shared.pt"
    torch.save({"train": data, "metadata": {"episodes": 1}}, path)
    hashes = []
    for depth in (1, 2):
        output = tmp_path / f"d{depth}"; output.mkdir()
        cfg = TrainConfig(episodes=1, model_arch="cnn", cnn_channels=[2, 2],
            cnn_activation="leaky_relu", device="cpu", seed=42, batch_size=4)
        checkpoints = []
        result = train_distillation(cfg, {"depth": depth, "dataset": str(path), "updates": 2,
            "batch_seed": 874, "checkpoint_updates": [1, 2]}, output,
            lambda p, _: checkpoints.append(load_checkpoint(p)))
        hashes.append(result["initialization"]["initial_weights_sha256"])
        assert [c.episode_count for c in checkpoints] == [1, 1]
        assert [c.training_step for c in checkpoints] == [1, 2]
        assert checkpoints[-1].algorithm == "policy_distillation"
        assert checkpoints[-1].eval_summary["output_semantics"] == "policy_logits"
    assert hashes[0] == hashes[1]
