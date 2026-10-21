"""Regression tests exercise the real trainer, checkpoint loader and Greedy."""
import random
import hashlib

import pytest
import torch
from quoridor_engine import RuleEngine

from scripts import train_dqn
from agent_system.training.dqn.checkpoint import load_checkpoint, save_checkpoint
from agent_system.training.dqn.model import CNNQNetwork
from agent_system.training.dqn.opponent import build_opponent
from agent_system.training.dqn.action_space import encode_engine_action, decode_action_id
from agent_system.runtime.service.agents.greedy_agent import score_position


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
def test_actual_training_truncation_keeps_learner_perspective(tmp_path, monkeypatch, cap):
    observed = []
    original = train_dqn.ReplayBuffer.push

    def capture(buffer, obs, action, reward, next_obs, done, mask):
        observed.append((obs, next_obs, done, mask))
        return original(buffer, obs, action, reward, next_obs, done, mask)

    monkeypatch.setattr(train_dqn.ReplayBuffer, "push", capture)
    cfg = train_dqn.TrainConfig(episodes=2, max_steps_per_episode=cap,
        warmup_size=100, buffer_capacity=128, batch_size=2,
        model_arch="cnn", cnn_channels=[2, 2, 2], device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"), checkpoint_interval=2)
    callbacks = []
    result = train_dqn.train(cfg, checkpoint_callback=lambda p, r: callbacks.append(load_checkpoint(p)))
    assert result.total_truncated_episodes == 2
    assert result.total_optimizer_steps == 0
    assert len(callbacks) == 1
    for obs, next_obs, done, mask in observed:
        assert not done and any(mask)
        # CNN goal plane distinguishes P1 and P2; it must never change within
        # a bootstrapped transition, including at an odd/even sampling cap.
        assert torch.equal(torch.tensor(obs)[6], torch.tensor(next_obs)[6])
    assert all(cap <= length <= cap + 1 for length in result.episode_lengths)


@pytest.mark.parametrize("activation", ["relu", "leaky_relu"])
def test_cnn_checkpoint_preserves_activation_and_width(tmp_path, activation):
    net = CNNQNetwork(cnn_channels=[4, 8], dense_width=32, activation=activation)
    x = torch.randn(2, 7, 9, 9)
    path = tmp_path / "cnn.pt"
    save_checkpoint(path, net, model_arch="cnn")
    restored = load_checkpoint(path).network
    assert restored.activation == activation
    assert torch.equal(net(x), restored(x))


def test_historical_cnn_without_activation_metadata_stays_relu(tmp_path):
    net = CNNQNetwork(cnn_channels=[2, 2, 2])
    path = tmp_path / "old.pt"
    save_checkpoint(path, net, model_arch="cnn")
    data = torch.load(path, weights_only=False)
    data["model_config"].pop("cnn_activation")
    data["model_config"].pop("dense_width")
    torch.save(data, path)
    restored = load_checkpoint(path).network
    x = torch.randn(7, 9, 9)
    assert restored.activation == "relu"
    assert torch.equal(net(x), restored(x))


def test_training_greedy_matches_runtime_tie_break():
    engine, rng = RuleEngine.standard(), random.Random(22)
    state = engine.initial_state()
    opponent = build_opponent("greedy")
    for _ in range(12):
        candidates = score_position(state)
        expected = encode_engine_action(max(candidates, key=lambda c: c["score"])["action"])
        legal = [encode_engine_action(c["action"]) for c in candidates]
        assert opponent.select_action_id(engine, state, legal, rng) == expected
        state = engine.apply_action(state, decode_action_id(rng.choice(legal), state.current_player))


def test_demonstrations_use_complete_same_player_transitions(tmp_path):
    from agent_system.training.dqn.demonstrations import collect_demonstrations, symmetry_key
    from agent_system.training.dqn.reward import RewardConfig
    data = collect_demonstrations({"episodes": 22, "data_seed": 123},
        RewardConfig(mode="distance_delta", distance_reward_weight=.04, distance_delta_clip=1), .95, tmp_path)
    for split in ("train", "validation"):
        rows = data[split]
        ongoing = ~rows["done"]
        assert torch.equal(rows["obs"][ongoing, 6], rows["next_obs"][ongoing, 6])
        assert torch.equal(rows["return"][rows["done"]], rows["reward"][rows["done"]])
        assert rows["return"].abs().max() <= 1.040001
        assert rows["legal"].gather(1, rows["expert"][:, None]).all()
        assert not rows["next_mask"][rows["done"]].any()
    train_keys = {symmetry_key(o.numpy()) for o in data["train"]["obs"]}
    assert not train_keys.intersection(symmetry_key(o.numpy()) for o in data["validation"]["obs"])


def test_expert_margin_ignores_illegal_outputs_and_has_learning_gradient():
    from agent_system.training.dqn.demonstrations import expert_margin
    q = torch.tensor([[.1, .2, 1000.]], requires_grad=True)
    loss = expert_margin(q, torch.tensor([[True, True, False]]), torch.tensor([0]), .1)
    assert loss.item() == pytest.approx(.2)
    loss.backward()
    assert q.grad.tolist() == [[-1., 1., 0.]]


def test_opponent_win_on_closing_reply_is_terminal_not_truncation(tmp_path, monkeypatch):
    import quoridor_engine
    from types import SimpleNamespace
    from agent_system.analysis.state import Position, build_state
    actual = RuleEngine.standard()
    start = build_state(Position(pawns=((0, 0), (8, 1)), turn=1))

    class Engine:
        def initial_state(self):
            return start
        def __getattr__(self, name):
            return getattr(actual, name)

    monkeypatch.setattr(quoridor_engine, "RuleEngine", SimpleNamespace(standard=Engine))
    monkeypatch.setattr(train_dqn, "select_epsilon_greedy_action", lambda *a, **kw: 1)  # a2
    # Force the opponent's i1 finish; this tests transition semantics rather
    # than assuming Greedy always prioritizes a win over a longer wall detour.
    monkeypatch.setattr(train_dqn, "build_opponent", lambda *a, **kw:
        SimpleNamespace(select_action_id=lambda *a: 72))
    pushed = []
    original = train_dqn.ReplayBuffer.push
    def capture(buffer, *transition):
        pushed.append(transition)
        return original(buffer, *transition)
    monkeypatch.setattr(train_dqn.ReplayBuffer, "push", capture)
    cfg = train_dqn.TrainConfig(episodes=1, max_steps_per_episode=1,
        warmup_size=100, buffer_capacity=128, model_arch="cnn", cnn_channels=[2, 2, 2],
        device="cpu", opponent="greedy", checkpoint_dir=str(tmp_path / "checkpoints"))
    result = train_dqn.train(cfg)
    assert result.total_env_steps == 2
    assert result.total_truncated_episodes == 0
    assert result.total_neg_rewards == 1
    assert len(pushed) == 1
    assert pushed[0][2] == -1.0 and pushed[0][4] is True
    assert not any(pushed[0][5])


def test_actual_trainer_can_retain_demonstrations(tmp_path):
    from agent_system.training.dqn.demonstrations import collect_demonstrations
    from agent_system.training.dqn.reward import RewardConfig
    collect_demonstrations({"episodes": 22, "data_seed": 123}, RewardConfig(), .95, tmp_path)
    cfg = train_dqn.TrainConfig(episodes=2, max_steps_per_episode=4,
        warmup_size=1, batch_size=1, buffer_capacity=16, gamma=.95,
        model_arch="cnn", cnn_channels=[2, 2, 2], cnn_activation="leaky_relu",
        device="cpu", demonstration_dataset=str(tmp_path / "demonstrations.pt"),
        checkpoint_dir=str(tmp_path / "checkpoints"))
    result = train_dqn.train(cfg)
    assert result.total_optimizer_steps > 0
    assert 0 < result.last_demonstration_loss < 10
    assert load_checkpoint(result.final_checkpoint_path).eval_summary["demonstration_dataset"] == cfg.demonstration_dataset


def test_demonstration_warm_start_preserves_parent_weights_and_provenance(tmp_path):
    from agent_system.training.dqn.demonstrations import train_demonstrations
    from agent_system.training.dqn.observation_cnn import encode_observation_cnn
    from agent_system.training.dqn.action_space import legal_action_mask
    engine = RuleEngine.standard()
    state = engine.initial_state()
    obs = torch.tensor([encode_observation_cnn(state)])
    mask = torch.tensor([legal_action_mask(engine, state)])
    rows = {"obs": obs, "next_obs": obs, "legal": mask, "next_mask": mask,
            "action": torch.tensor([37]), "expert": torch.tensor([37]),
            "reward": torch.tensor([0.]), "return": torch.tensor([.1]), "done": torch.tensor([False])}
    dataset_path = tmp_path / "demo.pt"
    torch.save({"train": rows, "validation": rows,
                "metadata": {"gamma": .95, "completed_games": 1}}, dataset_path)
    parent = CNNQNetwork(cnn_channels=[2, 2], activation="relu")
    with torch.no_grad():
        parent.head[-1].bias[37] = .12345
    parent_path = tmp_path / "parent.pt"
    save_checkpoint(parent_path, parent, model_arch="cnn", episode_count=2000, training_step=123)
    output = tmp_path / "run"; output.mkdir()
    cfg = train_dqn.TrainConfig(model_arch="cnn", cnn_channels=[2, 2], cnn_activation="leaky_relu",
        device="cpu", warm_start_checkpoint=str(parent_path), gamma=.95, lr=0., batch_size=1)
    saved = []
    result = train_demonstrations(cfg, {"dataset": str(dataset_path), "updates": 1,
        "checkpoint_updates": 1}, output, lambda path, _: saved.append(load_checkpoint(path)))
    assert len(saved) == 1
    assert saved[0].network.activation == "leaky_relu"
    assert all(torch.equal(v, saved[0].network.state_dict()[k]) for k, v in parent.state_dict().items())
    assert saved[0].training_step == 1
    info = saved[0].eval_summary["initialization"]
    assert info == result["initialization"]
    assert info["mode"] == "weights_only" and info["parent_episode_count"] == 2000
    assert info["parent_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
