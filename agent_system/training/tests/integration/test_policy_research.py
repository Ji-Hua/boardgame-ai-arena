"""Algorithm boundaries, same-player transitions and serving compatibility."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from agent_system.analysis.state import ENGINE, Move, Position, build_state
from agent_system.runtime.inference.dqn import DQNInference
from agent_system.training.dqn.checkpoint import load_checkpoint, save_checkpoint
from agent_system.training.dqn.distillation import _collect_game
from agent_system.training.dqn.model import CNNQNetwork
from agent_system.training.dqn.observation_cnn import encode_observation_cnn
from agent_system.training.dqn.policy_research import (
    ActorCritic, ComputeMeter, collect_games, gae, initialize, masked_distribution,
    observation, ppo_objective, train_research, weights_hash, shortest_path_features,
    potential, potential_shaping_reward, discounted_shaping_sum,
)
from quoridor_engine import Player
from scripts.train_dqn import TrainConfig


@pytest.fixture
def parent(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "parent.pt"
    network = CNNQNetwork(cnn_channels=[2, 2], activation="leaky_relu")
    save_checkpoint(path, network, model_arch="cnn", algorithm="policy_distillation",
                    eval_summary={"output_semantics": "policy_logits"})
    return path, network


def test_actor_unchanged_value_head_and_transfer_only_conv(parent):
    path, network = parent
    cfg = TrainConfig(seed=71, model_arch="cnn", cnn_channels=[2, 2], cnn_activation="leaky_relu", device="cpu")
    config = {"parent_checkpoint": str(path), "parent_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    initialized = {}
    for mode in ("policy", "conv_only", "random"):
        initialized[mode] = initialize(cfg, {**config, "initialization": mode})[0]
    assert weights_hash(initialized["policy"]) == weights_hash(network)
    assert weights_hash(initialized["conv_only"].conv) == weights_hash(network.conv)
    assert weights_hash(initialized["conv_only"].head) == weights_hash(initialized["random"].head)
    assert weights_hash(initialized["conv_only"].head) != weights_hash(network.head)
    ac = ActorCritic(initialized["policy"])
    obs = torch.randn(3, 7, 9, 9)
    logits, value = ac(obs)
    assert torch.equal(logits, network(obs)) and torch.equal(value, torch.zeros(3))
    value.square().sum().backward()
    assert ac.value.weight.grad is not None


def test_gae_terminal_truncation_and_same_player_sign():
    # A win after the opponent reply is still +1 for the learner.
    advantage, returns = gae([0., 1.], [.2, .4], [False, True], 99., .9, 1.)
    np.testing.assert_allclose(returns, [.9, 1.], atol=1e-7)
    np.testing.assert_allclose(advantage, [.7, .6], atol=1e-7)
    # A time limit bootstraps; it is not recorded as a terminal draw/loss.
    _, returns = gae([0., 0.], [.2, .4], [False, False], .5, .9, 1.)
    np.testing.assert_allclose(returns, [.405, .45], atol=1e-7)
    _, returns = gae([-1.], [.4], [True], .9, .9, .95)
    np.testing.assert_allclose(returns, [-1.], atol=1e-7)


def test_potential_shaping_uses_fixed_learner_view_and_telescopes():
    state = build_state(Position(pawns=((4, 3), (2, 5)), turn=1))
    p1 = shortest_path_features(ENGINE, state, Player.P1)
    p2 = shortest_path_features(ENGINE, state, Player.P2)
    assert p1["path_difference"] == -p2["path_difference"]
    assert potential(p1, alpha=.01, beta=.01, terminal=True) == 0.
    middle = {"self_distance": p1["self_distance"] - 1, "opponent_distance": p1["opponent_distance"],
              "path_difference": p1["path_difference"] + 1}
    final = {"self_distance": middle["self_distance"] - 1, "opponent_distance": middle["opponent_distance"] + 1,
             "path_difference": middle["path_difference"] + 2}
    gamma = .99
    first, phi0, phi1 = potential_shaping_reward(p1, middle, alpha=.01, beta=.01, gamma=gamma, terminal=False)
    second, _, phi2 = potential_shaping_reward(middle, final, alpha=.01, beta=.01, gamma=gamma, terminal=False)
    # Time-limit bootstrap retains the final potential residual exactly.
    assert discounted_shaping_sum([first, second], phi2, gamma) == pytest.approx(-phi0)
    terminal, _, zero = potential_shaping_reward(middle, final, alpha=.01, beta=.01, gamma=gamma, terminal=True)
    assert zero == 0.
    assert discounted_shaping_sum([first, terminal], zero, gamma) == pytest.approx(-phi0)


def test_ppo_legal_distribution_ratio_clipping_and_critic_gradient():
    logits = torch.tensor([[0., 100., 0.], [0., 100., 0.]], requires_grad=True)
    legal = torch.tensor([[True, False, True], [True, False, True]])
    distribution = masked_distribution(logits, legal)
    assert torch.equal(distribution.probs[:, 1], torch.zeros(2))
    action = torch.tensor([0, 2])
    value = torch.tensor([.1, .2], requires_grad=True)
    batch = {"legal": legal, "action": action, "logp": distribution.log_prob(action).detach(),
             "advantage": torch.tensor([1., -1.]), "return": torch.tensor([1., -1.])}
    loss, metrics = ppo_objective(logits, value, batch, clip=.2, value_weight=.5, entropy_weight=.01)
    assert metrics["approx_kl"] == 0 and metrics["clip_fraction"] == 0
    loss.backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2)) and torch.isfinite(logits.grad).all()
    assert value.grad[0] < 0 and value.grad[1] > 0
    moved = torch.tensor([[3., 100., -3.], [3., 100., -3.]])
    _, metrics = ppo_objective(moved, value, batch, clip=.2, value_weight=.5, entropy_weight=0.)
    assert metrics["approx_kl"] > 0 and metrics["clip_fraction"] == 1
    with pytest.raises(ValueError, match="legal"):
        masked_distribution(logits, torch.zeros_like(legal))


@pytest.mark.parametrize("kind", ["dagger", "ppo", "double_dqn"])
def test_collector_closes_reply_on_odd_cap_and_replays_real_actions(parent, kind):
    _, actor = parent
    net = ActorCritic(actor) if kind == "ppo" else actor
    config = {"method": kind, "rollout_seed": 991, "random_prefix_max": 6,
              "max_plies": 5, "gamma": .99, "gae_lambda": .95}
    rows, counts, traces, keys = collect_games(net, config, first_episode=0, count=4,
        device="cpu", meter=ComputeMeter("cpu"), epsilon=.3)
    assert counts["episodes"] == 4 and counts["learner_steps"] == len(rows)
    assert counts.get("minimax_search_calls", 0) == (len(rows) if kind == "dagger" else 0)
    offset = 0
    for trace in traces:
        state = build_state(Position(**trace["opening"]))
        for action in trace["actions"]:
            learner = action["player"] == trace["learner_seat"]
            if learner:
                row = rows[offset]
                np.testing.assert_array_equal(row["obs"], observation(state))
                assert row["legal"][row["action"]]
                offset += 1
            state = ENGINE.apply_action(state, Move(**action).engine_action())
            if not learner or ENGINE.is_game_over(state):
                if offset and np.array_equal(rows[offset - 1]["next_obs"], observation(state)):
                    assert rows[offset - 1]["done"] or np.array_equal(rows[offset - 1]["obs"][6], observation(state)[6])
        assert trace["termination"] == "time_limit" and trace["outcome"] is None
        assert not rows[offset - 1]["done"] and rows[offset - 1]["next_mask"].any()
    assert offset == len(rows) and len(keys) > 0


def test_collector_zero_shaping_and_time_limit_residual(parent):
    _, actor = parent
    rows, counts, _, _ = collect_games(ActorCritic(actor), {
        "method": "ppo_imitation", "rollout_seed": 991, "random_prefix_max": 0,
        "max_plies": 5, "gamma": .99, "gae_lambda": .95,
        "reward_shaping": {"alpha": 0., "beta": 0.},
    }, first_episode=0, count=4, device="cpu", meter=ComputeMeter("cpu"))
    assert counts["auxiliary_reward_sum"] == 0 and counts["auxiliary_reward_max_abs"] == 0
    assert all(r["auxiliary_reward"] == 0 and r["reward"] == r["terminal_reward"] for r in rows)
    # The final transition is a nonterminal time cap and therefore bootstraps.
    assert not rows[-1]["done"]


@pytest.mark.parametrize("method", ["dagger", "ppo", "ppo_imitation", "selfplay_ppo", "double_dqn"])
def test_all_methods_train_save_and_keep_evaluation_out_of_rng(parent, tmp_path, method):
    path, _ = parent
    game_rows, _ = _collect_game((0, {"seed": 81, "max_plies": 8, "random_prefix_max": 0,
        "exploration": .2, "random_pawn_probability": .7}))
    dataset = tmp_path / "dataset.pt"
    torch.save({"train": {k: torch.from_numpy(np.asarray([r[k] for r in game_rows])) for k in game_rows[0]}}, dataset)
    output = tmp_path / method
    output.mkdir()
    config = {"method": method, "parent_checkpoint": str(path), "parent_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "initialization": "conv_only" if method == "double_dqn" else "policy", "batch_seed": 18,
        "dataset": str(dataset), "games_per_batch": 2, "rollout_seed": 92, "random_prefix_max": 0,
        "max_plies": 4, "gae_lambda": .95, "epochs": 2, "clip": .2, "target_kl": .03,
        "value_weight": .5, "entropy_weight": .01, "imitation_weight": .3,
        "updates_per_batch": 2, "updates_per_transition": .5, "checkpoint_episodes": [2, 4],
        "opponents": [{"id": "parent", "checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    cfg = TrainConfig(seed=71, episodes=4, model_arch="cnn", cnn_channels=[2, 2],
        cnn_activation="leaky_relu", device="cpu", batch_size=4, lr=1e-4, grad_clip_norm=.5, target_sync_interval=1)
    saved = []
    def callback(checkpoint, _):
        saved.append(load_checkpoint(checkpoint))
        torch.randn(50)
    result = train_research(cfg, config, output, callback)
    assert result["totals"]["episodes"] == 4 and result["updates"] > 0
    assert [c.episode_count for c in saved] == [2, 4]
    assert result["totals"]["minimax_search_calls"] > 0 if method == "dagger" else result["totals"]["minimax_search_calls"] == 0
    checkpoint = output / "checkpoints/ep00004.pt"
    inference = DQNInference(str(checkpoint), hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    assert inference.metadata["output_semantics"] == ("action_values" if method == "double_dqn" else "policy_logits")
    assert len(result["opponent_pool"]) == (2 if method == "selfplay_ppo" else 0)
    assert (output / "training-state/ep00004.pt").is_file()
