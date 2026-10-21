"""Search perspective, legal targets, calibration and checkpoint integration."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from quoridor_engine import Player

from agent_system.analysis.state import ENGINE, Move, Position, build_state
from agent_system.runtime.search.puct import SearchConfig, TorchEvaluator, backup, search_batch
from agent_system.training.dqn.action_space import encode_engine_action
from agent_system.training.dqn.alphazero_lite import (
    actor_hash, collect, initialize, policy_value_loss, reflect_examples, train_alphazero,
)
from agent_system.training.dqn.checkpoint import load_checkpoint, save_checkpoint
from agent_system.training.dqn.model import CNNQNetwork
from agent_system.training.dqn.policy_research import ComputeMeter, observation
from scripts.train_dqn import TrainConfig


def uniform(nodes):
    return [(n.legal.astype(float) / n.legal.sum(), 0.) for n in nodes]


def prior_to(action_fn, value=0.):
    def evaluate(nodes):
        result = []
        for node in nodes:
            action = action_fn(node.state)
            p = np.zeros(209)
            p[encode_engine_action(action)] = 1.
            result.append((p, value))
        return result
    return evaluate


def test_legality_visits_distribution_and_batch_equivalence():
    states = [ENGINE.initial_state(), build_state(Position(pawns=((2, 3), (6, 5)), turn=2))]
    config = SearchConfig(simulations=17)
    batch = search_batch(states, uniform, config)
    for state, result in zip(states, batch):
        single = search_batch([state], uniform, config)[0]
        legal = {encode_engine_action(a) for a in ENGINE.legal_actions(state)}
        assert result.action_id in legal
        assert result.root.visits.sum() == result.stats['simulations'] == 17
        assert result.distribution.sum() == pytest.approx(1.)
        assert all(result.distribution[i] == 0 for i in range(209) if i not in legal)
        np.testing.assert_array_equal(result.distribution, single.distribution)
        assert result.stats['model_evaluations'] == 18


@pytest.mark.parametrize('seat', [1, 2])
def test_terminal_value_and_winning_edge_both_seats(seat):
    state = build_state(Position(pawns=((3, 7), (4, 3)) if seat == 1 else ((4, 5), (3, 1)), turn=seat))
    win = Move(type='pawn', target=(3, 8 if seat == 1 else 0), player=seat).engine_action()
    terminal = ENGINE.apply_action(state, win)
    result = search_batch([terminal], lambda nodes: pytest.fail('terminal must not call network'))[0]
    expected = 1. if terminal.current_player == ENGINE.winner(terminal) else -1.
    assert result.root.value == expected
    assert result.stats['model_evaluations'] == result.stats['simulations'] == 0
    assert result.distribution.sum() == 0
    result = search_batch([state], prior_to(lambda s: win, value=-.9), SearchConfig(simulations=7))[0]
    assert result.action_id == encode_engine_action(win)
    i = result.ranked_indices[0]
    assert result.root.q()[i] == 1. and result.root.visits[i] == 7
    assert result.stats['terminal_leaves'] == 7 and result.stats['model_evaluations'] == 1


def test_backup_changes_perspective_on_every_edge():
    nodes = [SimpleNamespace(player=p, visits=np.zeros(1), value_sum=np.zeros(1))
             for p in (Player.P1, Player.P2, Player.P1)]
    backup([(n, 0) for n in nodes], Player.P2, -.8)
    np.testing.assert_allclose([n.value_sum[0] for n in nodes], [.8, -.8, .8])
    assert all(n.visits[0] == 1 for n in nodes)


def test_opponent_immediate_win_is_negative_at_root():
    state = build_state(Position(pawns=((4, 0), (2, 1))))
    def choose(s):
        if s.current_player == Player.P1:
            return Move(type='pawn', target=(4, 1), player=1).engine_action()
        return Move(type='pawn', target=(2, 0), player=2).engine_action()
    result = search_batch([state], prior_to(choose), SearchConfig(simulations=2))[0]
    i = np.where(result.root.ids == 37)[0][0]  # e2
    assert result.root.visits[i] == 2 and result.root.q()[i] == -.5
    assert result.stats['terminal_leaves'] == 1
    assert result.root.children[i].q().max() == 1.


def test_search_path_repetition_is_zero_and_bounded():
    def oscillate(s):
        x, y = s.pawn_pos(s.current_player)
        return Move(type='pawn', target=(3 if x == 4 else 4, y),
                    player=1 if s.current_player == Player.P1 else 2).engine_action()
    result = search_batch([ENGINE.initial_state()], prior_to(oscillate), SearchConfig(simulations=12))[0]
    assert result.stats['repetition_leaves'] > 0
    assert result.stats['max_depth'] <= 4
    assert result.root.q().max() == 0.


def test_soft_targets_mask_gradient_and_symmetry():
    logits = torch.zeros(2, 209, requires_grad=True)
    legal = torch.zeros_like(logits, dtype=torch.bool)
    legal[:, [3, 81, 200]] = True
    pi = legal.float() / 3
    value = torch.zeros(2, requires_grad=True)
    target = torch.tensor([1., -1.])
    loss, _, _ = policy_value_loss(logits, value, legal, pi, target)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and bool((logits.grad[~legal] == 0).all())
    assert value.grad[0] < 0 < value.grad[1]
    obs = torch.randn(2, 7, 9, 9)
    obs[:, 2:4, 8, :] = 0
    obs[:, 2:4, :, 8] = 0
    for fx, fy in [(True, False), (False, True), (True, True)]:
        x, lm, p = reflect_examples(obs, legal, pi, fx, fy)
        restored = reflect_examples(x, lm, p, fx, fy)
        for a, b in zip(restored, (obs, legal, pi)):
            assert torch.equal(a, b)
    with pytest.raises(ValueError, match='legal'):
        policy_value_loss(logits, value, legal, torch.ones_like(pi)/209, target)


def test_collect_real_outcomes_each_player_and_no_policy_calibration_update(tmp_path):
    torch.set_num_threads(1)
    parent = CNNQNetwork(cnn_channels=[2], dense_width=8, activation='leaky_relu')
    path = tmp_path/'parent.pt'
    save_checkpoint(path, parent, model_arch='cnn', algorithm='policy_distillation')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    network = initialize(path, digest)
    assert actor_hash(network) == actor_hash(parent)
    x = torch.randn(3, 7, 9, 9)
    assert torch.equal(network(x), parent(x))
    assert bool((network.forward_policy_value(x)[1] == 0).all())
    config = {'phase': 'calibration', 'parent_checkpoint': str(path), 'parent_sha256': digest,
        'batch_seed': 42, 'rollout_seed': 711, 'random_prefix_max': 0, 'max_plies': 6,
        'search': {'simulations': 2}, 'temperature_plies': 2, 'games_per_batch': 5,
        'calibration_epochs': 2, 'calibration_lr': .0001, 'checkpoint_episodes': [10],
        'replay_capacity': 200, 'updates_per_batch': 2}
    rows, totals, traces, keys = collect(network, config, 0, 2, 'cpu', ComputeMeter('cpu'))
    offset = 0
    for game in traces:
        state = build_state(Position(**game['opening']))
        for wire in game['actions']:
            row = rows[offset]
            np.testing.assert_array_equal(row['obs'], observation(state))
            assert row['player'] == state.current_player
            assert row['outcome'] == 0
            state = ENGINE.apply_action(state, Move(**wire).engine_action())
            offset += 1
    assert totals['games'] == 2 and totals['simulations'] == 24
    output = tmp_path/'calibration'
    output.mkdir()
    cfg = TrainConfig(seed=42, episodes=10, model_arch='cnn', device='cpu', batch_size=4, lr=1e-4, grad_clip_norm=1.)
    diagnosed = []
    result = train_alphazero(cfg, config, output, lambda p, r: diagnosed.append(p))
    assert result['actor_sha256'] == result['initial_actor_sha256'] == actor_hash(parent)
    assert len(diagnosed) == 1
    calibrated = load_checkpoint(diagnosed[0])
    assert calibrated.model_config['cnn_value_head'] and calibrated.eval_summary['value_head_status'] == 'calibrated'
    config.update(phase='selfplay', parent_checkpoint=str(diagnosed[0]),
        parent_sha256=hashlib.sha256(diagnosed[0].read_bytes()).hexdigest(), checkpoint_episodes=[10])
    output2 = tmp_path/'selfplay'
    output2.mkdir()
    result2 = train_alphazero(cfg, config, output2, lambda p, r: None)
    assert result2['episodes'] == 10 and result2['gradient_updates'] == 4
    final = load_checkpoint(result2['checkpoints'][0])
    values = final.network.forward_policy_value(x)[1]
    assert bool((values.abs() <= 1).all())
    evaluator = TorchEvaluator(final.network)
    search_batch([ENGINE.initial_state()], evaluator, SearchConfig(simulations=3))
    assert evaluator.stats['model_evaluations'] == 4


@pytest.mark.parametrize('seat', [1, 2])
def test_random_prefix_near_goal_never_finishes_the_game(seat):
    import random
    from agent_system.training.dqn.alphazero_lite import prefix_action
    state = build_state(Position(pawns=((3, 7), (4, 3)) if seat == 1 else ((4, 5), (3, 1)), turn=seat))
    for seed in range(100):
        action = prefix_action(ENGINE, state, random.Random(seed))
        assert action is not None
        assert not ENGINE.is_game_over(ENGINE.apply_action(state, action))


def test_regression_winning_prefix_at_episode_1043():
    torch.set_num_threads(1)
    network = CNNQNetwork(cnn_channels=[2], dense_width=8, value_head=True)
    config = {'rollout_seed': 2026102702, 'random_prefix_max': 24, 'max_plies': 2,
              'search': {'simulations': 1}, 'temperature_plies': 2}
    rows, totals, traces, _ = collect(network, config, 1042, 1, 'cpu', ComputeMeter('cpu'))
    assert totals['games'] == 1 and rows
    state = ENGINE.initial_state()
    for wire in traces[0]['prefix']:
        state = ENGINE.apply_action(state, Move(**wire).engine_action())
        assert not ENGINE.is_game_over(state)


def test_collection_terminal_targets_follow_each_recorded_player():
    class ForwardPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = torch.nn.Linear(1, 1)
        def forward_policy_value(self, obs):
            goal = obs[:, 6].sum(1).argmax(-1)
            logits = torch.full((len(obs), 209), -100., device=obs.device)
            logits[:, :81] = -(torch.arange(81, device=obs.device)[None, :] % 9 - goal[:, None]).abs().float()
            return logits, torch.zeros(len(obs), device=obs.device)
    config = {'rollout_seed': 11, 'random_prefix_max': 0, 'max_plies': 40,
              'search': {'simulations': 1}, 'temperature_plies': 0}
    rows, totals, traces, _ = collect(ForwardPolicy(), config, 0, 2, 'cpu', ComputeMeter('cpu'))
    assert totals['goal'] == 2
    offset = 0
    for game in traces:
        episode_rows = rows[offset:offset+len(game['actions'])]
        offset += len(episode_rows)
        winner = Player.P1 if game['winner'] == 1 else Player.P2
        assert episode_rows[-1]['outcome'] == 1.
        assert episode_rows[-2]['outcome'] == -1.
        assert all(r['outcome'] == (1. if r['player'] == winner else -1.) for r in episode_rows)
