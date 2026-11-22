"""Actual teacher search, rollout transitions, and episode-counted training."""
import json
import random

import numpy as np
import pytest
import torch
from quoridor_engine import RuleEngine

from agent_system.training.dqn import teacher_training as teacher
from agent_system.training.dqn.action_space import encode_engine_action
from agent_system.training.dqn.checkpoint import load_checkpoint
from agent_system.training.dqn.opponent import build_opponent
from agent_system.training.dqn.reward import RewardConfig
from scripts.train_dqn import TrainConfig


@pytest.mark.parametrize("depth", [1, 2])
def test_teacher_choice_is_a_full_window_minimax_maximum(depth):
    from agent_system.runtime.service.agents.minimax_agent import inspect_branch
    from agent_system.evaluation.arena.agents.scorers import MinimaxScorer
    from backend.adapters.engine_adapter import _serialize_state
    from agent_system.analysis.state import Move
    engine, rng = RuleEngine.standard(), random.Random(500)
    state = engine.initial_state()
    for _ in range(8):
        candidates, chosen, _ = teacher.teacher_candidates(state, depth)
        exact = [(encode_engine_action(c["action"]), inspect_branch(state, c["action"], depth)[0])
                 for c in candidates]
        # Alpha-beta bounds are sufficient to choose the correct first maximum,
        # but those bounds must not be treated as action-value targets.
        assert chosen == max(exact, key=lambda item: item[1])[0]
        arena = MinimaxScorer(depth).score(_serialize_state(state))
        assert chosen == encode_engine_action(Move(**max(arena, key=lambda item: item[1])[0]).engine_action())
        state = engine.apply_action(state, rng.choice(candidates)["action"])


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
@pytest.mark.parametrize("episode", [0, 1])
def test_capped_teacher_rollout_has_no_mc_and_keeps_same_player(cap, episode):
    rows, stats, _, keys = teacher.collect_teacher_episode(RuleEngine.standard(), None,
        build_opponent("greedy"), random.Random(33), episode=episode, depth=1,
        max_plies=cap, teacher_probability=1., epsilon=0., gamma=.95,
        reward_config=RewardConfig(), device="cpu")
    assert stats["truncated_games"] == 1 and stats["completed_games"] == 0
    assert cap <= stats["env_plies"] <= cap + 1
    assert keys
    for row in rows:
        assert not row["done"] and not row["mc_valid"]
        assert np.array_equal(row["obs"][6], row["next_obs"][6])
        assert row["next_mask"].any() and row["legal"][row["expert"]]


def test_completed_teacher_rollout_returns_are_real_discounted_rewards():
    rows, stats, _, _ = teacher.collect_teacher_episode(RuleEngine.standard(), None,
        build_opponent("greedy"), random.Random(404), episode=0, depth=2,
        max_plies=3000, teacher_probability=1., epsilon=.15, gamma=.95,
        reward_config=RewardConfig(), device="cpu")
    assert stats["completed_games"] == 1 and rows[-1]["done"]
    assert rows[-1]["return"] in (-1., 1.)
    for i, row in enumerate(rows):
        assert row["mc_valid"] and row["legal"][row["expert"]]
        expected = row["reward"] + (0. if row["done"] else .95 * rows[i+1]["return"])
        assert row["return"] == pytest.approx(expected)


def test_replay_wrap_keeps_complete_aligned_rows():
    replay = teacher.TeacherReplay(3, "cpu")
    replay.append([{"action": i, "expert": 2*i} for i in range(7)])
    assert replay.size == 3
    assert set(replay.tensors["action"].tolist()) == {4, 5, 6}
    assert torch.equal(replay.tensors["expert"], 2 * replay.tensors["action"])


def test_real_trainer_counts_games_and_uses_identical_fresh_initialization(tmp_path):
    hashes = []
    for depth in (1, 2):
        output = tmp_path / f"d{depth}"; output.mkdir()
        cfg = TrainConfig(episodes=4, max_steps_per_episode=4, warmup_size=1,
            buffer_capacity=16, batch_size=2, model_arch="cnn", cnn_channels=[2, 2],
            cnn_activation="leaky_relu", device="cpu", seed=77, lr=0.,
            checkpoint_interval=2, log_interval=4, checkpoint_dir=str(output / "checkpoints"))
        checkpoints = []
        result = teacher.train_teacher(cfg, {"depth": depth, "batch_seed": 88,
            "teacher_probability_start": 1., "teacher_probability_end": 1.,
            "margin": .1, "margin_weight": 1.}, output,
            lambda path, _: checkpoints.append(load_checkpoint(path)))
        assert [c.episode_count for c in checkpoints] == [2, 4]
        assert result["episodes"] == 4 and result["totals"]["truncated_games"] == 4
        assert result["updates"] == result["totals"]["learner_transitions"] == 8
        assert checkpoints[-1].training_step == 8
        assert checkpoints[-1].eval_summary["teacher_depth"] == depth
        assert result["initialization"]["mode"] == "fresh"
        hashes.append(result["initialization"]["initial_weights_sha256"])
        assert json.loads((output / "training-state-keys.json").read_text())
        saved = torch.load(output / "teacher-replay.pt", weights_only=True)["train"]
        assert not saved["mc_valid"].any()
        assert torch.equal(saved["obs"][:, 6], saved["next_obs"][:, 6])
    assert hashes[0] == hashes[1]


def test_teacher_training_rejects_parent_weights(tmp_path):
    cfg = TrainConfig(model_arch="cnn", warm_start_checkpoint="parent.pt")
    with pytest.raises(ValueError, match="fresh flat CNN"):
        teacher.train_teacher(cfg, {}, tmp_path, lambda *_: None)
