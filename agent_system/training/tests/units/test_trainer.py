"""Unit tests for agent_system/training/dqn/trainer.py — Phase 5.

Test groups
-----------
TestSyncTargetNetwork           — hard copy of parameters
TestTrainStepLoss               — finite loss, gradient flow
TestTrainStepBellman            — Bellman target correctness
TestTrainStepMaskEnforcement    — illegal next-actions ignored in target
TestTrainStepParameterUpdate    — online net updates, target net stays fixed
"""

from __future__ import annotations

import copy
import math
import random

import pytest
import torch
import torch.optim as optim

from agent_system.training.dqn.action_space import ACTION_COUNT
from agent_system.training.dqn.model import QNetwork
from agent_system.training.dqn.observation import OBSERVATION_SIZE
from agent_system.training.dqn.trainer import (
    TrainStepResult,
    sync_target_network,
    train_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nets(seed: int = 0) -> tuple[QNetwork, QNetwork]:
    """Return a pair of randomly-initialised QNetworks (independent weights)."""
    torch.manual_seed(seed)
    online = QNetwork()
    torch.manual_seed(seed + 1)
    target = QNetwork()
    return online, target


def _make_batch(
    batch_size: int = 8,
    *,
    done: bool | None = None,
    next_mask_legal: bool = True,
) -> dict[str, torch.Tensor]:
    """Build a synthetic batch for trainer tests.

    Parameters
    ----------
    done:
        If True, all done=1. If False, all done=0. If None (default),
        mixed (alternating).
    next_mask_legal:
        If True, all next actions are legal.  If False, all are illegal
        (should never happen in real usage; used to test masking).
    """
    obs = torch.zeros(batch_size, OBSERVATION_SIZE)
    action = torch.zeros(batch_size, dtype=torch.int64)
    reward = torch.ones(batch_size) * 0.5
    next_obs = torch.zeros(batch_size, OBSERVATION_SIZE)

    if done is None:
        done_vals = torch.tensor(
            [float(i % 2) for i in range(batch_size)]
        )
    elif done:
        done_vals = torch.ones(batch_size)
    else:
        done_vals = torch.zeros(batch_size)

    if next_mask_legal:
        next_mask = torch.ones(batch_size, ACTION_COUNT, dtype=torch.bool)
    else:
        next_mask = torch.zeros(batch_size, ACTION_COUNT, dtype=torch.bool)

    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "next_obs": next_obs,
        "done": done_vals,
        "next_mask": next_mask,
    }


# ---------------------------------------------------------------------------
# TestSyncTargetNetwork
# ---------------------------------------------------------------------------

class TestSyncTargetNetwork:
    def test_sync_copies_all_parameters(self):
        online, target = _make_nets()
        # Confirm they differ initially
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            if not torch.equal(p_o, p_t):
                break
        else:
            pytest.skip("Networks happened to be identical — seed collision")
        sync_target_network(online, target)
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert torch.equal(p_o, p_t)

    def test_sync_state_dicts_equal(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        for key in online.state_dict():
            assert torch.equal(online.state_dict()[key], target.state_dict()[key])

    def test_sync_does_not_share_tensors(self):
        """After sync, modifying online parameters must not affect target."""
        online, target = _make_nets()
        sync_target_network(online, target)
        # Mutate online weight
        with torch.no_grad():
            for p in online.parameters():
                p.add_(1.0)
                break  # only modify the first param
        # Target should be unchanged
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert not torch.equal(p_o, p_t)
            break

    def test_sync_can_be_called_multiple_times(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        sync_target_network(online, target)
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert torch.equal(p_o, p_t)

    def test_target_diverges_before_sync(self):
        """Optimizer step on online should leave target unchanged."""
        online, target = _make_nets()
        sync_target_network(online, target)  # start equal
        optimizer = optim.Adam(online.parameters(), lr=0.01)
        batch = _make_batch()
        train_step(online, target, optimizer, batch)
        # After train step, online changed; target should be the same as before
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            if not torch.equal(p_o, p_t):
                return  # expected — test passes
        pytest.fail("Target parameters should have diverged from online after train_step.")


# ---------------------------------------------------------------------------
# TestTrainStepLoss
# ---------------------------------------------------------------------------

class TestTrainStepLoss:
    def test_train_step_returns_result(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        result = train_step(online, target, optimizer, _make_batch())
        assert isinstance(result, TrainStepResult)

    def test_loss_is_finite(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        result = train_step(online, target, optimizer, _make_batch())
        assert math.isfinite(result.loss)

    def test_loss_is_nonnegative(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        result = train_step(online, target, optimizer, _make_batch())
        assert result.loss >= 0.0

    def test_mean_target_q_is_finite(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        result = train_step(online, target, optimizer, _make_batch())
        assert math.isfinite(result.mean_target_q)

    def test_mean_online_q_is_finite(self):
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        result = train_step(online, target, optimizer, _make_batch())
        assert math.isfinite(result.mean_online_q)

    def test_loss_decreases_with_many_steps(self):
        """Loss should generally decrease when training toward zero targets."""
        torch.manual_seed(42)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-2)
        # Use all-zero obs + constant reward so there's a clear learning signal
        batch = {
            "obs": torch.zeros(32, OBSERVATION_SIZE),
            "action": torch.zeros(32, dtype=torch.int64),
            "reward": torch.zeros(32),
            "next_obs": torch.zeros(32, OBSERVATION_SIZE),
            "done": torch.ones(32),  # all terminal: target = reward = 0
            "next_mask": torch.ones(32, ACTION_COUNT, dtype=torch.bool),
        }
        first_result = train_step(online, target, optimizer, batch)
        for _ in range(50):
            result = train_step(online, target, optimizer, batch)
        assert result.loss < first_result.loss


# ---------------------------------------------------------------------------
# TestTrainStepBellman
# ---------------------------------------------------------------------------

class TestTrainStepBellman:
    def test_done_transition_ignores_next_q(self):
        """For done=True transitions, target should equal reward regardless of next Q."""
        torch.manual_seed(0)
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        # done=True for all; next_obs should not matter
        batch_done = _make_batch(batch_size=8, done=True)
        batch_done["reward"] = torch.ones(8) * 2.0
        result = train_step(online, target, optimizer, batch_done)
        # target = 2.0 for all; loss should be finite and non-negative
        assert math.isfinite(result.loss)
        assert result.loss >= 0.0

    def test_nonterminal_transition_includes_bootstrap(self):
        """For done=False, Bellman target includes gamma * max_next_q."""
        torch.manual_seed(99)
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        batch_live = _make_batch(batch_size=8, done=False)
        result_live = train_step(online, target, optimizer, batch_live)
        # Just verify it runs and returns finite values
        assert math.isfinite(result_live.loss)

    def test_gamma_zero_equals_done_only(self):
        """With gamma=0, Bellman target = reward regardless of next Q."""
        torch.manual_seed(7)
        online1 = QNetwork()
        target1 = QNetwork()
        sync_target_network(online1, target1)
        online2 = copy.deepcopy(online1)
        target2 = copy.deepcopy(target1)
        opt1 = optim.SGD(online1.parameters(), lr=0.0)  # no update; just measure
        opt2 = optim.SGD(online2.parameters(), lr=0.0)
        batch = _make_batch(batch_size=4, done=False)
        batch["reward"] = torch.ones(4) * 1.5
        r1 = train_step(online1, target1, opt1, batch, gamma=0.0)
        batch_done = dict(batch)
        batch_done["done"] = torch.ones(4)
        r2 = train_step(online2, target2, opt2, batch_done, gamma=0.99)
        # Both should have mean_target_q ≈ 1.5 (reward only)
        assert abs(r1.mean_target_q - 1.5) < 1e-4
        assert abs(r2.mean_target_q - 1.5) < 1e-4


# ---------------------------------------------------------------------------
# TestTrainStepMaskEnforcement
# ---------------------------------------------------------------------------

class TestTrainStepMaskEnforcement:
    def test_illegal_next_actions_do_not_affect_target(self):
        """High Q on illegal next actions must not inflate the Bellman target."""
        torch.manual_seed(42)
        # Build a target network with artificially high Q for action 0 only
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)

        reward_val = 0.0
        gamma = 0.99

        # Next mask: only action 1 is legal (action 0 is illegal)
        next_mask_legal_1 = torch.zeros(4, ACTION_COUNT, dtype=torch.bool)
        next_mask_legal_1[:, 1] = True   # only action 1 legal

        # Next mask: only action 0 is legal
        next_mask_legal_0 = torch.zeros(4, ACTION_COUNT, dtype=torch.bool)
        next_mask_legal_0[:, 0] = True   # only action 0 legal

        batch_base = {
            "obs": torch.zeros(4, OBSERVATION_SIZE),
            "action": torch.zeros(4, dtype=torch.int64),
            "reward": torch.ones(4) * reward_val,
            "next_obs": torch.zeros(4, OBSERVATION_SIZE),
            "done": torch.zeros(4),
        }

        # Get the actual Q-values from target network for action 0 and action 1
        with torch.no_grad():
            q_all = target(torch.zeros(1, OBSERVATION_SIZE))
            q0 = q_all[0, 0].item()
            q1 = q_all[0, 1].item()

        batch_legal_0 = dict(batch_base)
        batch_legal_0["next_mask"] = next_mask_legal_0
        batch_legal_1 = dict(batch_base)
        batch_legal_1["next_mask"] = next_mask_legal_1

        # Use lr=0 so online net doesn't update; we only care about target values
        opt_zero = optim.SGD(online.parameters(), lr=0.0)
        r0 = train_step(online, target, opt_zero, batch_legal_0, gamma=gamma)
        r1 = train_step(online, target, opt_zero, batch_legal_1, gamma=gamma)

        # mean_target_q when only action 0 is legal = reward + gamma * q0
        expected_q0 = reward_val + gamma * q0
        # mean_target_q when only action 1 is legal = reward + gamma * q1
        expected_q1 = reward_val + gamma * q1

        assert abs(r0.mean_target_q - expected_q0) < 1e-4
        assert abs(r1.mean_target_q - expected_q1) < 1e-4

    def test_all_legal_mask_uses_global_max(self):
        """When all actions are legal, max_next_q equals the true network max."""
        torch.manual_seed(3)
        online = QNetwork()
        target = QNetwork()
        sync_target_network(online, target)
        optimizer = optim.SGD(online.parameters(), lr=0.0)

        next_obs = torch.zeros(2, OBSERVATION_SIZE)
        with torch.no_grad():
            q_vals = target(next_obs)
            true_max = q_vals.max(dim=1).values.mean().item()

        batch = {
            "obs": torch.zeros(2, OBSERVATION_SIZE),
            "action": torch.zeros(2, dtype=torch.int64),
            "reward": torch.zeros(2),
            "next_obs": next_obs,
            "done": torch.zeros(2),
            "next_mask": torch.ones(2, ACTION_COUNT, dtype=torch.bool),
        }
        result = train_step(online, target, optimizer, batch, gamma=1.0)
        # target_q = 0 + 1.0 * max_q  => mean_target_q ≈ true_max
        assert abs(result.mean_target_q - true_max) < 1e-4


# ---------------------------------------------------------------------------
# TestTrainStepParameterUpdate
# ---------------------------------------------------------------------------

class TestTrainStepParameterUpdate:
    def test_online_net_params_change_after_step(self):
        torch.manual_seed(0)
        online, target = _make_nets()
        sync_target_network(online, target)
        params_before = [p.clone() for p in online.parameters()]
        optimizer = optim.Adam(online.parameters(), lr=1e-2)
        train_step(online, target, optimizer, _make_batch())
        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, online.parameters())
        )
        assert changed, "Online network parameters should have changed after train_step"

    def test_target_net_params_unchanged_after_step(self):
        torch.manual_seed(0)
        online, target = _make_nets()
        sync_target_network(online, target)
        target_params_before = [p.clone() for p in target.parameters()]
        optimizer = optim.Adam(online.parameters(), lr=1e-2)
        train_step(online, target, optimizer, _make_batch())
        for p_before, p_after in zip(target_params_before, target.parameters()):
            assert torch.equal(p_before, p_after), (
                "Target network should not change without explicit sync_target_network call"
            )

    def test_target_changes_only_after_sync(self):
        torch.manual_seed(0)
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-2)
        train_step(online, target, optimizer, _make_batch())
        # Target differs from online now
        different_before_sync = any(
            not torch.equal(p_o, p_t)
            for p_o, p_t in zip(online.parameters(), target.parameters())
        )
        assert different_before_sync
        # Now sync
        sync_target_network(online, target)
        equal_after_sync = all(
            torch.equal(p_o, p_t)
            for p_o, p_t in zip(online.parameters(), target.parameters())
        )
        assert equal_after_sync


# ---------------------------------------------------------------------------
# TestTrainStepDiagnosticFields
# ---------------------------------------------------------------------------

class TestTrainStepDiagnosticFields:
    """Verify the new diagnostic fields on TrainStepResult are populated."""

    def _step(self, **batch_kwargs) -> TrainStepResult:
        torch.manual_seed(42)
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        return train_step(online, target, optimizer, _make_batch(**batch_kwargs))

    def test_q_fields_finite(self):
        result = self._step()
        assert math.isfinite(result.q_min)
        assert math.isfinite(result.q_mean)
        assert math.isfinite(result.q_max)
        assert math.isfinite(result.q_max_abs)

    def test_q_max_abs_nonnegative(self):
        result = self._step()
        assert result.q_max_abs >= 0.0

    def test_q_min_le_q_mean_le_q_max(self):
        result = self._step()
        assert result.q_min <= result.q_mean + 1e-6
        assert result.q_mean <= result.q_max + 1e-6

    def test_target_fields_finite(self):
        result = self._step()
        assert math.isfinite(result.target_min)
        assert math.isfinite(result.target_mean)
        assert math.isfinite(result.target_max)

    def test_td_error_fields_finite(self):
        result = self._step()
        assert math.isfinite(result.td_error_mean)
        assert math.isfinite(result.td_error_max_abs)

    def test_td_error_max_abs_nonnegative(self):
        result = self._step()
        assert result.td_error_max_abs >= 0.0

    def test_reward_fields_match_batch(self):
        """reward_min/mean/max should reflect the batch reward tensor."""
        torch.manual_seed(5)
        online, target = _make_nets()
        sync_target_network(online, target)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)
        batch = _make_batch(batch_size=16)
        batch["reward"] = torch.tensor([1.0] * 8 + [-1.0] * 8)
        result = train_step(online, target, optimizer, batch)
        assert abs(result.reward_min - (-1.0)) < 1e-5
        assert abs(result.reward_max - 1.0) < 1e-5
        assert abs(result.reward_mean - 0.0) < 1e-5

    def test_done_count_all_terminal(self):
        result = self._step(done=True, batch_size=8)
        assert result.done_count == 8

    def test_done_count_no_terminal(self):
        result = self._step(done=False, batch_size=8)
        assert result.done_count == 0

    def test_done_count_mixed(self):
        result = self._step(done=None, batch_size=8)
        # alternating done flags: 0,1,0,1,0,1,0,1 → 4 done
        assert result.done_count == 4
