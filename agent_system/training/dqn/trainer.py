"""DQN training step, target network synchronization, and training diagnostics.

Public API:
    sync_target_network(online, target) -> None
        Hard-copy all parameters from online network to target network.

    train_step(online_net, target_net, optimizer, batch, gamma) -> TrainStepResult
        One minibatch Bellman update on the online network.

    TrainStepResult
        Dataclass with loss and diagnostic scalars.

Bellman target:
    target[i] = reward[i] + gamma * max_{legal j} Q_target(next_obs[i])[j]  if not done[i]
    target[i] = reward[i]                                                      if done[i]

Illegal next actions are excluded by setting their Q-values to -inf before
taking the max.  This guarantees the Bellman bootstrap is computed only over
the legal next-action set.

Loss:
    Smooth L1 / Huber loss between Q_online(obs)[action] and target.
    This is more robust than MSE to occasional large Bellman errors during
    early training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from agent_system.training.dqn.model import QNetwork

# Sentinel for masking illegal Q-values before max.
_NEG_INF: float = -math.inf

# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

@dataclass
class TrainStepResult:
    """Diagnostics returned by a single DQN minibatch update.

    Attributes
    ----------
    loss:
        Scalar Huber loss for this batch (Python float).
    mean_target_q:
        Mean Bellman target value across the batch.
    mean_online_q:
        Mean online Q-value for the chosen actions.
    q_min, q_mean, q_max, q_max_abs:
        Min/mean/max/max-abs of online Q-values for chosen actions.
    target_min, target_mean, target_max:
        Min/mean/max of Bellman target values.
    td_error_mean, td_error_max_abs:
        Mean and max-abs of (q_taken - target_q) before backward pass.
    reward_min, reward_mean, reward_max:
        Reward statistics within this batch.
    done_count:
        Number of terminal (done=True) transitions in this batch.
    """

    loss: float
    mean_target_q: float
    mean_online_q: float
    # Per-batch Q-value statistics
    q_min: float = 0.0
    q_mean: float = 0.0
    q_max: float = 0.0
    q_max_abs: float = 0.0
    # Per-batch Bellman target statistics
    target_min: float = 0.0
    target_mean: float = 0.0
    target_max: float = 0.0
    # Per-batch TD error statistics
    td_error_mean: float = 0.0
    td_error_max_abs: float = 0.0
    # Per-batch reward statistics
    reward_min: float = 0.0
    reward_mean: float = 0.0
    reward_max: float = 0.0
    # Terminal transition count
    done_count: int = 0


# ---------------------------------------------------------------------------
# Target network synchronization
# ---------------------------------------------------------------------------

def sync_target_network(online: nn.Module, target: nn.Module) -> None:
    """Hard-copy all parameters from *online* to *target*.

    After this call ``target.state_dict()`` is identical to
    ``online.state_dict()``.  No gradient tracking is modified.

    Parameters
    ----------
    online:
        The network being trained (parameters are the source).
    target:
        The network used for stable Bellman bootstrapping (parameters are
        the destination).
    """
    target.load_state_dict(online.state_dict())


# ---------------------------------------------------------------------------
# One minibatch DQN update
# ---------------------------------------------------------------------------

def train_step(
    online_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    batch: dict[str, torch.Tensor],
    gamma: float = 0.99,
) -> TrainStepResult:
    """Perform one minibatch Bellman update on *online_net*.

    Parameters
    ----------
    online_net:
        The Q-network being trained.  Must be in training mode.
    target_net:
        The target Q-network used for stable bootstrapping.  Not updated here.
    optimizer:
        PyTorch optimizer attached to *online_net* parameters.
    batch:
        Dictionary produced by ``ReplayBuffer.sample()``.  Required keys:
        ``obs``, ``action``, ``reward``, ``next_obs``, ``done``, ``next_mask``.
    gamma:
        Discount factor.  Default 0.99.

    Returns
    -------
    TrainStepResult with loss and mean Q diagnostics.

    Algorithm
    ---------
    1. Q_online(obs) → shape (B, action_count)
    2. q_taken = Q_online(obs)[b, action[b]]  → shape (B,)
    3. Q_target(next_obs) → shape (B, action_count)   [no_grad]
    4. Mask illegal next actions: Q_target[~next_mask] = -inf
    5. max_next_q = max_j Q_target(next_obs)[b, j]   → shape (B,)
    6. target = reward + gamma * max_next_q * (1 - done)
    7. loss = SmoothL1(q_taken, target.detach())
    8. optimizer.zero_grad(); loss.backward(); optimizer.step()
    """
    obs = batch["obs"]                  # (B, obs_size)
    action = batch["action"]            # (B,)  int64
    reward = batch["reward"]            # (B,)  float32
    next_obs = batch["next_obs"]        # (B, obs_size)
    done = batch["done"]                # (B,)  float32  0 or 1
    next_mask = batch["next_mask"]      # (B, action_count)  bool

    # ---- Online Q for chosen actions ----
    online_net.train()
    q_all_online = online_net(obs)                              # (B, action_count)
    q_taken = q_all_online.gather(1, action.unsqueeze(1)).squeeze(1)  # (B,)

    # ---- Target Q for next state ----
    with torch.no_grad():
        q_all_target = target_net(next_obs)                    # (B, action_count)
        # Mask illegal next actions: set Q to -inf where mask is False.
        illegal = ~next_mask                                   # (B, action_count)
        q_all_target = q_all_target.masked_fill(illegal, _NEG_INF)
        max_next_q = q_all_target.max(dim=1).values            # (B,)
        # For terminal transitions (done=1), max_next_q must not contribute.
        # Use torch.where to avoid -inf * 0 = NaN when next_mask is all-False
        # at terminal states.
        max_next_q_bootstrap = torch.where(
            done.bool(), torch.zeros_like(max_next_q), max_next_q
        )
        target_q = reward + gamma * max_next_q_bootstrap        # (B,)

    # ---- Huber loss ----
    loss = nn.functional.smooth_l1_loss(q_taken, target_q)

    # ---- TD errors for diagnostics (before gradient update) ----
    td_errors = (q_taken - target_q).detach()

    # ---- Gradient update ----
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return TrainStepResult(
        loss=float(loss.item()),
        mean_target_q=float(target_q.mean().item()),
        mean_online_q=float(q_taken.detach().mean().item()),
        q_min=float(q_taken.detach().min().item()),
        q_mean=float(q_taken.detach().mean().item()),
        q_max=float(q_taken.detach().max().item()),
        q_max_abs=float(q_taken.detach().abs().max().item()),
        target_min=float(target_q.min().item()),
        target_mean=float(target_q.mean().item()),
        target_max=float(target_q.max().item()),
        td_error_mean=float(td_errors.mean().item()),
        td_error_max_abs=float(td_errors.abs().max().item()),
        reward_min=float(reward.min().item()),
        reward_mean=float(reward.mean().item()),
        reward_max=float(reward.max().item()),
        done_count=int(done.sum().item()),
    )
