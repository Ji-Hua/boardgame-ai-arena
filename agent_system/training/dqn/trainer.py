"""DQN training step, target network synchronization, and training diagnostics.

Public API:
    sync_target_network(online, target) -> None
        Hard-copy all parameters from online network to target network.

    train_step(online_net, target_net, optimizer, batch, gamma,
               grad_clip_norm, algorithm) -> TrainStepResult
        One minibatch Bellman update on the online network.

    TrainStepResult
        Dataclass with loss and diagnostic scalars.

Algorithm modes
---------------
``algorithm="dqn"`` (default) — Vanilla DQN Bellman target:
    target[i] = reward[i] + gamma * max_{legal j} Q_target(next_obs[i])[j]  if not done[i]
    target[i] = reward[i]                                                      if done[i]

``algorithm="double_dqn"`` — Double DQN Bellman target:
    next_action[i] = argmax_{legal j} Q_online(next_obs[i])[j]
    target[i] = reward[i] + gamma * Q_target(next_obs[i])[next_action[i]]  if not done[i]
    target[i] = reward[i]                                                    if done[i]

In both modes:
- Illegal next actions are excluded from argmax via -inf masking.
- done=True disables bootstrap (avoids -inf*0=NaN corner cases).

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

# Supported algorithm mode strings.
_VALID_ALGORITHMS: frozenset[str] = frozenset({"dqn", "double_dqn"})

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
    # Gradient norm before clipping (None if clipping was not applied)
    grad_norm: float | None = None


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
    grad_clip_norm: float | None = None,
    algorithm: str = "dqn",
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
    grad_clip_norm:
        If not None and > 0, clips gradient norm to this value before the
        optimizer step.  Returns the pre-clip norm in TrainStepResult.
    algorithm:
        ``"dqn"`` (default) or ``"double_dqn"``.
        Controls how the Bellman bootstrap action is selected.

    Returns
    -------
    TrainStepResult with loss and mean Q diagnostics.

    Algorithm
    ---------
    Common:
    1. Q_online(obs) → shape (B, action_count)
    2. q_taken = Q_online(obs)[b, action[b]]  → shape (B,)

    DQN (algorithm="dqn"):
    3. Q_target(next_obs) → shape (B, action_count)   [no_grad]
    4. Mask illegal next actions: Q_target[~next_mask] = -inf
    5. max_next_q = max_j Q_target(next_obs)[b, j]   → shape (B,)

    Double DQN (algorithm="double_dqn"):
    3. Q_online(next_obs) → shape (B, action_count)   [no_grad]
    4. Mask illegal next actions: Q_online_next[~next_mask] = -inf
    5. next_action = argmax_j Q_online_next[b, j]     → shape (B,)  int64
    6. Q_target(next_obs) → shape (B, action_count)   [no_grad]
    7. max_next_q = Q_target[b, next_action[b]]       → shape (B,)

    Both:
    N-1. max_next_q_bootstrap = 0 if done else max_next_q  (avoids -inf*0=NaN)
    N.   target_q = reward + gamma * max_next_q_bootstrap
    N+1. loss = SmoothL1(q_taken, target_q.detach())
    N+2. optimizer.zero_grad(); loss.backward(); [clip]; optimizer.step()
    """
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Expected one of: {sorted(_VALID_ALGORITHMS)}"
        )

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
        illegal = ~next_mask                                   # (B, action_count)

        if algorithm == "double_dqn":
            # --- Double DQN: online net selects next action ---
            q_online_next = online_net(next_obs)               # (B, action_count)
            q_online_next_masked = q_online_next.masked_fill(illegal, _NEG_INF)
            next_action = q_online_next_masked.argmax(dim=1)   # (B,)  int64

            # --- Target net evaluates the online-selected action ---
            q_all_target = target_net(next_obs)                # (B, action_count)
            max_next_q = q_all_target.gather(
                1, next_action.unsqueeze(1)
            ).squeeze(1)                                       # (B,)
        else:
            # --- Vanilla DQN: target net selects and evaluates ---
            q_all_target = target_net(next_obs)                # (B, action_count)
            q_all_target_masked = q_all_target.masked_fill(illegal, _NEG_INF)
            max_next_q = q_all_target_masked.max(dim=1).values # (B,)

        # For terminal transitions (done=1), bootstrap must be zero.
        # Use torch.where to avoid -inf * 0 = NaN when next_mask is all-False.
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
    grad_norm: float | None = None
    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), grad_clip_norm).item()
        )
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
        grad_norm=grad_norm,
    )
