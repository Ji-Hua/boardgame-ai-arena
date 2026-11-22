# Quoridor DQN Agent Target-State Blueprint

Author: Ji Hua
Created Date: 2026-04-25
Last Modified: 2026-04-26
Current Version: 2
Document Type: Design
Document Subtype: DQN Agent Target-State Blueprint
Document Status: Active
Document Authority Scope: Agent System / Training System
Document Purpose:
This document defines the intended completed state of the first DQN-based trainable agent capability in the Quoridor RL project. It describes what the system should be able to do after the DQN feature is implemented. It is a target-state blueprint only. It does not define detailed implementation tasks, file-level plans, or phased migration steps.

Changelog:
- v2 (2026-04-26): Added Sections 26-28 — DQN v0 training semantics, stabilized baseline configuration, and Phase 12C baseline results. DQN v0 is now complete.
- v1 (2026-04-25): Initial target-state blueprint.

---

## 1. Purpose

The DQN feature is the first formal trainable reinforcement learning capability in the Quoridor RL project.

Its purpose is not to produce the strongest possible Quoridor agent immediately.

Its purpose is to establish the first complete trainable-agent loop:

- Convert the Quoridor engine into a reinforcement learning environment.
- Train a neural Q-value agent.
- Save trained checkpoints.
- Load trained checkpoints as playable agents.
- Evaluate trained checkpoints through Arena.
- Compare trained DQN agents against existing baseline agents.
- Make the project capable of real RL experimentation.

After this feature is complete, Quoridor RL should no longer be only a rule engine, live game platform, agent host, or evaluation platform.

It should become a system that can actually train a model, preserve the trained result, and evaluate whether the model improved.

---

## 2. Target System State

After the DQN feature is implemented, the system should support the following end-to-end workflow:

1. Start a DQN training run.
2. Create a Quoridor RL environment backed by the existing Engine.
3. Let the DQN agent interact with the environment through repeated games.
4. Store transitions in a replay buffer.
5. Train a Q-network using Bellman targets.
6. Periodically update a target network.
7. Save model checkpoints.
8. Load a checkpoint as a normal Quoridor agent.
9. Run Arena evaluation against Random, Greedy, and Minimax agents.
10. Produce evaluation results showing whether the trained checkpoint improved.
11. Optionally expose the trained checkpoint through the existing agent runtime for live gameplay.

The completed DQN feature should prove that the project has a working train -> checkpoint -> evaluate -> run loop.

---

## 3. Design Intent

The DQN feature should be treated as the first practical bridge between the current agent platform and future advanced RL systems.

It should intentionally be simpler than PPO or AlphaZero.

The system should prioritize:

- clarity over sophistication
- reproducibility over raw strength
- simple training loop over advanced optimization
- stable environment contract over model complexity
- Arena compatibility over standalone scripts
- checkpoint usability over one-off experiments

The DQN agent is a learning baseline.

It is not the final architecture for AlphaZero.

It is not expected to outperform a strong Minimax agent immediately.

Its success is measured by whether the system can train, save, load, evaluate, and compare a learned policy.

---

## 4. Core Deliverable

The core deliverable is a DQN-based trainable Quoridor agent.

At minimum, the delivered system should contain:

- A Quoridor reinforcement learning environment.
- A fixed discrete action space.
- An observation encoder.
- A legal action mask.
- A Q-network.
- A replay buffer.
- A DQN trainer.
- A target network update mechanism.
- Epsilon-greedy exploration.
- Checkpoint save and load support.
- Arena integration for checkpoint evaluation.
- Runtime adapter support for using a trained checkpoint as an agent.

---

## 5. Reinforcement Learning Environment

The system should provide a Quoridor RL environment that wraps the existing game engine.

The environment should expose a simple RL-style interaction model:

- reset
- step
- legal actions
- action mask
- current observation
- terminal state detection

The environment must not implement its own Quoridor rules.

All rule validation, state transition, path preservation, wall legality, and win detection must continue to come from the Engine.

The RL environment is an adapter over the Engine, not a replacement for the Engine.

The environment should allow training code to repeatedly simulate games without going through the live frontend/backend gameplay path.

---

## 6. Environment Responsibility

The RL environment owns training-facing interaction semantics.

It is responsible for:

- creating a new game state at reset
- converting engine state into model observation
- converting discrete action ids into engine actions
- applying actions through the Engine
- returning reward after each step
- reporting whether the episode is finished
- exposing legal action masks
- returning useful debug information

The RL environment is not responsible for:

- defining Quoridor legality
- directly mutating game state outside Engine rules
- managing live rooms
- serving frontend clients
- running Arena tournaments
- deciding deployment status

---

## 7. Action Space

The first DQN implementation should use a fixed discrete action space.

For a standard 9x9 Quoridor board, the action space should contain 209 possible actions:

- 81 pawn move target actions
- 64 horizontal wall placement actions
- 64 vertical wall placement actions

The action id mapping should be stable and versioned.

Suggested conceptual mapping:

- 0 to 80: MovePawn to square target
- 81 to 144: Place horizontal wall
- 145 to 208: Place vertical wall

Most actions will be illegal in most states.

Therefore, legal action masking is required.

The DQN model may output Q-values for all 209 actions, but action selection must only consider legal actions.

---

## 8. Legal Action Mask

The DQN feature must include a legal action mask.

The legal action mask should be generated from Engine-provided legal actions.

The mask should indicate which of the 209 action ids are currently legal.

During training, epsilon-greedy exploration must sample only from legal actions.

During greedy action selection, the agent must select the highest-Q legal action, not the highest-Q action overall.

During Arena and runtime execution, the same masking rule must apply.

A correctly implemented DQN agent should not submit illegal actions during normal operation.

Illegal action count should be tracked as a diagnostic metric and should normally be zero.

---

## 9. Observation Model

The first DQN implementation should use a simple, stable observation representation.

The observation does not need to be perfect.

It must be:

- deterministic
- fixed-shape
- compatible with neural network input
- sufficient to represent the current game state
- versioned so checkpoints can be loaded safely later

A reasonable first representation is a tensor-style board encoding.

The observation should include at least:

- current player pawn position
- opponent pawn position
- horizontal wall occupancy
- vertical wall occupancy
- current player remaining walls
- opponent remaining walls
- side-to-move information
- optionally shortest-path information

The first version should avoid overly complex feature engineering.

The purpose is to make DQN trainable and debuggable, not to design the final best representation.

---

## 10. Reward Model

The first reward model should be simple.

The minimal reward structure should include:

- win reward
- loss penalty
- small step penalty

A possible initial design:

- win: positive terminal reward
- loss: negative terminal reward
- each non-terminal step: small negative reward

The system may optionally include light shaping based on shortest-path progress.

However, reward shaping must remain simple in the first version.

The first DQN feature should avoid complicated handcrafted rewards that make it difficult to understand whether learning is actually working.

---

## 11. DQN Agent Behavior

The DQN agent should learn an action-value function.

Given an observation, the Q-network outputs one Q-value per discrete action.

At decision time:

1. Encode the current game state into an observation.
2. Run the Q-network.
3. Apply the legal action mask.
4. Choose an action using epsilon-greedy exploration during training.
5. Choose the highest-Q legal action during evaluation or deterministic runtime.
6. Decode the selected action id into an Engine action.
7. Submit the action through the normal environment or runtime path.

The trained DQN agent should be usable as a normal Quoridor agent after checkpoint loading.

---

## 12. Training Loop

The DQN training loop should support repeated game simulation.

The training loop should:

- initialize environment
- initialize online Q-network
- initialize target Q-network
- initialize replay buffer
- run episodes
- select legal actions using epsilon-greedy
- collect transitions
- sample mini-batches from replay buffer
- compute Bellman targets
- update the online Q-network
- periodically update the target network
- periodically save checkpoints
- periodically run evaluation

The trainer should be usable from a command-line entry point.

The first version does not need distributed training.

The first version does not need GPU-specific optimization.

The first version should be able to run locally for small-scale experiments.

---

## 13. Replay Buffer

The replay buffer stores training transitions.

Each transition should contain:

- observation
- action id
- reward
- next observation
- done flag
- legal action mask or next legal action mask if needed
- optional metadata for debugging

The replay buffer enables the model to train from sampled historical experience rather than only from the most recent transition.

The first version may use a simple fixed-capacity replay buffer.

Prioritized replay is not required for the initial DQN feature.

---

## 14. Target Network

The DQN implementation should use a target network.

The online network is updated by gradient descent.

The target network is used to compute stable Bellman targets.

The target network should be periodically synchronized from the online network.

This is a core part of stable DQN training and should be included in the first formal implementation.

---

## 15. Exploration

The DQN agent should use epsilon-greedy exploration during training.

The exploration policy should:

- with probability epsilon, choose a random legal action
- otherwise, choose the highest-Q legal action

Epsilon should decay over training.

The first version should make epsilon configuration explicit.

Evaluation should use deterministic greedy action selection unless otherwise configured.

---

## 16. Checkpoint Capability

The DQN feature must save checkpoints.

A checkpoint should contain enough information to reload the trained agent safely.

At minimum, a checkpoint should include:

- checkpoint id
- agent id
- training step
- model weights
- optimizer state if training is to resume
- observation version
- action space version
- model architecture config
- training config
- created timestamp

The checkpoint must not be only a temporary training artifact.

It should be a first-class agent realization that can be evaluated and potentially used in runtime.

---

## 17. Checkpoint Loading

The system should support loading a saved DQN checkpoint as an executable agent.

A loaded checkpoint agent should be able to:

- receive a game state
- encode the observation
- run the Q-network
- apply legal action masking
- choose an action
- return a valid Quoridor action

This loaded agent should be usable by Arena and, eventually, by the live agent service.

---

## 18. Arena Integration

The DQN feature is incomplete unless trained checkpoints can be evaluated through Arena.

Arena should be able to run DQN checkpoint agents against existing baseline agents.

The first required opponents should include:

- Random
- Greedy
- Minimax depth 1
- Minimax depth 2 if performance allows

Arena evaluation should support side-swapped matches.

The basic evaluation output should include:

- win rate
- number of games
- side-specific win rate
- average game length
- illegal action count
- checkpoint id
- opponent id
- seed information if available

The DQN feature should make it possible to answer:

- Is the trained DQN better than random?
- Does the checkpoint improve over earlier checkpoints?
- Does performance change as training continues?
- Does the agent behave differently from heuristic agents?

---

## 19. Runtime Integration

The first DQN target state should support at least a clean path toward runtime usage.

Ideally, a trained DQN checkpoint can be exposed as a normal agent in the existing agent runtime.

The live gameplay system should not need to understand the details of DQN training.

From the runtime perspective, the DQN checkpoint agent should behave like any other agent:

- receive state
- choose action
- return action

If full runtime integration is too much for the first implementation, the design should still ensure that the checkpoint agent interface is compatible with future runtime exposure.

---

## 20. Relationship to Existing Agents

The DQN agent should coexist with existing non-trainable agents.

Existing agents such as Random, Greedy, and Minimax remain important as baselines.

The DQN feature should not replace them.

Instead, it should use them as:

- training opponents
- evaluation opponents
- regression baselines
- strength references

A successful DQN implementation should make the existing baseline agents more valuable because they become the first meaningful benchmark ladder for learned agents.

---

## 21. Relationship to PPO and AlphaZero

The DQN feature is not the final RL architecture.

It is an intermediate learning and system-validation step.

Compared with PPO, DQN is easier to reason about because it directly learns Q-values and uses Bellman updates.

Compared with AlphaZero, DQN avoids MCTS, policy-value self-play search, and complex candidate promotion loops.

DQN should establish the lower-level infrastructure that later PPO or AlphaZero can reuse:

- environment interface
- observation encoding
- action encoding
- legal action masking
- training entry point
- checkpoint format
- Arena checkpoint evaluation

The system should avoid building DQN in a way that blocks future PPO or AlphaZero work.

---

## 22. Success Criteria

The DQN feature should be considered successful if the system can demonstrate the following:

1. A DQN training run can start from scratch.
2. The training loop can complete multiple episodes.
3. The agent can select only legal actions during training and evaluation.
4. The replay buffer collects valid transitions.
5. The Q-network can be updated using Bellman targets.
6. The target network updates correctly.
7. Checkpoints can be saved.
8. Checkpoints can be loaded.
9. A loaded checkpoint can act as a Quoridor agent.
10. Arena can evaluate a DQN checkpoint against baseline agents.
11. Training progress can be inspected through checkpoint comparison.
12. A trained checkpoint performs measurably differently from an untrained checkpoint.
13. The trained checkpoint can beat Random at a rate above chance after sufficient training.
14. The system produces enough logs or metrics to debug training behavior.

The first success target should not be beating Minimax.

The first success target should be proving that learning happens.

---

## 23. Non-Goals

The first DQN feature should not attempt to complete all future RL ambitions.

The following are non-goals for the initial DQN target state:

- AlphaZero implementation
- MCTS-guided training
- PPO implementation
- distributed self-play
- large-scale training infrastructure
- automatic model promotion
- production-grade deployment
- advanced evaluation dashboard
- sophisticated curriculum learning
- prioritized replay
- dueling DQN
- double DQN unless added deliberately
- perfect reward shaping
- strongest possible Quoridor performance

These may be added later.

The first DQN feature should remain focused on establishing the first working trainable-agent loop.

---

## 24. Expected System Shape After Completion

After the DQN feature is complete, the project should feel different.

Before this feature, the project has agents, live play, and evaluation, but no real trainable RL loop.

After this feature, the project should have:

- an RL environment backed by the Quoridor Engine
- a neural agent that can train from experience
- checkpoints representing learned agents
- Arena evaluation for learned checkpoints
- a path from training output to runtime agent behavior
- a foundation for more advanced RL methods

The completed system should make it possible to run a simple experiment such as:

Train DQN for N episodes, save checkpoints every K steps, evaluate each checkpoint against Random and Greedy, and observe whether the checkpoint improves.

This is the key transformation.

---

## 25. Final Summary

The DQN feature should turn Quoridor RL into a genuine reinforcement learning project.

Its central contribution is not a perfect agent.

Its central contribution is the first complete trainable-agent loop:

Environment -> Experience -> Bellman Update -> Checkpoint -> Arena Evaluation -> Runtime Agent

Once this loop exists, the project can begin making real progress toward stronger RL agents, PPO, MCTS, and eventually AlphaZero-style systems.

---

## 26. DQN v0 Training Semantics

*Added v2, 2026-04-26. Documents the implemented and validated deferred-push transition semantics required for correct DQN training.*

### Deferred-Push Pattern

The observation encoder (`dqn_obs_v1`) is **current-player-centric**: it always encodes the board from the perspective of whoever is to move next. This creates a semantic hazard: after the learner acts and the opponent responds, a naive encoding of the resulting state would produce the board from the opponent's view, not the learner's.

The correct transition pattern is:

```
learner_obs  →  learner_action  →  opponent response  →  learner_next_obs  →  terminal / reward
```

`learner_next_obs` must be re-encoded from the **learner's perspective** after the opponent has moved. This is implemented as a **deferred push**: the replay buffer transition is not pushed immediately after the learner's action; it is deferred until the opponent responds (or the episode terminates), at which point `next_obs` and `next_legal_mask` are both computed from the learner's view.

### Required Invariants

- All replay buffer transitions must be **learner-centric**.
- `next_obs` must be encoded from the **learner's** perspective, not the perspective of the player to move after the learner.
- `next_legal_mask` must correspond to the **learner's** next decision state.
- Opponent-only actions must **not** be stored as learner transitions.
- When the opponent wins (terminal state after opponent's move), the learner receives reward **= −1**.
- When the learner wins (terminal state after learner's move), the learner receives reward **= +1**.
- Non-terminal steps produce reward **= 0**.
- Legal action masking is **mandatory** during epsilon-greedy exploration (training) and greedy selection (evaluation). Never select over all 209 actions without masking.

### Known Bugs Fixed in Phase 8

1. **Missing negative terminal reward**: opponent wins produced reward 0 for the learner instead of −1. Fixed via deferred-push terminal handling.
2. **Wrong perspective on next_obs**: `next_obs` was encoded from the opponent's current-player perspective. Fixed by re-encoding from the learner's fixed perspective after opponent response.

---

## 27. DQN v0 Stabilized Baseline Configuration

*Added v2, 2026-04-26. Records hyperparameters validated through Phases 12A–12C.*

The following configuration is the recommended vanilla DQN baseline. It eliminates Q-value divergence at the 5000-episode scale and achieves stable learning.

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| `lr` | `1e-4` | 1e-3 caused catastrophic divergence (Phase 11) |
| `gamma` | `0.95` | 0.99 worsened divergence in Phase 11 |
| `grad_clip_norm` | `10.0` | Required; prevents Q-value explosion |
| `batch_size` | `128` | Fastest on RTX 3080 Ti; larger batches slower |
| `buffer_capacity` | `100000` | For 5000-episode runs |
| `warmup_size` | `5000` | Fill buffer before gradient updates begin |
| `epsilon_start` | `1.0` | Full random exploration at start |
| `epsilon_end` | `0.05` | 5% minimum exploration |
| `epsilon_decay_steps` | `250000` | For 5000-episode runs |
| `target_sync_interval` | `1000` | **Stable default**: best final performance without divergence |
| `target_sync_interval` | `2000` | **High-peak experimental**: best peak 17% but some post-peak decline |
| `eval_interval` | `500` | |
| `eval_games` | `100` | vs `random_legal` |
| `device` | `auto` | Auto-selects CUDA |
| `obs_version` | `dqn_obs_v1` | 292 features |

**Do not use `target_sync_interval=200`.** The Phase 12B run with sync=200 diverged at ep3200 and collapsed to 0% win rate at ep4500.

---

## 28. DQN v0 Baseline Results

*Added v2, 2026-04-26. Records the final Phase 12C baseline outcomes.*

### Summary

| Checkpoint | Path | Win rate vs random_legal | Episode | `target_sync_interval` |
|---|---|---:|---:|---:|
| Peak | `long_train_005_sync2000/checkpoints/ep03500_step1765553.pt` | **17%** | 3500 | 2000 |
| Stable final | `long_train_004_sync1000/checkpoints/ep05000_step2678737.pt` | **11%** | 5000 | 1000 |
| (Superseded) Phase 12B | `long_train_002/checkpoints/ep02500_step1277239.pt` | 11% | 2500 | 200 |

Canonical copies: `agent_system/training/artifacts/dqn/baselines/dqn_v0_peak.pt` and `dqn_v0_stable.pt`.

### Key Findings from Phase 12C

- Illegal action count: **0** in all major long runs (Phases 9, 12B, 12C).
- All three Phase 12C ablation runs (sync=500/1000/2000) completed 5000 episodes with **no divergence**, proving that increasing `target_sync_interval` from 200 fully eliminates the catastrophic Q-value divergence observed in Phase 12B.
- sync2000 achieved the highest single-evaluation score (17% at ep3500) but showed post-peak decline (9% at ep5000), likely due to overestimation bias accumulating over too-stale targets.
- sync1000 showed the most stable late-stage behavior: consistent improvement to 11% at ep5000 with avg_loss=0.0008.
- sync500 was the weakest (8% peak) but remained stable.

### Interpretation

The 17% peak is meaningful evidence of learning: without semantic bug fixes (Phase 8), win rate was effectively 0%. The agent clearly learns something. However, 17% against a fully random opponent is still weak; a simple Greedy agent would achieve much higher.

The next priority is **Double DQN** to address overestimation bias, with `target_sync_interval=1000` as the starting configuration.
