# PPO shortest-path reward-shaping benchmark

This experiment isolates potential-based reward shaping in the existing PPO +
Minimax-D2-imitation workflow. It does not change the CNN, policy head, PPO
objective, demonstration source, opponent, opening policy or hyperparameters.

## Controlled arms

All arms start from `cnn_policy_d2_2k_update012000` (`quoridor-cnn-policy`
v6, SHA256 `09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4`),
with a new optimizer and zero-initialized training-only critic. They use 4,096
games, the frozen Greedy opponent, 0–24 random legal prefix plies, seed 42,
the same rollout and batch RNG seeds, and the same D2 demonstration loss.

| Arm | alpha | beta | Potential |
|---|---:|---:|---|
| A | 0 | 0 | terminal reward control |
| B | 0.01 | 0 | `-alpha * d_self` |
| C | 0 | 0.01 | `beta * (d_opponent - d_self)` |
| D | 0.01 | 0.01 | sum of B and C |

For a learner whose identity is fixed across its own decision transitions,
`d_self` and `d_opponent` are queried from the Rust engine's authoritative
shortest-path calculation. The potential is

`Phi(s) = -alpha * d_self(s) + beta * (d_opponent(s) - d_self(s))`.

The added reward is `F(s,s') = gamma * Phi(s') - Phi(s)`, using PPO's existing
`gamma=0.99`. At a genuine goal terminal, `Phi(s')` is defined as zero. The
collector closes the opponent reply before forming a learner transition, so both
states have the same learner perspective. A time limit remains nonterminal:
the final critic bootstrap is retained and therefore carries the residual
potential instead of treating the cap as a loss or draw.

The final 4,096-game checkpoint is the prespecified primary comparison. Earlier
checkpoints receive development diagnostics but cannot be selected from the
independent holdout.

## Evidence and safeguards

Training records terminal reward separately from auxiliary reward, including
sum, absolute sum, squared sum, maximum magnitude and sign counts. It also
records path changes, pawn/wall action counts, PPO losses, KL, entropy,
advantages, outcomes, time limits, environment steps and CUDA event intervals.
The transition trace retains start/end path features and the potential residual
for every game.

Focused tests cover Rust shortest-path features, terminal potential zeroing,
learner-perspective symmetry, zero-coefficient behavior, GAE time-limit
bootstrapping and the numerical telescoping identity. For a truncated trajectory,
the test verifies that the discounted sum of shaping rewards plus
`gamma^T Phi(s_T)` equals `-Phi(s_0)`; genuine terminals have zero final
potential.

## Evaluation

Existing checkpoint diagnostics and the existing Arena are used. The fresh
holdout has independent four-ply openings, paired seat swaps and excludes
recorded training-state symmetry groups, development books and the D2 dataset.
Every final arm plays Greedy, Minimax D1, Minimax D2, the frozen D2 CNN and the
existing PPO + Imitation CNN. Terminal W/D/L is primary; shaped training reward
is never used to rank models. Report paired confidence intervals for B/C/D
against A, as well as repetitions, game length, walls, shortest-path behavior,
D2 agreement and missed immediate wins.

Artifacts, traces, checkpoints, Arena blocks and MLflow records are stored
outside Git in `/home/jihua/quoridor-data/experiments/cnn-ppo-reward-shaping-20261021`.

## 2026-10-21 result

All four 4,096-game arms completed. On the independent 100-opening-pair
Greedy holdout, A/B/C/D scored 0.590/0.615/0.570/0.555. The B-minus-A paired
estimate was +0.025, but its planned simultaneous 95% Hoeffding interval was
[-0.284, 0.334]; the experiment therefore does not establish a reliable
improvement. In the full existing Arena matrix, B was the strongest shaped arm
(aggregate score 0.483 versus A's 0.474), while C and D were lower (0.456 and
0.464). All final arms had zero missed immediate wins and zero invalid-action
or inference forfeits.

The observed mean absolute shaping reward per learner transition was 0.0176
for B, 0.0166 for C, and 0.0337 for D. The largest per-rollout values were
0.43, 0.41, and 0.83 respectively, so a subsequent study should compare a
clipped or normalized potential and use multiple training seeds before any
promotion decision.

The first D directory stopped at 1,088 games while an aggregation defect was
investigated: it summed rollout maxima in a reporting field. Replaying B/C
trajectories through the Rust path engine found valid paths, and the reward
fed to PPO was unaffected. The reporting aggregation was corrected; D was
rerun from scratch in `04-combined-retry`, while the interrupted directory was
preserved and excluded from evaluation and registration.
