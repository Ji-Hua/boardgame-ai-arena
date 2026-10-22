# Project Status

**Last verified:** 2026-10-22

## Unified research platform P1/P2 completed (isolated branch)

`feature/unified-platform-p1-p2` in the platform worktree, baseline `a8fc58d`,
has completed P1/P2 routed research navigation and exact checkpoint/replay/ply
handoff into an isolated same-origin Lab. Implementation and documentation are
committed on the named feature branch; no shared-baseline promotion
or production replacement is implied. Live HTTP 8751 + loopback Lab 8754 use a
foreground supervisor, not durable systemd restart management. Existing Training
services and the supplied unchanged Dev deployment remain separate.

Final real Playwright evidence: **3/3 passed in 75.572s**. Focused revalidation:
119 backend tests run (118 passed / 1 skipped), 38 Lab, 21 frontend Node and
18 React UI tests passed; typecheck/build passed on Node 22.22.0. The exact
C-resnet replay ply-7 import and explicitly clicked neural analysis returned
HTTP 200 with 121 legal policy-logit candidates. The live index has 92 unique
SHA identities with static capabilities, not 92 verified loads/deployments.

P3 session runtime/pinned-spec permissions/safe wall DTOs and P4 revision-safe
pause/isolation/React API wrappers are **not completed**. Play remains external;
Lab is a transition, not a full React Lab. There is no arbitrary checkpoint
session or persisted game pause, no auth/TLS and no shared source-list cache.
No Model Registry, champion alias or production roster mutation occurred.
See [full contracts, evidence, commands and handoff](UNIFIED_PLATFORM_P1_P2.md).

## Training dashboard integration

The isolated `feature/unified-training-dashboard` worktree integrates the shared
baseline and `feature/cnn-architecture-benchmark`. Legacy and MLflow experiments
now use one provider-neutral Runs/detail/metrics/evaluation/replay/Compare UI.
Training's Dashboard and Decision Lab deployment/acceptance facts, metric gaps,
test results and protected dirty-worktree boundaries are recorded in
[UNIFIED_DASHBOARD_20261022.md](UNIFIED_DASHBOARD_20261022.md).
This does not promote the feature branch into the shared integration baseline.

## Integrated baseline

The active shared integration branch is `integration/dev-training-20261021`.
It preserves both prior Dev work and the Training research branch:

| Item | Commit | Notes |
|---|---|---|
| Dev DQN dependency update | `9bd2aa2` | Added PyTorch dependency. |
| Shared Training baseline | `63ef2bb` | Training system and dashboard. |
| Training environment/deployment history | `04bd419` through `3cfb19f` | Dashboard, environment documentation, reproducible matrix, pinned inference, and MLflow registry. |
| Preserved research source | `9737338` | CNN research workflows, configs, tests, diagnostics, model-management helpers, and Decision Lab. |
| Dev integration | `7a43e7d`, `4996782` | Non-destructive merges of the Training baseline and research branch. |

The integration branch is the reviewable synchronization point. `feature/dqn`
and `codex/environment-deployment` remain intact and must not be rewritten.
A clean Training worktree at
`/home/jihua/workspace/boardgame-ai-arena-integration` checks out this same
branch, while the original Training worktree retains its external experiment
outputs. See [ENVIRONMENTS.md](ENVIRONMENTS.md) for the distinction.
See [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) for promotion steps.

## Implemented capabilities

- Rust rule engine with Python bindings; it remains authoritative for legal
  actions and state transitions.
- Backend/frontend gameplay, replay, and Agent Service.
- Agent definitions and Arena evaluation for Random, Greedy, Minimax, DQN, and
  registered checkpoint-backed agents.
- Vector-observation DQN training plus CNN observation, supervised
  policy-distillation, CNN research, benchmark, diagnostics, and Decision Lab
  tooling.
- Training dashboard, MLflow Tracking/Registry, an immutable SHA-addressed
  checkpoint catalog, and external experiment storage on Training.

## Research state

The frozen D2 policy-CNN checkpoint is stored outside Git and has SHA-256
`09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4`.
Historical D2 distillation data is external to Git and contains 124,493
training positions. Historical results are recorded in
[RESEARCH_LOG.md](RESEARCH_LOG.md); they are context, not a current benchmark.

The next architecture-comparison work must hold observation planes, legal mask,
D2 labels, cross-entropy objective, augmentation, data split, and evaluation
protocol constant. It must not combine architecture changes with PPO, DQN,
DAgger, MCTS, reward shaping, a value head, path features, or action-head
changes.

## Open risks and required follow-up

1. Training's worktree intentionally still has modified and untracked
   experiment outputs. They were not committed and must remain external.
2. Earlier commits tracked some files below `agent_system/training/artifacts/`.
   New artifacts are now ignored; do not normalize or delete historical files
   without a dedicated migration and data-retention decision.
3. The legacy D2 distillation dataset was all used for training. Create a new
   source-game/opening-group-held-out, teacher-labelled validation set before
   interpreting architecture generalization.
4. The integration branch needs normal review and a PR before becoming the
   branch used by both environments.
