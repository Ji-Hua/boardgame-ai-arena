# Quoridor AI

Quoridor AI is a Quoridor game platform and agent-research system with a Rust
rule engine, Python services, controlled evaluation, GPU training, model
management, and decision analysis.

## Project navigation

- [Agent guide](AGENTS.md) — required starting point for humans and AI agents.
- [Project status](docs/PROJECT_STATUS.md) — integrated baseline, research
  status, and open risks.
- [Environments](docs/ENVIRONMENTS.md) — Dev/Training responsibilities,
  verified services, GPU, MLflow, and storage.
- [Development workflow](docs/DEVELOPMENT_WORKFLOW.md) — Git, tests, handoff,
  synchronization, and model provenance.
- [Research log](docs/RESEARCH_LOG.md) — concise experiment and model index.
- [Completed unified platform P1/P2](docs/UNIFIED_PLATFORM_P1_P2.md) — isolated
  research UI, exact replay-to-Lab contracts, live acceptance and limitations.
- [Completed unified Play P3-A](docs/UNIFIED_PLATFORM_P3_A.md) — exact-checkpoint
  human games, bounded research sessions, cross-host preview and acceptance.
- [Architecture target](documents/system/design/quoridor-RL-vision.md) and
  [contribution governance](documents/system/governance/contribution-standard.md)
  remain authoritative design/governance sources.

## Architecture

One web entry for training, diagnosis and play ([design](documents/system/design/unified-rl-platform.md)):

- **Engine** (`engine/`) - Rust rule engine with Python bindings (PyO3/maturin); the only rule authority.
- **Web** (`apps/web`, shared board in `packages/board-ui`) - the single React app: overview, experiments,
  training, Jobs, ladder, models, Lab, play. `training-dashboard/frontend` is a compatibility symlink.
- **Platform API** (`training-dashboard/`) - `/api/v1` research API: model index, experiments, runs, seasons,
  diagnoses, Lab proxy (`platform_entry.py`).
- **Jobs** (`apps/jobs`) - host controller for training, evaluation, gate and diagnosis containers on the GPU;
  MLflow logging, verified model store, merged model index, promotion log.
- **Gameplay** (`apps/gameplay`) - durable human/agent games (`/api/v1/play`).
- **Agent System** (`agent_system/`) - AgentSpec, runtime, training, evaluation, probe suites, decision analysis.
- **Decision Lab** (`agent_system/analysis`, static assets in `decision-lab/web`) - position analysis behind `/api/v1/lab`.
- **Backend library** (`backend-server/`) - engine adapter and replay helpers imported by the services above.
- **Model Management** (`model-management/`) - MLflow server and catalog tooling.

## Services

| Host | Address | Service |
|---|---|---|
| Dev | `http://192.168.68.102:8790` | Single entry: SPA, gameplay, `/mlflow/`, routes to Training |
| Training | `192.168.68.88:8792` | API listener: platform, Lab, Jobs, promotions (for the Dev entry) |
| Training | `192.168.68.88:8742` | MLflow (also under `/mlflow/` in the entry) |
| Training | `192.168.68.88:8790`, `8740`, `8741`; Dev `8765` | Redirect to the Dev entry |

Releases and rollback: [deploy/RELEASE.md](deploy/RELEASE.md). Data lives in `/home/jihua/quoridor-data`
on Training (Dev keeps a read-only mirror). Status and history:
[unified platform task record](docs/tasks/20261031-unified-rl-platform.md); earlier previews:
[P1/P2](docs/UNIFIED_PLATFORM_P1_P2.md), [P3-A](docs/UNIFIED_PLATFORM_P3_A.md).

## Development

Workspace layout and every test command: [docs/WORKSPACES.md](docs/WORKSPACES.md). In short, at the
repository root: `npm ci`, `npm run typecheck`, `npm run test:ui`, `npm run build`; Python suites run with
`PYTHONPATH=training-dashboard:.:backend-server:engine`. `scripts/verify_stack.sh` starts an isolated
loopback stack for browser tests (set `VERIFY_JOBS_DATA` while the live Job controller runs).

## Decision analysis

Analysis documents:

- [Analysis architecture](documents/agent/analysis/decision-analysis.md)
- [Analysis API and trace contract](documents/agent/analysis/decision-analysis-interface.md)
- [Overall system architecture](documents/system/design/system-architecture.md)
- [Agent System architecture](documents/system/design/agent-system-design.md)
- [CNN training experiments and checkpoint diagnostics](documents/agent/training/cnn-repair.md)
- [Two-stage CNN selection and evaluation](documents/agent/training/cnn-hierarchy.md)
- [CNN Minimax D1/D2 demonstrations: 2,000 game episodes](documents/agent/training/cnn-minimax-demonstrations.md)
- [CNN policy distillation: shared trajectories and generalization](documents/agent/training/cnn-policy-distillation.md)
- [Frozen D2 CNN: DAgger, PPO, self-play and DDQN experiments](documents/agent/training/cnn-policy-rl.md)
- [AlphaZero-lite: calibrated value, PUCT and 2,000-game self-play benchmark](documents/agent/training/alphazero-lite.md)

## License

Apache License 2.0. See LICENSE.

## Trademark Notice

Quoridor is a registered trademark of Gigamic. This is an independent, unofficial reimplementation for research and educational purposes.

## Monorepo migration

See [accepted design](documents/system/design/monorepo-deployment.md), [agent workflow](docs/AGENT_WORKFLOW.md), [migration status](docs/MIGRATION_STATUS.md), and [release operations](deploy/README.md).

