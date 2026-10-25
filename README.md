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

- **Engine** (`engine/`) — Rust rule engine with Python bindings (via PyO3/maturin)
- **Backend** (`backend-server/`) — FastAPI server providing REST + WebSocket APIs
- **Frontend** (`frontend/`) — React/TypeScript UI with live play and game replay
- **Agent System** (`agent_system/`) — canonical Agent Spec, runtime, training, evaluation, and decision analysis
- **Training Dashboard** (`training-dashboard/`) — training runs, metrics, replays, and Arena matrix results
- **Model Management** (`model-management/`) — MLflow experiment tracking and model registry
- **Decision Lab** (`decision-lab/`) — independent developer analysis of Greedy, Minimax, and CNN/MLP DQN decisions; its analysis logic lives in `agent_system/analysis/`

Decision Lab belongs to Quoridor's Agent System alongside Training and Arena.
It has its own process and UI, and can inspect edited positions without starting a live game,
training job, evaluation tournament, or model-registry service.

## Ports

| Service  | Port |
|----------|------|
| Frontend | 8765 |
| Backend  | 8764 |
| Training API / dashboard | 8740 / 8741 |
| MLflow | 8742 |
| Training-host Agent Service | 8743 |
| Decision Lab (UI + API) | 8744 |

## Unified research platform P1/P2

Completed on `feature/unified-platform-p1-p2` in the isolated platform worktree,
implementation `be5b200`, handoff `7ec538d`; not promoted to the shared baseline.
Live research entry: `http://192.168.68.88:8751`, with a separate loopback Lab
at `127.0.0.1:8754`. Existing production services and Dev remain unchanged.
The foreground two-process launcher now runs under a transient user systemd unit,
not an installed reboot-autostart deployment. P1/P2 provides routed Training, Arena, Replay and SHA-addressed Models
plus exact replay-position import into the existing Lab. Play is external and
Lab is a transition, not a full React rewrite. These statements describe the
P1/P2 branch; P3-A adds research sessions separately below. Persistent pause
remains future work; model selection does not mutate the Model Registry.
See [contracts, commands and final 3/3 real-browser acceptance](docs/UNIFIED_PLATFORM_P1_P2.md).

## Unified Play P3-A

`feature/unified-platform-p3-a` adds Training -> Checkpoint -> Play -> human
versus the exact approved CNN inside the research React shell. Preview:
`http://192.168.68.88:8752`, using separate Dev Gameplay 8774 and Training
research Runtime 8753. Rust remains the rule authority; loaded SHA, algorithm,
output semantics and Runtime identity are visible per game. Existing production
and P1/P2 services are preserved. Six real browser tests passed, including full
C-resnet games as human P1 and P2, recovery and previous replay-to-Lab behavior.
Sessions are in-memory and expire within four hours. This trusted-LAN preview
does not add user login/TLS, persistent human games, pause/resume or live Lab
analysis. See [verified source, operations, evidence and later-phase prerequisites](docs/UNIFIED_PLATFORM_P3_A.md).

## Decision analysis

From a repository environment with the Rust Python engine, FastAPI, Uvicorn, Pydantic v2,
and PyYAML installed:

```bash
bash decision-lab/start.sh
```

Open `http://localhost:8744`. Choose a sample position or edit the board, inspect action
scores and shortest paths, then compare Greedy with Minimax at 1–3 plies. Clicking a
Minimax candidate computes its exact branch value and exposes the principal variation
and alpha-beta trace. For DQN, choose a catalog checkpoint and inspect observation encoding,
all 209 Q outputs and legal masks, layer activations, weight slices, and input sensitivity.
Neural diagnostics use the existing PyTorch environment on CPU and read the model-management
catalog (`LAB_MODEL_CATALOG`, default `~/quoridor-data/catalog.json`). Export raw JSON data
to preserve the position, checkpoint hash, and diagnostic evidence.

- [Component guide and deployment](decision-lab/README.md)
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

---

## Local Development (without Docker)

### Backend

```bash
# Install Python dependencies
pip install fastapi uvicorn pydantic

# Install the engine wheel (requires Rust toolchain + maturin)
cd engine && maturin develop --release && cd ..

# Start backend on port 8764
PYTHONPATH="engine:backend-server" uvicorn backend.main:app --host 0.0.0.0 --port 8764
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server on port 8765
npm run dev
```

Open http://localhost:8765 in your browser.

---

## Docker (single container)

Build and run both services in one container:

```bash
# Build the image
docker build -t quoridor .

# Run the container
docker run -p 8764:8764 -p 8765:8765 quoridor
```

Open http://localhost:8765 in your browser.

The container:
- Builds the Rust engine wheel
- Builds the frontend (Vite production build)
- Starts backend (uvicorn, port 8764) and frontend (vite preview, port 8765) concurrently

---

## Replay

The frontend includes a replay mode that plays through a full recorded game step-by-step (0.5s per step). Click "Replay Full Game" from the main menu.

---

## License

Apache License 2.0. See LICENSE.

## Trademark Notice

Quoridor is a registered trademark of Gigamic. This is an independent, unofficial reimplementation for research and educational purposes.

## Monorepo migration

See [accepted design](documents/system/design/monorepo-deployment.md), [agent workflow](docs/AGENT_WORKFLOW.md), [migration status](docs/MIGRATION_STATUS.md), and [release operations](deploy/README.md).
