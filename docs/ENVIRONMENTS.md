# Environments

**Last verified:** 2026-10-22 (P3-A cross-host play, P1/P2 and protected service health; other facts retain their original verification)

## Roles

| Environment | Location | Role |
|---|---|---|
| Dev | `/home/jihua/workspace/quoridor` | Development, review, tests, integration, and documentation. |
| Training | `192.168.68.88:/home/jihua/workspace/boardgame-ai-arena` | GPU training, benchmark execution, MLflow, model management, dashboard, and Decision Lab. |

Git is the shared source of truth. Do not copy source trees between these paths.
Use named branches and Git fetch/push/merge or PRs.

## Dev facts

- Current local GPU at verification: NVIDIA Quadro M1000M, 4 GiB.
- `/usr/bin/python3` is Python 3.12.3; it did not have PyTorch installed at
  verification. Use the project environment rather than assuming GPU training
  is available here.
- Dev is appropriate for focused tests, code review, integration, and
  documentation.

## Training facts

- Python: `/home/jihua/workspace/boardgame-ai-arena/.venv/bin/python`
  (Python 3.12.12 at verification).
- PyTorch: `2.11.0+cu126`; CUDA was available.
- GPU: NVIDIA GeForce RTX 3080 Ti, 12 GiB.
- Persistent non-Git data root: `/home/jihua/quoridor-data/`.
- Training source must use a named Git commit before any experiment. New
  outputs belong in a new external run directory, never inside the repository.
- The original Training worktree remains on `codex/environment-deployment` and
  intentionally retains external experiment output. The clean synchronized
  worktree is `/home/jihua/workspace/boardgame-ai-arena-integration` on
  `integration/dev-training-20261021`; use it for integration validation and
  newly approved work until the branch is promoted.

## Services on Training

Dashboard and Decision Lab currently serve the isolated
`/home/jihua/workspace/boardgame-ai-arena-unified-dashboard` worktree on
`feature/unified-training-dashboard`. The original legacy artifacts mount and
external data root are unchanged and read-only. Decision Lab's existing catalog
is unchanged; only its source worktree selection and frontend startup fix changed.
See [deployment/rollback and acceptance](UNIFIED_DASHBOARD_20261022.md).

| Service | Address | Persistence / role |
|---|---|---|
| Training dashboard | `http://192.168.68.88:8741` | Dashboard for training, replays, and matrix results. |
| MLflow | `http://192.168.68.88:8742` | Tracking server and model registry. |
| Agent Service | `http://192.168.68.88:8743` | Pinned-model inference service. |
| Decision Lab | `http://192.168.68.88:8744` | Developer decision-analysis UI/API. |

### Isolated unified platform P1/P2 (additional, not replacement)

- Source: `/home/jihua/workspace/boardgame-ai-arena-platform`, branch
  `feature/unified-platform-p1-p2`, implementation `be5b200`, handoff `7ec538d`.
  Original/shared worktrees are not overwritten.
- Live entry `http://192.168.68.88:8751`; separate Lab listens only on
  `http://127.0.0.1:8754`, proxied through `/research-lab/` on 8751.
- The foreground launcher now runs under transient user systemd unit
  `quoridor-platform-p1-p2-preview.service`, restored after its earlier caller
  exited. It is not an installed reboot-autostart deployment. Do not start a
  duplicate while the current isolated ports are occupied.
- Uses the original Training venv and legacy artifact roots, existing MLflow,
  and original catalogs. Only the isolated Lab child receives an externally
  generated research catalog. No production roster, alias or Model Registry
  mutation is involved; 8740–8744 and 8764–8765 are protected launcher ports.
- 8751 platform APIs and existing 8741/8744/8743 health returned HTTP 200.
  Supplied Dev handoff remains checkout `69b495f…`, production Backend HTTP
  8764 → Training Agent Service 8743; no Dev deployment or remote revalidation
  was performed by this documentation task.
- Plain HTTP on a trusted LAN only, without auth/TLS. Research model flags are
  static eligibility, not proof that all models load, play or are deployed.
- Final browser evidence stays outside Git at
  `/home/jihua/quoridor-data/platform-p1-p2/acceptance/live-e2e-20261022-final`.

See [P1/P2 contracts, build/start/test commands and limitations](UNIFIED_PLATFORM_P1_P2.md).
The earlier 8741/8744 deployment record above remains separate and unchanged.

### Isolated unified Play P3-A (additional, not replacement)

- Entry `http://192.168.68.88:8752`; separate research Runtime on Training 8753;
  Dev Gameplay `http://192.168.68.102:8774` forwards only to that Runtime.
- Training worktree `/home/jihua/workspace/boardgame-ai-arena-play`, branch
  `feature/unified-platform-p3-a`; Dev `/home/jihua/workspace/quoridor-play-p3-a`,
  branch `preview/unified-platform-p3-a`, synchronized by Git.
- Training transient unit `quoridor-play-p3-a-preview-v3.service`: restart on
  failure, 2 GiB memory and 150% CPU. Dev container `quoridor-gameplay-p3-a`:
  one CPU, 768 MiB, read-only source/filesystem, restart policy `no`.
- Existing P1/P2 loopback Lab 8754 is reused; production 8744 is not replaced.
  Production Dev Backend 8764 still uses Training Agent Service 8743.
- The external service key lives under
  `/home/jihua/quoridor-data/platform-p3-a/private/`; approval snapshots and
  browser evidence live under that same external P3-A root, never in Git.
- Session-scoped CNNs are bounded, SHA-verified and capability-protected, but
  public/user authentication and TLS are absent. Games are in memory for at most
  four hours; restarts lose them. No pause/resume or persistent human replay.
- Final six real browser tests passed; C-resnet human P1/P2 games ended naturally
  after 36/39 plies. Protected production services remained reachable and
  checkpoint/catalog/registry/AgentSpec hashes unchanged.

See [P3-A versions, contracts, safeguards, evidence and operations](UNIFIED_PLATFORM_P3_A.md).

### Verified MLflow configuration

The authoritative deployment files are
[`model-management/compose.yml`](../model-management/compose.yml) and
[`model-management/Dockerfile`](../model-management/Dockerfile).

- MLflow container: `model-management-mlflow-1`, host port `8742`.
- Tracking URI: `http://192.168.68.88:8742` externally; host-local scripts use
  `http://127.0.0.1:8742`.
- Backend store: `sqlite:////data/mlflow.db` in the container, persisted as
  `/home/jihua/quoridor-data/mlflow/mlflow.db` on Training.
- Artifact store: MLflow `--serve-artifacts` with destination `/data/artifacts`,
  persisted as `/home/jihua/quoridor-data/mlflow/artifacts`.
- Persistence is a bind mount
  `/home/jihua/quoridor-data/mlflow:/data`; no MLflow named Docker volume is
  used. `/home/jihua/quoridor-data` is additionally mounted read-only at
  `/quoridor-data`.
- The Model Registry uses MLflow's built-in registry backed by the same SQLite
  store. `/home/jihua/quoridor-data/registry.json` is a separate Quoridor
  catalog mapping model IDs, MLflow versions/runs, and checkpoint hashes; it is
  not the MLflow registry itself.

See [model-management/README.md](../model-management/README.md) for operating
commands and [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) for provenance.
