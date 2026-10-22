# Environments

**Last verified:** 2026-10-22 (Training dashboard/Decision Lab; other facts retain their original verification)

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
