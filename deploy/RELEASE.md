# Platform releases (current)

Last verified: 2026-11-01. Design: [unified-rl-platform.md](../documents/system/design/unified-rl-platform.md).
Release history and evidence: [task record](../docs/tasks/20261031-unified-rl-platform.md).

## Topology

| Host | Piece | Runs as | Source of truth |
|---|---|---|---|
| Dev 192.168.68.102 | Entry gateway 8790 (SPA, `/mlflow/`, routes) and gameplay | compose project `quoridor-entry`, `~/quoridor-releases/p5-dev` (`release.env`, `compose.dev-entry.yaml`) | `deploy/compose.dev-entry.yaml` |
| Dev | Evidence mirror of `~/quoridor-data` every 10 min (and on demand after a registration) | user unit `quoridor-mirror.timer` / `.service` -> `sync_gameplay_mirror.sh` | `scripts/sync_gameplay_mirror.sh` |
| Training 192.168.68.88 | Platform API, Lab, API listener 8792, 8790 redirect | compose project `quoridor-agent-preview`, `~/quoridor-releases/agent-ladder-preview` (`release.env`, `compose.yaml`) | `deploy/compose.training-api.yaml` |
| Training | Job controller (host process, only writer of the data root) | user unit `quoridor-jobs.service`, source snapshot `~/quoridor-releases/jobs/src-<commit>` | `apps/jobs` |
| Training | MLflow 8742 | `model-management` compose | unchanged by releases |

Images: platform `quoridor-agent-preview:platform-<commit>` (`deploy/Dockerfile.research`, runs typecheck and UI
tests while building), gateway `quoridor-agent-preview-ui:platform-<commit>` (`deploy/Dockerfile.preview-ui`, after
`npm run build`). The gateway image renders both templates; every deployment must set `PLATFORM_UPSTREAM`,
`JOBS_UPSTREAM`, `GAMEPLAY_UPSTREAM` and `MLFLOW_UPSTREAM`.

## Making a release

1. Commit on the branch; tests per [WORKSPACES.md](../docs/WORKSPACES.md); `scripts/verify_stack.sh` for browser checks
   (with `VERIFY_JOBS_DATA` pointing at a throwaway directory: only one controller may use a data root).
2. On Training from a clean worktree at the commit: `npm run build`, then build the two images.
3. `cp -r deploy/release ~/quoridor-releases/<name>`, write `release.env` from `release.env.example`, add
   release-specific checks to `checks.d/` and one-off steps to `hooks.d/` if needed.
4. `~/quoridor-releases/<name>/deploy.sh`. It refuses while a Job is preparing or running, saves the current unit,
   compose files and env files (both hosts) in `previous/` once, exports the source snapshot, runs hooks, restarts the
   controller, recreates platform/gateway/redirect (and the Lab when `LAB_CATALOG` is set), restarts the Training
   gateway (nginx resolves upstreams once), updates the Dev entry, and runs `check.sh`.
5. `rollback.sh` in the same directory restores everything saved in `previous/`.

Writes need the platform token (`~/quoridor-releases/jobs/private/token`). Releases never promote a model,
change a roster, a champion alias or a season; promotion is a decision made in the UI (`promotions.jsonl`).
