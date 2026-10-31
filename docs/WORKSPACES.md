# Worktrees and workspace layout

**Last verified:** 2026-10-31. Active development branch: `feature/unified-rl-platform`.

Every removed or archived worktree had its commits contained in `feature/unified-rl-platform`; branches were kept, so any of them can be recreated with `git worktree add <path> <branch>`. Uncommitted changes were saved as patches before archiving.

## Training (192.168.68.88)

| Path | Branch | Why it stays |
|---|---|---|
| `~/worktrees/quoridor/unified-rl-platform` | `feature/unified-rl-platform` | Active development (unified platform, design `documents/system/design/unified-rl-platform.md`). |
| `~/worktrees/quoridor/agent-ladder-preview` | `feature/agent-ladder-preview` | Source of the running 8790 preview images. |
| `~/worktrees/quoridor/monorepo-migration-20261025` | `codex/monorepo-migration-20261025` | Source of the running 8740/8741 unified frontend release. |
| `~/workspace/boardgame-ai-arena` | `codex/environment-deployment` | Original checkout with intentional uncommitted experiment outputs; its `agent_system/training/artifacts` is mounted by the 8790 and 8741 platforms, and its venv runs the 8743 Agent Service. Do not modify. |
| `~/workspace/boardgame-ai-arena-unified-dashboard` | `feature/unified-training-dashboard` | Source of the systemd Decision Lab on 8744. |
| `~/worktrees/archive/boardgame-ai-arena-platform` | `feature/unified-platform-p1-p2` | Archived; uncommitted diff is identical to commit `bc49843`. |
| `~/worktrees/archive/quoridor-preserve-20261025` | `codex/preserve-training-fixes-20261025` | Archived; only an untracked `node_modules`. |

Removed (clean, unused): `boardgame-ai-arena-architecture-benchmark`, `boardgame-ai-arena-integration`, `boardgame-ai-arena-play`.
Patch: `~/quoridor-backups/workspace-tidy-20261031/`.

## Dev (192.168.68.102)

| Path | Branch | Why it stays |
|---|---|---|
| `~/workspace/quoridor` | `integration/dev-training-20261021` | Compose project directory of the running Classic backend 8764 and Agent container 8090; mounted by `quoridor-gameplay-p3-a`. |
| `~/workspace/quoridor-play-p3-a` | `preview/unified-platform-p3-a` | Mounted by running container `quoridor-gameplay-p3-a` (8774). |
| `~/worktrees/quoridor/unified-web-20261025` | `codex/unified-web-dev-20261025` | Source of the running 8765/8776 release. |
| `~/worktrees/archive/quoridor-unify-20261025` | `codex/unify-dev-training-20261025` | Archived with its unfinished merge of `bc49843` intact; that merge was completed on `codex/monorepo-migration-20261025`. |

Removed (clean, unused): `quoridor-architecture-benchmark`, `quoridor-platform-p1-p2`.
Patches: `~/quoridor-backups/workspace-tidy-20261031/`.

## Platform Jobs data (Training)

`/home/jihua/quoridor-data/jobs/jobs.sqlite` (records), `jobs/<id>/` (job.json, source snapshot, stdout.log), `jobs/runs/<id>/` (run directory, Training artifact root), `catalog-jobs.json` and `models/<sha>.pt` (registered checkpoints). The write token is `~/quoridor-releases/jobs/private/token`. Only one controller may use this data at a time.

Games: `/home/jihua/quoridor-data/games/games.sqlite`, written only by the gameplay service (container uid 1000). After P5, Dev holds a read-only mirror of `/home/jihua/quoridor-data` (same path, `quoridor-mirror.timer`, excluding `games/`, `mlflow/` and Job snapshots); Dev `games/` is owned by the Dev gameplay service. P5 cutover: `~/quoridor-releases/p5-cutover-379a943/` on Training. Old service retirement and rollback: `~/quoridor-releases/retire-legacy-20261031/` on Training and Dev.

## Tests in a worktree

- Frontend (repository root): `npm ci`, `npm run typecheck`, `npm run test:ui`, `npm run build`.
- Platform backend (repository root): `PYTHONPATH=training-dashboard:.:backend-server:engine ~/workspace/boardgame-ai-arena/.venv/bin/python -m unittest discover -s training-dashboard/backend -p "test_*.py"`. `training-dashboard` must be on the path: `test_mlflow.py` imports the local adapter `backend/mlflow.py` as `mlflow`, not the MLflow package. Add `ARENA_REPLAY_TEST_ROOT=/home/jihua/quoridor-data/experiments/cnn-architecture-20261021-213018/arena-v2` to run the optional real-replay test.
- Gameplay: `PYTHONPATH=training-dashboard:.:backend-server:engine ~/workspace/boardgame-ai-arena/.venv/bin/python -m unittest discover -s apps/gameplay/tests -t .`.
- Jobs: `PYTHONPATH=.:backend-server:engine ~/workspace/boardgame-ai-arena/.venv/bin/python -m unittest discover -s apps/jobs/tests -t .` and `-m unittest agent_system.training.tests.test_jobs`.
- Browser: `npm run build`, `JOBS_IMAGE=quoridor-train-gpu:39aba9c scripts/verify_stack.sh start` (also starts a Job controller on 127.0.0.1:18796; `jobs.spec.mjs` needs `E2E_PLATFORM_TOKEN` and runs a real GPU Job) (Platform 18793, Lab 18794, gateway 18795, gameplay 18797 with a throwaway games database in the evidence directory, all on loopback; never the live 8790), then in `apps/web` `E2E_BASE_URL=http://127.0.0.1:18795 E2E_RESULTS_DIR=<external dir> npx playwright test --config tests/e2e/playwright.config.mjs`, then `scripts/verify_stack.sh stop`.
