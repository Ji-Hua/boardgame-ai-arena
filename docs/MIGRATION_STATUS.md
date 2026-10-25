# Monorepo migration status
Reference host: Training (192.168.68.88).
Branch: codex/monorepo-migration-20261025.
Worktree: /home/jihua/worktrees/quoridor/monorepo-migration-20261025.

## Implemented in this stage
- Isolated P3 baseline merged with bc49843 Training fixes, preserving both document histories.
- Accepted design precedence and shared agent workflow.
- Explicit image-only Dev and Training Compose definitions and release validation tooling.
- Existing module locations and all original worktrees remain intact.

## Preservation
Training backup: /home/jihua/quoridor-backups/20261025T162124Z.
Full directories.tar includes source, Git metadata, ignored files and untracked content; verification.json records archive and source comparison.
Dev backup: /home/jihua/quoridor-backups/20261025T162123Z.
Dev: 18,382,295,040 bytes; SHA256 273496effdbeac3b3e9f8707f0c60ac3b08d8b1c1087ff6b535683a44f804cd9.
Training: 24,083,200,000 bytes; SHA256 82a3c24c686081f7c7ca1fc16813404b8d3ed2690be6eb5a792ca63788453987.
Both tar creation and full source comparison returned exit 0.
The external /home/jihua/quoridor-data tree is outside this source backup and remains untouched.
Backups are same-host recovery copies, not independent disaster-recovery storage.

## Deferred decisions
1. Training decision-lab/web/app.js dirty patch: preserve in backup; review semantics against newer Lab before porting.
2. Dev feature/backend_ai 176065d: preserved in full Git backup; historical porting decision pending.
3. Old v0/legacy repos: archive preserved, no automatic merger into current architecture.
4. MLflow version/database migration and GPU image dependency locks require separate validation.
5. Durable live sessions and coordinated service switching require acceptance and explicit rollout approval.

## Not delivered yet
No live service replacement, deleted directories, merged main, new GitHub repository, completed app-directory relocation, Jobs implementation or durable game recovery is claimed.

## Validation observed 2026-10-25
- Release validation: 7 passed; both Compose models rendered successfully without containers.
- Backend/Runtime/AgentSpec pytest: 80 passed, 1 optional external-model test skipped.
- Dashboard unittest: 122 run, 1 skipped, successful.
- React UI: 24 passed; TypeScript and production build passed.
- Python tests used the existing Training venv and installed engine; independent image builds and actual model-load acceptance are not covered.
