# Unified RL platform: single entry for training, diagnosis and play
Status: implementing (P0 verified; P1 not started)
Owner: Ji Hua (decisions), Claude (design and implementation)
Branch: feature/unified-rl-platform (from feature/agent-ladder-preview bbc6e28)
Design/contract: documents/system/design/unified-rl-platform.md
## Scope and non-goals
Scope: one entry (Training 8790 first) covering experiments, training, Jobs, ladder, models, Lab and Play; Jobs subsystem for submitting GPU training/evaluation/diagnosis; convergence to one Runtime, one Lab and one Gameplay.
Non-goals: user accounts/TLS, rule changes, rewriting historical evidence, multi-GPU scheduling, production roster or champion changes.
## Expected behavior and compatibility
The running 8790 preview and all other existing services stay unchanged until each phase passes acceptance. Old API paths remain proxied until consumers migrate.
## Acceptance criteria
Per phase, see section 11 of the design.
## Implementation
P0 (2026-10-31):
- `231ecb7` splits `apps/web/src/views/AgentWorkspace.tsx` into `views/agent/{shared,Ladder,AgentDetail,CheckpointPage,Matches,Replay,StepAnalysis,Play}.tsx`. Code lines were moved unchanged (verified line-by-line); the shell re-exports `evidence`, `route`, `Participant` and `StepAnalysis` for existing importers.
- `c24f411` adds `PLATFORM_MODEL_CATALOGS=discover` (every top-level `catalog*.json` plus `experiments/*/*catalog*.json`), `Evidence.rankings()` (exact-SHA season appearances, never pooled) and `GET /api/v1/models`, `GET /api/v1/models/{sha}` with a separate `ranked` state and per-capability summary. Existing `/api/platform/models` and `/api/agents/*` are unchanged; default catalog selection is unchanged unless the variable is set.
## Verification commands and observed results
- `npm run typecheck`, `npm run test:ui` (55 passed), `npm run build`: passed before and after the split.
- Backend (documented command, see docs/WORKSPACES.md): 134 run, all passed, 1 optional skip; with `ARENA_REPLAY_TEST_ROOT` set, 0 skipped. An earlier run reported a `backend.test_mlflow` import error only because `training-dashboard` was missing from `PYTHONPATH`; the test imports the local adapter, not the MLflow package. Five new tests pass.
- Real data (`/home/jihua/quoridor-data`): discovery reads 11 catalogs and indexes 104 models, equal to the distinct catalog SHA count (default selection: 92). No identity/metadata conflicts. 19 models ranked; all 19 season SHAs are present in the index. Through the full app with MLflow: listing HTTP 200 in 5.4 s cold (run lineage dominates, as for `/api/platform/models`), detail 0.2 s with SHA verified, unknown/invalid SHA 404.
- Browser: isolated loopback stack (Platform 18793, Lab 18794, preview gateway image on 18795; all stopped afterwards). 13 ladder-related tests: 10 passed, 3 failed. The same 3 fail identically with the base-commit frontend in the same stack, so they predate P0: two `agent-preview.spec.mjs` cases still expect the pre-compact Agent detail layout (a standalone 训练记录 heading and 该类型无需训练 text), and one `replay-refinement.spec.mjs` layout check measures 790 px against a 768 px limit. To be updated in P1 together with the new shell.
## External artifacts and deployment impact
Evidence: `/home/jihua/quoridor-backups/unified-rl-platform-p0/` (`e2e/`, `e2e-baseline/`, logs, temporary gateway config). Nothing deployed; 8790 and all other services unchanged. Not pushed to GitHub.
## Deferred decisions
Classic Play feature retention (design section 12, item 5). All other section-12 items were decided on 2026-10-31 (design section 0).
