# Quoridor platform web

The single React application for the unified RL training, diagnosis and play platform
([design](../../documents/system/design/unified-rl-platform.md)). Run `npm ci`, `npm run typecheck`,
`npm run test:ui` and `npm run build` from the repository root; the root `package-lock.json` is the
dependency authority.

## Navigation

| Section | Routes | Data |
|---|---|---|
| 总览 | `/` | `/api/v1/models` summary, latest season, `/api/v1/research/runs`, `/api/v1/experiments` |
| 实验 | `/experiments`, `/experiments/:id` | `/api/v1/experiments` (reports, assessment, files, models by exact arm) |
| 训练 | `/training`, `/training/runs/:runId`, `/replays/:runId` | `/api/v1/research/runs*`; the run page keeps benchmarks, diagnosis and evaluation replays |
| 天梯 | `/ladder`, `/agents/:id`, `/games/:id`, `/arena` | `/api/v1/seasons*`, `/api/v1/research/arena/*` |
| 模型 | `/models`, `/models/:sha`, `/checkpoints/:sha` | `/api/v1/models*`; checkpoint page lists exact-SHA season matches and replays |
| 实验室 | `/lab` | `/api/v1/research/context`, `/api/v1/lab/*` |
| 对战 | `/play` | `/api/v1/play*` (gameplay service: durable games, player and watch tokens, published finished games) |

Only Ladder, Agent, Game and Play wait for the selected season roster. The shell lives in
`src/views/AgentWorkspace.tsx`; pages are in `src/views/agent/`. Shared board rendering is in
`packages/board-ui`. Classic WebSocket play and the external research Play were removed from this
build on 2026-10-31; the released 8741/8765 entries still serve them from their own branches.

## Verification

Unit tests use mocked APIs. Browser tests run against an isolated loopback stack, never the live
preview: `scripts/verify_stack.sh start`, then in `apps/web`
`E2E_BASE_URL=http://127.0.0.1:18795 E2E_RESULTS_DIR=<external dir> npx playwright test --config tests/e2e/playwright.config.mjs`,
then `scripts/verify_stack.sh stop`.
