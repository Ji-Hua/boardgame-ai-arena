# Agent workspace

Current preview navigation is Ladder and Play. Agent detail contains pair records and exact-SHA training runs; replay embeds per-ply analysis. Global training records remain at /training. See [preview release](../../docs/AGENT_LADDER_PREVIEW_20261025.md).

# Earlier unified web baseline
This is the single active React application. Run npm ci, npm run typecheck,
npm run test:ui and npm run build from the repository root. Root package-lock.json
is the workspace dependency authority; the migrated nested lock is historical.
Routes: Play (exact checkpoint and classic modes), Training, Arena, Replay,
Decision Lab and Models. Lab is native React; no iframe or external site is
required. Standalone old Lab remains available for recovery.
Shared board rendering and projection code lives in packages/board-ui. Classic
controllers/pages remain in frontend/src and are imported into this same bundle;
compatibility exports prevent a second implementation. Future relocation can
move those source files without changing behavior.
The gateway routes classic HTTP/WebSocket to the preserved Gameplay service;
research Play uses its separate capability-protected service. Lab snapshot
navigation never pauses, rewinds or mutates a running game.
