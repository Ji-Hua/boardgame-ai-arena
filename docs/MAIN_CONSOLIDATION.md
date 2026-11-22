# Main consolidation — Quoridor only

2026-11-22. Main is the sole integration baseline. Dev hosts entry/gameplay; Training hosts research, Lab, Jobs and MLflow. Both formal source checkouts use `/home/jihua/workspace/boardgame-ai-arena`. Data remains `/home/jihua/quoridor-data`. Jobs Python is `/home/jihua/quoridor-runtime/python/bin/python`; release and job source snapshots are immutable evidence, not development worktrees.

## Scope and source disposition
- Preserve all unified Quoridor P1–P8 work through `772e88f`.
- Preserve project-scoped Quoridor Jobs from `e61cc63` with Quoridor-only registry.
- Carry forward latest project manifest permissions/filtering, generic metric-step handling, chart labels, mirror privacy and source-isolation checks.
- Exclude game-specific policy/runtime/training/play features that only support other games. Preserve original `93323ca` and all old work in server archives; no model or experiment data is deleted.
- Historical `feature/backend_ai` engine/game-manager source is present in P8, with later Python-binding additions. Its old architecture docs remain historical evidence.
- Training's dirty Lab patch is superseded: current Lab already has top-level multiCompare and replay-position import; preserve the original patch in the source archive instead of importing its incorrectly nested function.
- v0 split repositories and their uncommitted work are preserved as historical implementation; current Rust engine, gameplay, agents and Jobs replace their execution paths.

## Validation and operations
See workflow roadmap for AC and validation coverage. Scope test first failed against the original candidate (two registered games), then passed with Quoridor only. Additional preservation test confirms unsupported historical project rows remain stored but are not advertised. Existing Jobs and Rust-adapter regressions passed. Other checks and rollout records are stored under `~/quoridor-backups/main-consolidation-20261122` on each host.

Only merge tested changes via PR. Start future work on `codex/<topic>` from updated main, then merge and deploy a fixed main commit. Never run experiments from a dirty source tree. Do not use reset/clean/force-push to reconcile machines.

Deployment uses the existing topology. Lab, platform and gameplay use the same source image; gateway is built from the same commit. The GPU worker base contains dependencies/engine and loads the recorded job source snapshot. Any source-independent dependency image is recorded separately. Preserve one Jobs writer. Database rollback must preserve writes made during cutover; never overwrite live data with a stale snapshot.
