# Unified web rollout and rollback

> Historical (2026-10-25 releases). The services described here were retired on 2026-10-31;
> current topology and release procedure: [RELEASE.md](RELEASE.md).

User authorization: replace running components, preserve old code/configuration/data.

## Components and boundaries
One React build in apps/web. Shared board implementation in packages/board-ui.
Classic Play imports existing frontend/src controllers into the same bundle and
uses the preserved Gameplay service. Research Play uses a separate Gameplay /
Runtime pair with a release-specific capability key. Model catalogs are immutable
research snapshots; production agent definitions and MLflow remain unchanged.
All live services consume immutable image IDs; no development source is mounted.
Training evidence remains a transitional read-only artifacts mount.

## Preparation
1. Capture old container inspect, effective Compose files, systemd units and old
   frontend images into a private timestamped backup. Preserve original source
   backups and data paths. Never print service keys or commit private env files.
2. Build and record web/gateway/runtime image IDs and actual source revisions.
3. Create private release configuration and a fresh research service key.
4. Start candidates on unused ports; use existing source/model storage read-only.
5. Validate types, UI/contracts, real model analysis, cross-host P1/P2 games,
   replay-to-Lab exact position/SHA, mobile layouts and classic mode.

## Switch
Only frontend entries are replaced. Existing classic Gameplay and its Agent
Service stay running, preserving active backend games. Stop the old frontend
containers without removing them. Start the new gateway on the original entry
ports. Keep the new research API/Runtime and separate Gameplay at their candidate
addresses, now recorded as this release's managed services.
Check both host entrypoints, APIs, WS upgrade, actual bundle identity and Lab.
If any gate fails, stop the new entry and start the saved original container.
Do not restart or restore MLflow, overwrite a database, or prune Docker images.

## Rollback
Each host's private release directory contains rollback-entry.sh. It stops only the
new public gateway and starts the original preserved frontend container. This
restores the prior entry without changing old source or data. New research games
remain in the new Gameplay process until explicitly closed; keep that process
running during frontend rollback so snapshots are not lost.

## Remaining product boundaries
Live pause/resume and durable human game history are not implemented. Play-to-Lab
is explicitly a frozen snapshot; the live game continues. Old standalone Lab
remains available, while normal navigation uses native React Lab.
Public GitHub synchronization is separate from the user's deployment permission.

Actual release, exact image IDs and tested commands: [2026-10-25 release](../docs/UNIFIED_WEB_RELEASE_20261025.md).
