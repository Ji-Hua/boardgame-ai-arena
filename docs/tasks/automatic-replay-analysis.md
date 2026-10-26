# Paused replay automatic acting-Agent analysis
Status: verified on preview 8790
Scope: preview gateway 8790 only; no backend/service/data changes.
Behavior: debounce paused replay positions for 350 ms, follow the current Agent, cancel/invalidate on playback or seeking, never auto-run comparison models. Reuse up to 16 successful position/Agent results per mounted replay; explicit Run analysis refreshes. Terminal/unsupported actors do not auto-run. Existing live snapshot analysis stays manual. Abort prevents stale UI updates but cannot guarantee stopping already-running backend computation.
Acceptance: request-count tests for playback, pause, rapid seeks, caching, errors without retry loops, late responses; real-browser replay and existing regression flows. Preserve gateway rollback and all other container identities.

## Verification and handoff
- Branch: feature/agent-ladder-preview; runtime source: 16adb01.
- 48 Vitest tests passed; TypeScript, production build and git diff --check passed.
- Four Playwright scenarios passed: auto replay request counts, acting-Agent diagnosis/tensors, replay controls/layout, Agent-origin play seats.
- Browser artifacts: /home/jihua/quoridor-backups/automatic-replay-20261025.
- Gateway image: sha256:eb4e5eb8b9b1da9082a3f4e71d501ffa7e2707bd4e96b9e2d987861418015a36.
- Rollback config and container snapshots: /home/jihua/quoridor-releases/agent-ladder-preview/ui-refinement-20261026T002504Z. Restore saved GATEWAY_IMAGE and run manage.sh up -d --no-deps gateway to roll back.
- Deployment verified that every other running container retained its ID, image and start time. Training 192.168.68.88:8741, MLflow :8742 and Dev 192.168.68.102:8765 returned HTTP 200 (8741 binds LAN address, not loopback).
- No GitHub push, production updates, backend restart, model/data/config migration. Successful result cache is session-local; failures are not automatically retried.
