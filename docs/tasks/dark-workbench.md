# Dark research workbench — 2026-10-30

Status: verified on 8790. Branch: feature/agent-ladder-preview.
Scope: visual refinement of existing UI; only preview gateway 8790. Preserve navigation, APIs, models, data, MLflow and replay auto-analysis behavior.
Direction: graphite/navy surfaces, cyan interaction accent, violet neural and mint heuristic tags, dense tables, quiet borders, tabular numbers, matching dark board. References: https://linear.app/now/behind-the-latest-design-refresh and https://grafana.com/whats-new/2026-10-26-grafana-visual-ui-refresh/ . No decorative copy or simulated data.
New theme preference key quoridor-workbench-theme defaults to dark on first visit to this design; thereafter respects light/dark choice. Old quoridor-theme remains preserved for rollback. Board CSS variables retain historical fallback colors outside this shell.
Acceptance: typecheck/build, existing UI tests, real-browser ladder/Agent/training/replay/Play screens, desktop/mobile overflow and persisted theme switching; check replay controls and auto analysis. Save screenshots externally. Snapshot deployment, replace only gateway, preserve rollback image/config and verify other container identities.

## Verified release
- Runtime source c7ebe8b; branch feature/agent-ladder-preview. Committed locally on Training; not pushed to GitHub.
- Image sha256:6294eca5fa7987eabf1ed40ed105eaf91f86ff025f44bb0514b1b953c9ef09a8.
- Backup /home/jihua/quoridor-releases/agent-ladder-preview/ui-refinement-20261030T161154Z. Restore its saved GATEWAY_IMAGE in release.env and run manage.sh up -d --no-deps gateway for rollback. Old images retained.
- TypeScript, production build, git diff --check and 48 Vitest tests passed.
- Seven Playwright scenarios passed: dark routes/theme persistence, automatic replay requests/cache, acting-Agent/tensor/themes, replay transport/responsive layout, play seats, training/checkpoint/replay/play flow, training mobile evidence/config.
- Screenshots and reports: /home/jihua/quoridor-backups/dark-workbench-20261030. Linux test runner lacks CJK fonts; Chinese text/rendering additionally checked in user's Windows in-app browser on the Agent page.
- Only preview gateway changed; all other running Docker containers retained ID/image/start time. Training 8741, MLflow 8742 and Dev 8765 returned HTTP 200.
- No backend restart, production switch, model/data/MLflow mutation or external font dependency.