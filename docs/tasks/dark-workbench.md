# Dark research workbench — 2026-10-30

Status: implementing. Branch: feature/agent-ladder-preview.
Scope: visual refinement of existing UI; only preview gateway 8790. Preserve navigation, APIs, models, data, MLflow and replay auto-analysis behavior.
Direction: graphite/navy surfaces, cyan interaction accent, violet neural and mint heuristic tags, dense tables, quiet borders, tabular numbers, matching dark board. References: https://linear.app/now/behind-the-latest-design-refresh and https://grafana.com/whats-new/2026-10-26-grafana-visual-ui-refresh/ . No decorative copy or simulated data.
New theme preference key quoridor-workbench-theme defaults to dark on first visit to this design; thereafter respects light/dark choice. Old quoridor-theme remains preserved for rollback. Board CSS variables retain historical fallback colors outside this shell.
Acceptance: typecheck/build, existing UI tests, real-browser ladder/Agent/training/replay/Play screens, desktop/mobile overflow and persisted theme switching; check replay controls and auto analysis. Save screenshots externally. Snapshot deployment, replace only gateway, preserve rollback image/config and verify other container identities.
