# Live research-platform acceptance

Run `npm run e2e` from the dashboard frontend against the already-running real
platform (default `http://127.0.0.1:8751`). Install dependencies with `npm install`
and the browser with `npx playwright install chromium` if necessary.
On this host Chromium's missing `libasound2` was downloaded and extracted without
privileges into the Playwright user cache under `local-libs`. The configuration
automatically uses that library directory when present; no system package or
production service was changed.

No mocks, application startup, training, registry writes, deployments, or generated
games are used. Only the isolated Lab position validation, explicitly clicked
neural analysis, and its read-only branch/tensor diagnostics may POST. No analysis
or branch requests may occur during import. Every browser request is recorded. Tests use the exact
C-resnet seed-42 run and checkpoint SHA, not aliases or whichever run is newest.

Environment overrides:

- `E2E_BASE_URL`: existing platform origin.
- `E2E_CHROMIUM_PATH`: optional installed Chromium executable.
- `E2E_RESULTS_DIR`: absolute external evidence directory (not in a workspace).

By default each invocation writes timestamped JSON/HTML reports, screenshots,
request audits, exported decision cases, and failure traces below
`/home/jihua/quoridor-data/platform-p1-p2/acceptance`. Nothing generated is placed
in Git. Upstream requests/assertions allow 120 seconds; tests run sequentially,
without retries so real failures remain visible.

The analysis click is a real UI click. Force is allowed only after a documented
browser actionability/layout failure; the target and hit-tested obstructing
element are attached to the report. A force click cannot manufacture a response:
the test still requires a successful actual analysis response, matching position,
exact neural SHA, rendered decision, and exported result.