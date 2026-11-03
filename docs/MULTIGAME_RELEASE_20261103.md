# Multi-game research release candidate

2026-11-03. Branch: codex/multi-game-rl-platform.
First research implementation complete; not deployed.
Executable source: 02316f41312610beaf4271151086fd8a2b0bda8f.

## Delivered
- Explicit project/game/rules scope for Jobs, policies, evaluations and MLflow.
- Quoridor Rust authority, legacy hashes, Arena and routes preserved.
- Kuhn Poker OpenSpiel CFR/tabular Q-learning and exact exploitability.
- Mr. Jack London original base game, all eight powers, private alibis, chance events, witness/escape/accusations; independent SVG board.
- SB3-Contrib Maskable PPO CPU training, numeric-only actor export, probability parity check, immutable multi-file policy ZIP.
- Human versus random/heuristic/exact-policy play, role capabilities, public spectator projection, durable RNG recovery, projected replay and analysis context.
- AgentSpec v2 identity binds rules/role/runtime/encoder/memory/semantics/policy SHA. Manual research eligibility additionally binds project, target and exact evaluation evidence; it never changes production deployment.
- Privacy-safe mirror, atomic public project catalog and fail-closed policy dependency checks.

See [rules sources and conventions](../agent_system/games/MRJACK_RULES.md). SB3-Contrib replaces the proposal's provisional RLlib candidate. No third-party GUI AI or artwork was imported.

## Verified
- 70 games/Jobs Python tests, including legacy Quoridor parity.
- 11 Gameplay tests (separate process because two historical backend package roots coexist).
- 79 UI tests; TypeScript and production build.
- 8 migration tests; scripts/project.py check.
- Chromium: create, spectator isolation, action, reload, projected replay, close; screenshot inspected.
- Final-source real Docker: three 128-step training runs (seeds 7/42/101), six evaluations totaling 120 games, controller recovery, bundle download, role eligibility, gameplay restart and cancellation.
- Kuhn final-source Docker regression: two train, two evaluate, controller recovery and cancellation.
- Final platform image: complete neural-policy game with read-only evidence, temporary writable sessions, networking disabled.

Evidence on Dev, outside Git:
- /tmp/mrjack-acceptance-20261103-final/acceptance.json
- /tmp/kuhn-final-regression-20261103/acceptance.json
- /tmp/mrjack-final-image-acceptance.json
- /tmp/mrjack-browser-20261103-b/results.json
- /tmp/mrjack-browser-20261103-b/mrjack-board.png
- /tmp/mrjack-ui-tests.log

These are functional pilots, not evidence of learning improvement. Paired mean returns against random were -0.1/0.0/-0.2; against heuristic, -0.9 for all three seeds. Detective play was particularly weak. Each opponent/seed uses only 10 scenario pairs. Full role results and standard errors are in the reports.

The initial encoder is a memoryless visible summary. Recurrent training, optimizer resume, stronger opponent pools and hidden-information search remain future research directions. No publisher certification or automatic statistical promotion gate is claimed.

## Images on Dev
Local immutable image IDs; these are not published registry digests:

| Component | Tag | Image ID |
|---|---|---|
| Platform/Gameplay | quoridor-multigame-platform:dev-20261103-final | sha256:b2597727e2c288f85829eb327c7fb64c43b0a46b16c82affb3fa70c9f5fed38d |
| Gateway/SPA | quoridor-multigame-gateway:dev-20261103-final | sha256:ecd5387afbf9ea1ec95cb52b6569e66015637b3ac6f43b105485cf23cf912e6c |
| Neural CPU | quoridor-research-neural:dev-20261103 | sha256:8eecd18fa00020fd1ffd36cda6a9746afb9905908d0bcc808890989837143362 |
| OpenSpiel CPU | quoridor-openspiel:dev-20261103 | sha256:1193ba92514a1ebed862e6148cbde332ddef80c15c93e1cd512c68c26b060f11 |

Workers execute the immutable Job source snapshot at /job/source; worker images pin dependencies separately from application source.

## Usage after coordinated rollout
1. Configure Jobs source and its pinned OpenSpiel/NumPy dependencies; set JOBS_OPENSPIEL_IMAGE and JOBS_NEURAL_IMAGE to available immutable image IDs.
2. In Projects create a Mr. Jack London base-game project.
3. Submit train_mrjack_ppo from its Jobs page.
4. In project policies choose Evaluate (evaluate_mrjack_policy), download the ZIP, or Play.
5. Choose detective or Jack. The spectator link uses a separate read-only capability. Tokens are URL fragments and request headers.
6. Optional research eligibility requires an exact completed evaluation, role and human note. This records review, not deployment.

Research play uses /api/v1/play-v2; legacy play remains /api/v1/play. Downloads and eligibility writes require the platform token. Private replay files have no public download route. Analysis context contains only the caller's visible history; generic CNN activation analysis and recurrent state reconstruction are not exposed for Mr. Jack.

## Rollout and rollback
No live service, roster, alias or production database was switched. Only isolated test data, containers, candidate images and this branch changed. Previously deployed releases were not re-audited.

A later rollout must distribute and verify images on both hosts, preserve current release settings and back up databases, install the mirror update, then coordinate Jobs, Training platform, Dev Gameplay and gateway. The older release helper does not replace Gameplay, so it alone cannot deploy this feature. Verify project catalog and policy dependencies arrive on Dev before enabling research play. Keep one controller per data root.

Rollback disables new v2 submissions/sessions and restores previous service images; retain new databases/artifacts for a compatible reader. Do not restore old backups over new evidence or rewrite legacy hashes. Live rollout remains a separate operational action.

## Reproduce
Use a NEW data root for each container run. Frontend requires Node 22+.
```sh
.venv/bin/python -m pytest agent_system/games/tests apps/jobs/tests -q
PYTHONPATH=training-dashboard:.:engine .venv/bin/python -m pytest -o pythonpath="training-dashboard . engine" apps/gameplay/tests -q
npm run typecheck
npm run test:ui
npm run build
python3 scripts/project.py check
.venv/bin/python -m unittest discover -s tests/migration -q
PYTHONPATH=training-dashboard:.:engine .venv/bin/python scripts/verify_mrjack_jobs.py --data-root /tmp/NEW-mrjack-acceptance --image quoridor-research-neural:dev-20261103 --source 02316f4
```
