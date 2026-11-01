# Unified RL platform: single entry for training, diagnosis and play
Status: deployed 2026-10-31 (P4 deploy.sh and P5 cutover.sh run at the user request; entry http://192.168.68.102:8790; Training 8790 redirects; old services retired on both hosts the same day, rollback in retire-legacy-20261031)
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
P1 (2026-10-31):
- `8be81f0` adds `/api/v1/experiments` and `/api/v1/experiments/{id}` (arms = run directories holding `checkpoints/` up to three levels deep; recorded Markdown reports capped at 256 KB; `assessment.json` up to 1 MB; models joined only by catalog `training_experiment`/`source_path`) and a 60 s cache for the model listing.
- `61f091e` turns the 8790 shell into seven sections (总览, 实验, 训练, 天梯, 模型, 实验室, 对战). New model library/detail and experiment pages; run detail, replays, Arena and native Lab are mounted in the same shell. LegacyApp, external research Play, Classic Play, the old Compare, run list, models and MLflow views and their tests were removed; links to them now point to replacements. The replay board height reserve now includes the history-trail toggle. `scripts/verify_stack.sh` starts/stops an isolated loopback stack.
- The Jobs section (任务) is deferred to P2 so the shell has no placeholder page.
P2 (2026-10-31), `87e9a28` plus fixes through `ae293ca`:
- `apps/jobs`: host controller (SQLite record, scheduler thread, Docker CLI, git source export) and `/api/v1/jobs` API. One GPU Job at a time, at most `JOBS_MAX_CPU_JOBS` CPU Jobs; each Job runs in `docker run --init --network none --read-only` with `--device nvidia.com/gpu=all` for GPU Jobs, host uid, a read-only snapshot of the submitted commit (historical `agent_system/training/artifacts` excluded) and only its run directory writable. Restart reconciliation keeps tracking live containers and fails lost containers or interrupted preparation visibly; nothing is rerun. Writes require `X-Platform-Token` (`~/quoridor-releases/jobs/private/token`, mode 600).
- `agent_system/training/jobs.py`: `train_dqn` (existing `scripts/train_dqn.py` with the submitted YAML; checkpoint_dir and device set by the Job) and `evaluate_checkpoints` (registered SHA checkpoints against fixed opponents via `run_evaluation`). Writes `events.jsonl` and `manifest.json`.
- Registration copies verified checkpoints to `models/<sha>.pt` (0444) and appends `catalog-jobs.json` entries with `training_job`, `source_commit` and an exact `run_ref` to the Job run. The model index rediscovers catalogs on each build. `jobs/runs` is a Training artifact root, so Job runs show curves in the Training workspace.
- `deploy/Dockerfile.train-gpu`: python 3.12 + torch 2.11.0+cu126 + engine wheel, image `quoridor-train-gpu:39aba9c` (10.3 GB).
- Web: 任务 section (list with GPU state, templated submission with token, live curves/evaluations/logs/events, cancel), model detail links to a prefilled evaluation Job. Gateway template routes `/api/v1/jobs` to `JOBS_UPSTREAM`, which every deployment of this template must now set.
- Deviations from the design: the UI polls (3-5 s) instead of SSE; the controller is a host process from an exported source snapshot because it needs git and Docker; every checkpoint in a successful Job is registered.
P3 (2026-10-31), `020eba4`, `1c807f2`, `84282eb`, `e52c034`:
- `train_cnn_experiment` runs `scripts/train_cnn_repair.py` (distillation, PPO, DAgger, DDQN transfer, AlphaZero-lite, teacher, CNN DQN) with any committed experiment JSON (34 templates under `agent_system/training/configs/cnn_*`), keeping its per-checkpoint diagnosis; the runner records the Job commit through `QUORIDOR_SOURCE_COMMIT`. Job detail shows research metrics and diagnoses.
- `run_arena_matrix` evaluates registered checkpoints against `agent_system/evaluation/arena/gates/gate-v1.json` (the opponents, 25 openings and settings of `cnn-architecture-20261021-213018`) and publishes a complete report as ladder season `arena/<job id>`. Training Jobs may request a gate follow-up on their final checkpoint. Gate seasons are labelled and are not the ladder default.
- Experiments list their Jobs (also Job-only experiments) and Job-registered models by Job; model detail links to the producing Job; Training `run_ref` is only recorded for runs the Training workspace can list.
- `c7c7797` failure mining: Agent detail lists recorded games it lost, drew by repetition or ply limit, or where it missed a win (from recorded outcomes and per-game decision diagnostics only); for a recorded missed win the Rust engine locates the plies with a legal winning pawn move and links to them.
- `728dd73` `diagnose_checkpoint` Jobs run the existing CNN diagnosis on registered checkpoints; `/api/v1/models/{sha}/diagnoses` collects every diagnosis of the exact SHA from experiment and Job outputs, shown on model detail.
- Versioned probe suites as separate data (design section 7) were not created: the tactical probes stay inside `cnn_diagnostics` and books are referenced by manifest path and recorded `book_sha256`.
P6a training visibility (2026-11-01), `d054bf4` plus fixes:
- `scripts/train_dqn.py` adds window metrics to every `metrics.jsonl` record (`_window_metrics`): `win_rate`, `loss_rate`, `truncation_rate`, `wall_ratio`, `repeat_rate` (learner decisions in a position already seen this episode; the observations have no move counter), `max_episode_length`, `q_mean`, `target_mean`, `td_error_max`, `grad_norm`, `elapsed_s`, `episodes_per_s`. Existing fields are unchanged.
- `apps/jobs/health.py` turns rows into alerts: NaN/Inf, truncation >= 30%, repetition >= 25%, mean |Q| > 5, loss > 10x the earlier median, evaluation win rate 20 points below the best earlier one per opponent, and no new metrics for 10 minutes while running. Alerts never stop a Job.
- `GET /api/v1/jobs/{id}/metrics?after=N` returns `offset`, `total` and `alerts`; NaN/Inf become null (they previously made the response fail). `GET /api/v1/jobs/{id}/stream` sends the same data as server-sent events every 2 s while it changes, a keep-alive comment every 15 s, and `end` for a finished Job. The existing `/api/v1/jobs` gateway locations already disable buffering, so no gateway change.
- Job page: alert banner, latest values, six grouped multi-series panels (results, behaviour, length, loss, value, optimisation), EMA smoothing, log scale for loss panels, episode or environment-step axis, any other field under 其他指标, and an evaluation win-rate curve per opponent. EventSource while the Job is active (a reconnect replaces rows instead of duplicating), polling otherwise.
- `scripts/verify_stack.sh` accepts `VERIFY_JOBS_DATA` so a test controller does not share the live Job database.
- Not in P6a: the CNN research runner (`train_cnn_experiment`) keeps its own metrics; the dashboard shows them under 其他指标.
P6b checkpoints, sample games and comparison (2026-11-01), `8602c29`, `b35471a` plus fixes:
- Job detail lists `checkpoints` (manifest SHA once written, files while running; episode and environment step from `epNNNNN_stepN.pt`); `/metrics` also returns the `eval_replays.jsonl` index as `replays`.
- Job page tab Checkpoint: per-opponent W/D/L at each checkpoint, the recorded evaluation games linked to the existing replay viewer (`/replays/<job id>?replay=...`), diagnosis by exact SHA, and prefilled 诊断 / 门槛评测 Jobs for registered checkpoints.
- `/jobs/compare?ids=a,b,...` (2-4 Jobs, one color each): summary table (latest window rates, loss, best evaluation win rate) and overlaid smoothed curves, evaluation win rate per opponent and any other metric. Reached from checkboxes on the Job list or the experiment Job tab.
- Fix: `train_dqn` Jobs now record the Job id as `agent_id` (the template value goes to the `process` event). The replay writer uses `agent_id` as the run id and the research API only serves replays whose run id is the run directory, so every recorded game of earlier Jobs whose template sets `agent_id` (for example `platform_job_smoke`) is listed as unavailable; those files are left unchanged.
- Known, not changed: the legacy replay viewer keeps its light panels inside the dark shell; recorded replays carry no checkpoint SHA (the viewer says so).
## Verification commands and observed results
- `npm run typecheck`, `npm run test:ui` (55 passed), `npm run build`: passed before and after the split.
- Backend (documented command, see docs/WORKSPACES.md): 134 run, all passed, 1 optional skip; with `ARENA_REPLAY_TEST_ROOT` set, 0 skipped. An earlier run reported a `backend.test_mlflow` import error only because `training-dashboard` was missing from `PYTHONPATH`; the test imports the local adapter, not the MLflow package. Five new tests pass.
- Real data (`/home/jihua/quoridor-data`): discovery reads 11 catalogs and indexes 104 models, equal to the distinct catalog SHA count (default selection: 92). No identity/metadata conflicts. 19 models ranked; all 19 season SHAs are present in the index. Through the full app with MLflow: listing HTTP 200 in 5.4 s cold (run lineage dominates, as for `/api/platform/models`), detail 0.2 s with SHA verified, unknown/invalid SHA 404.
- Browser: isolated loopback stack (Platform 18793, Lab 18794, preview gateway image on 18795; all stopped afterwards). 13 ladder-related tests: 10 passed, 3 failed. The same 3 fail identically with the base-commit frontend in the same stack, so they predate P0: two `agent-preview.spec.mjs` cases still expect the pre-compact Agent detail layout (a standalone 训练记录 heading and 该类型无需训练 text), and one `replay-refinement.spec.mjs` layout check measures 790 px against a 768 px limit. To be updated in P1 together with the new shell.
- P1 frontend: typecheck and build passed; React 52 tests passed (legacy-only tests removed, eight new shell/model/experiment tests added).
- P1 backend: 139 run, all passed, 1 optional skip.
- P1 real data: 10 experiments listed in 0.01 s; 98 of 104 models joined to an experiment arm by exact provenance (the other 6 come from legacy artifact runs). Cached model listing 0.035 s after the cold 5.4 s request.
- P1 browser (isolated stack, `/home/jihua/quoridor-backups/unified-rl-platform-p1/e2e-final`): 17 of 17 passed, including the three previously stale cases. Desktop 1440 and mobile 390 screenshots of overview, models, model detail and experiments show no horizontal overflow (`.../shots`; CJK glyphs render as boxes because the headless browser lacks CJK fonts).
- P2 unit: Jobs controller 8 tests (`PYTHONPATH=.:backend-server:engine <venv python> -m unittest discover -s apps/jobs/tests -t .`), entrypoints 5 tests (`-m unittest agent_system.training.tests.test_jobs`), platform backend 139 + 1 new, React 56 (4 new Jobs tests). All passed.
- P2 GPU image: CUDA available on the RTX 3080 Ti, a GPU matmul and `RuleEngine.standard()` succeed as uid 1000 with `--network none --read-only`.
- P2 real Jobs on Training through the isolated stack (controller on 127.0.0.1:18796): smoke training ran with `resolved_device=cuda` and registered its checkpoints, which appear in `/api/v1/models` (mlp, q_values, analyzable, SHA verified, run_ref to the Job) and as a run in `/api/runs`; an evaluation Job recorded per-opponent W/D/L with zero illegal actions; restarting the whole stack while a 300-episode Job ran kept the container and the new controller finalized it as succeeded (15 checkpoints); cancelling a running Job ends it in 5 s (exit 143) without registration. Two early runs failed and are kept as records: the image had no passwd entry for uid 1000 (fixed with USER/LOGNAME), and the evaluation script import needed `sys.modules` registration.
- P2 browser: 18 of 18 passed (`.../unified-rl-platform-p2/e2e-final`), including a real GPU Job submitted, followed and opened through the UI.
- Acceptance data left in `/home/jihua/quoridor-data`: `jobs/` (9 records, 74 MB), 21 `catalog-jobs.json` entries from 4 smoke Jobs and their `models/<sha>.pt` copies. The first smoke Job's 2 entries predate `run_ref`.
- P3 reproduction acceptance (`/home/jihua/quoridor-backups/unified-rl-platform-p3/reproduction-shaping-A.json`): the reward-shaping arm A config submitted unchanged as `job-20261031-161625-b2fc26`. Initialization weights equal; training games identical through batch 11; loss values differ by up to 7e-8 in batch 0 (GPU floating-point nondeterminism) and the drift changes a sampled action in batch 12, so the run is not bitwise reproducible. Learner win rate 0.367 vs 0.371; development diagnosis wins 38/39/38 vs 37/38/38 of 50, tactical 18/18 and no missed immediate wins in both.
- P3 gate acceptance (`gate-shaping-A-season.json`): historical and reproduced final checkpoints in one gate Job (`job-20261031-162303-62e284`, 260 blocks in 10 s, published as a season). Versus random/greedy/D1/D2/frozen D2 CNN: historical 48-2-0, 38-0-12, 38-0-12, 0-12-38, 6-43-1; reproduction 49-1-0, 36-0-14, 36-0-14, 0-14-36, 3-43-4 (Elo 2179 vs 2154); differences are within 50-game sampling error.
- P3 failure mining on real seasons: all 5 recorded missed-win games of `cnn-architecture-20261021-213018` recompute to exactly the recorded counts (17 Agents scanned in 4 s; `matrix-20261020-v2` has none).
- P3 diagnosis acceptance: `job-20261031-163034-278f72` diagnosed the frozen D2 CNN on the `matrix-20261020-v2` book (`c6268c82…`); every count shared with the historical `cnn-distillation-20261020/02-policy-d2` diagnosis is identical (38 wins / 12 losses, 734 decisions, 18/18 tactical, score 0.76); only the newer `search_changed_policy_argmax` key is added.
- P3 browser at `ba7adbb` (`/home/jihua/quoridor-backups/unified-rl-platform-p3/e2e-final-3*`): 18 of 18 passed, including a real GPU Job. The first run failed the 390 px overflow check on the ladder: eight navigation links and the long gate-season option widened the page; fixed in `ba7adbb`.
- P3 tests at `728dd73`: Jobs controller 11, entrypoints 9, platform backend 143 run with 1 optional skip, React 62; all passed.
- The reproduction Job's three catalog entries were registered before `1c807f2` and still carry a Training run_ref that the Training workspace cannot list.

P4 (2026-10-31), `ddef780`. Classic play is not kept (user decision 2026-10-31), so no Classic feature is ported.
- `apps/gameplay` is the only owner of game state:
  - It stores actions in SQLite (`quoridor-data/games/games.sqlite`); every position is recomputed by the Rust engine.
  - It keeps a player token (moves, close, publish) and a read-only watch token, both stored as SHA-256 digests.
  - After a restart, Agents are rebuilt from the recorded identity with the exact SHA check.
  - A finished game can be published to an unranked list, which anyone can open read-only.
  - Limits: 64 active games, abandonment after 14 idle days, 500 plies.
  - The in-memory `Sessions` were removed from `agent_arena.py`.
- `/api/v1` is the only prefix the SPA uses:
  - `/api/v1/research/{runs,summary,arena,mlflow,context,catalog}` and `/api/v1/seasons/*` reuse the legacy handlers (`alias_legacy`).
  - The old `/api/*` paths still answer.
  - `/api/v1/lab/*` goes through the existing Lab allowlist.
  - `/api/v1/play` goes to the gameplay service.
- The gateway drops the Classic `/classic-api`, `/ws` and `/play-api` proxies, and `platform_play.py` is deleted. `GAMEPLAY_UPSTREAM` is required wherever the template is used.
- The Play page gained watch links, a read-only watch view that polls every 2 s, and the published-games list.
- P4 tests at `ddef780`: gameplay 6, platform backend 141 (1 optional skip), React 65; all passed.
- Browser, isolated loopback stack with gameplay on 18797: 18/18 passed, including a new watch-link test (`~/quoridor-backups/unified-rl-platform-p4/e2e-1`).
- Restart durability (`.../p4/restart.txt`): a human-vs-`C-resnet_seed-42` game, with the gameplay process killed and restarted after 2 plies, returned the same actions and position; the CNN then reloaded by SHA and moved.
- Image check (`.../p4/image-verify`, `e2e-image`):
  - The new platform and gateway images, plus the gameplay container, ran as an isolated compose project on 127.0.0.1:18890 (since removed).
  - All `/api/v1` routes returned 200.
  - A CNN game resumed after `docker restart` of the gameplay container; the SQLite file is owned by uid 1000.
  - Browser 18/18 passed.
  - The nested read-write `games` mount needs the directory to exist on the host; `deploy.sh` creates it, and `~/quoridor-data/games` now exists (empty).
- Retirement: see below. Before retirement, the old entries logged only curl probes in the last 72 h, and the gameplay and Classic backends logged no requests.
UI refinement (2026-10-31), `6753bdf`:
- The overview became a dashboard: KPI tiles for models, ranked models, experiments, active Jobs and GPU; quick actions; Elo bars for the top six; recent Jobs with status.
- `Theme.css` refines every page using theme tokens only: quiet links, primary buttons, sticky header, row hover, focus rings, loading spinners, capability chips without strikethrough.
- Screenshots before and after (with a user-level Noto Sans SC font installed on Training for headless rendering): `~/quoridor-backups/unified-rl-platform-p5/shots-*`.
- React 65 passed; browser 18/18 passed (`e2e-ui`).

P5 (2026-10-31), `54593aa`, `379a943`: entry and gameplay on Dev; Training keeps compute and evidence.
- The gateway image renders a second listener (`gateway-api.conf.template`, port 8081) that serves the research API, Lab and Jobs without the SPA; `/api/v1/play` returns 404 there. Training publishes it on 8792.
- `compose.dev-entry.yaml` runs the Dev gateway (8790) and the gameplay service.
- `compose.training-api.yaml` keeps the platform and Lab, publishes the API listener, and turns 8790 into a 301 to Dev.
- `sync_gameplay_mirror.sh` with a 10-minute user timer mirrors `quoridor-data` to Dev at the same path:
  - excluded: `mlflow/`, `games/`, Job snapshots, `jobs.sqlite`
  - size 5.8 GB; incremental run about 1 s
  - every catalog checkpoint is under the data root, so no artifact mirror is needed
- Upstream connect timeout is 5 s, so a stopped service fails fast.
- Verification, all on isolated ports. Evidence in `~/quoridor-backups/unified-rl-platform-p5/`.
  - Browser 18/18 through a Dev test entry backed by a Training test API listener (`e2e-dev`), plus the real GPU Job submitted via Dev (`e2e-dev-jobs`).
  - Latency (`drills.json`, median ms, Training direct → via Dev):

    | Operation | Training direct | Via Dev |
    |---|---|---|
    | roster | 11.8 | 18.0 |
    | Lab Minimax d2 | 38.6 | 42.9 |
    | heuristic move | 6.0 | 15.6 |
    | CNN move | 4.4 | 14.9 |

  - Training API down: Dev games keep working, research and Jobs routes return 502, and everything recovers.
  - Dev gameplay down: 504 within 5 s and other pages are unaffected; the game resumes after restart.
  - Full cutover and rollback simulation on test ports (`sim.log`): games moved both ways, including moves made on Dev.
- The Dev mirror `~/quoridor-data` (5.8 GB) stays in place for the real cutover; every test stack and test unit was removed.
- The images are already loaded on Dev.

- P6a (2026-11-01): Jobs 20 tests, `test_jobs` 10, training regression set 181 passed, platform backend OK, typecheck, UI 66 passed (new stream test with a fake EventSource). Live check on loopback with a throwaway Job data root (`VERIFY_JOBS_DATA`): GPU Job `job-20261101-224802-e9d4a5` (smoke config, 150 episodes, source `d054bf4`) streamed a row every ~2 s through the gateway and ended with `end`; it raised the truncation (40%) and repetition (62%) alerts, matching the smoothed curves (win rate down from 40% to about 10%). Screenshots desktop and 390 px without horizontal overflow: `/home/jihua/quoridor-backups/unified-rl-platform-p6a/shots/`.
- P6b (2026-11-01): Jobs 21 tests, UI 68 (timeline, list selection, compare), platform backend OK. Loopback with the throwaway Job root: `job-20261101-232735-9ad114` (distance_delta, source `8602c29`) and `job-20261101-232951-404c27` (greedy training opponent, source `b35471a`). Replays of the first are refused (identity mismatch), of the second served (200) and opened from the timeline. Compare of the three P6 Jobs: the greedy-trained run loses every game in about 18 plies (0% truncation and repetition), the other two loop (truncation 40-60%, repetition about 60%). Screenshots: `/home/jihua/quoridor-backups/unified-rl-platform-p6b/shots/`.
## External artifacts and deployment impact
P5 cutover package (prepared, not run): `/home/jihua/quoridor-releases/p5-cutover-379a943/`, containing `cutover.sh`, `rollback.sh`, `lib.sh`, the Training compose, the redirect, the Dev files and a README.
- Run on Training after the P4 deploy.
- The retirement scripts now follow the 8790 redirect, so they work before or after the cutover.
P4 packages (prepared, not run):
- `/home/jihua/quoridor-releases/agent-ladder-preview/unified-platform-ddef780/` deploys platform `sha256:47de7af6…`, gateway `sha256:671598fd…` and the gameplay container, and installs the Job controller unit if it is missing. `ba7adbb` is marked SUPERSEDED.
- `/home/jihua/quoridor-releases/retire-legacy-20261031/` exists on Training and on Dev:
  - `retire-dev.sh`: 8765 entry, 8776, 8774, Classic 8764 and 8090; 8765 then redirects to 8790.
  - `retire-training.sh`: 8740/8741 entry, 8781, 8783, the old Lab, `quoridor-agent.service` 8743 and `quoridor-decision-lab.service` 8744; 8740/8741 then redirect.
  - Containers are stopped with restart policy `no`, units are disabled, and the state is saved once.
  - `rollback.sh` restores each host.
  - `selftest.sh` ran retire → check → rollback with a throwaway container on both hosts (`selftest.log`).
Evidence: `/home/jihua/quoridor-backups/unified-rl-platform-p0/`, `.../p1/` and `.../p2/` (logs, browser results, `jobctl.py`). Deployment package (current): `/home/jihua/quoridor-releases/agent-ladder-preview/unified-platform-ba7adbb/`; the earlier p1, p2 and p3 packages are marked SUPERSEDED. Original P2 package: `/home/jihua/quoridor-releases/agent-ladder-preview/unified-p2-20261031/` (`deploy.sh`, `rollback.sh`, compose, user unit, README). Deploying needs the user: the automated session was not permitted to replace the running preview. Current images: platform `sha256:54635638…`, gateway `sha256:4e0d7e8b…`, both from `ba7adbb`; Job controller snapshot `~/quoridor-releases/jobs/src-ba7adbb`. (`e2e/`, `e2e-baseline/`, logs, temporary gateway config). Nothing deployed; 8790 and all other services unchanged. Not pushed to GitHub.
## Deferred decisions
All section-12 items are decided (design section 0); Classic play is not kept. Still open: MLflow `/mlflow/` proxy and versioned probe suites. Deployment of P4, retirement and P5 is pending the user.
