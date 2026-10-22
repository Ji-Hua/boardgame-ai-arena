# Unified research platform: completed P1/P2

**Verified:** 2026-10-22. Implementation/acceptance record, not a replacement
for the authoritative [system architecture](../documents/system/design/system-architecture.md).

## Scope and source state

- Worktree: `/home/jihua/workspace/boardgame-ai-arena-platform`.
- Branch: `feature/unified-platform-p1-p2`; implementation baseline:
  `a8fc58d5217de34d380b1a3fce9dab18c6be2ddc`.
- Implementation is committed on the named feature branch, not promoted into
  the shared integration branch. Resolve the implementation commit through
  `git log feature/unified-platform-p1-p2`; the baseline SHA alone does not
  include P1/P2. The final handoff records exact commit IDs and push status.
- P1 provides a routed React research shell, Training/Compare, Arena, global
  Replay navigation, SHA-addressed Models, and Play/Lab transition pages.
  P2 connects exact recorded replay positions and checkpoint identities to an
  isolated, same-origin instance of the existing Decision Lab.
- This is not a complete React Lab, integrated Play runtime, checkpoint-backed
  session service, or durable pause/resume implementation. P3/P4 remain future
  work. No authoritative architecture documents were changed.

## Running topology and protected boundaries

| Entry | Address | Actual role |
|---|---|---|
| Unified platform | `http://192.168.68.88:8751` | Built SPA, existing dashboard GET APIs, new platform GET APIs, allowlisted Lab proxy. |
| Isolated Lab | `http://127.0.0.1:8754` | Separate analysis process, accessible through `/research-lab/` on 8751. |
| Existing dashboard | `http://192.168.68.88:8741` | Retained; existing dashboard API on 8740 is not replaced. |
| MLflow | `http://192.168.68.88:8742` | Existing Tracking/Model Registry, unchanged. |
| Production Agent Service | `http://192.168.68.88:8743` | Existing pinned inference service, unchanged. |
| Existing Decision Lab | `http://192.168.68.88:8744` | Retained; not replaced by the isolated Lab. |

[The launcher](../scripts/start_unified_platform.sh) is a **foreground supervisor**
of two Uvicorn children. It traps termination, cleans up both children, and exits
if either child exits. It is not an installed systemd unit, has no automatic
restart policy, and does not establish survival across logout/reboot. A detached
caller is not durable service management. The live process and both child ports
were inspected; no service restart was performed for this documentation task.

The launcher rejects protected ports 8740–8744 and 8764–8765, occupied ports,
duplicate ports, and privileged ports. It uses the original Training venv and
legacy artifact roots without copying source or outputs into this worktree.
The supplied Dev handoff remains unchanged: checkout `69b495f…`, production
Backend HTTP 8764 → Training Agent Service 8743 retained. Dev was not redeployed
or remotely revalidated by this documentation task.

### Model sources and no registry mutation

The live index returned **92 unique full-SHA model identities**, not 92 proven
loads or deployments. Its source manifest contained baseline catalog 6 records,
Lab catalog 80, architecture catalog 18, registry mapping 80, and 6 pinned
AgentSpecs; overlapping records are merged by full checkpoint SHA, not added
as separate models.

Server configuration reads the original baseline, Lab and architecture catalogs,
the external registry mapping and a flat AgentSpec directory. The default paths
are defined in [ModelIndex](../training-dashboard/backend/platform_models.py).
Relevant overrides are `QUORIDOR_DATA_ROOT`, `PLATFORM_MODEL_CATALOGS` (JSON
path/name manifest), `PLATFORM_MODEL_BASELINE_CATALOG`, `LAB_MODEL_CATALOG`,
`PLATFORM_MODEL_ARCHITECTURE_CATALOG`, `PLATFORM_MODEL_REGISTRY`,
`PLATFORM_AGENT_SPEC_DIR`, and `PLATFORM_MODEL_APPROVED_ROOTS`.

At startup the launcher writes a deterministic, content-addressed **research-only**
catalog under `/home/jihua/quoridor-data/platform-p1-p2/catalogs/`. The observed
file was `research-catalog-17a0fc10c1260ac107745604a5836f580087ab3936465e520b804ce1ed338e46.json`.
It excludes unavailable/conflicting entries, uses full SHA IDs, and records
static eligibility without loading or hashing all checkpoints. Only the Lab
child receives this generated `LAB_MODEL_CATALOG`; the dashboard index keeps
the original source manifest. It is a startup snapshot, not a live catalog cache.

There is **no Model Registry mutation**: no registration, model-version write,
alias/champion movement, roster edit, checkpoint export, training, or tournament
launch. Reading MLflow and the separate Quoridor registry JSON does not publish
or deploy anything. The only generated catalog is external research material;
the production catalog and original Lab catalog are not rewritten.

## Actual frontend routes and navigation

Routes are implemented by [App](../training-dashboard/frontend/src/App.tsx),
with URL helpers in [platform](../training-dashboard/frontend/src/platform.ts).
Run IDs and replay keys must be URL-encoded, not shortened or interpreted as
local filesystem paths.

| Route | URL state / behavior |
|---|---|
| `/training` | Provider-neutral Legacy/MLflow run list and filters. |
| `/training/runs/:runId` | `tab=overview\|metrics\|benchmarks\|checkpoints\|config\|diagnosis\|replays`; default overview. |
| `/training/compare` | Repeated `run=` selects up to five distinct runs. |
| `/arena` | Existing latest matrix; `run_id`, `checkpoint_sha` (or legacy `checkpoint`), `evaluation` select recorded run evidence. |
| `/replays` | Global run replay index; `q` searches runs, groups load evidence lazily. |
| `/replays/:runId` | `replay` (also accepts `replay_path`), `ply`, `model_sha`; archive filters `checkpoint_sha`, `checkpoint`, `evaluation`, `opponent`, `result`, plus exact `checkpoint_group`/`evaluation_group` keys. |
| `/models`, `/models/:sha` | Full SHA detail and reverse run links; list `q` and `capability` filters. |
| `/lab` | `run_id`, `replay_path`, `ply`, `model_sha` resolve a context via GET, then require an explicit link to the isolated Lab. |
| `/research-lab/?context=…` | Existing non-React Lab, UTF-8 JSON encoded as unpadded base64url, through the isolated proxy. |
| `/play` | Explicit external HTTP(S) link configured at build time with `VITE_PLAY_URL`; no embedded game or model deployment. |

`/` redirects to `/training` (`/#arena` to `/arena`); legacy `/run/:runId`
and `/compare` redirect to the new routes. HTML navigation has history fallback;
unknown APIs, missing assets and non-HTML requests do not silently become SPA HTML.
Tabs, replay selection and ply are reloadable deep links; autoplay is a local
viewer timer, not persistent game pause. Light/dark preference uses localStorage.

The completed path is Run → checkpoint → Arena aggregate → evaluation/opponent
→ recorded game → selected ply → Lab transition → exact context import →
explicit analysis → return to replay at the same ply. A selected analysis model
is hypothetical research selection and never rewrites the recorded actor.

## Additive API contracts

The existing dashboard API remains in [app](../training-dashboard/backend/app.py);
the four new routes below are GET-only evidence operations. They do not perform
neural inference or search. Unknown required identity/evidence returns 404;
invalid/conflicting context returns 422; unavailable runtime/import/I/O evidence
returns 503. FastAPI also rejects a negative `ply` with 422. Existing delegated
run APIs retain their own 400/404/503 behavior rather than adopting a new global
error contract.

| Endpoint | Request | Response contract |
|---|---|---|
| `/api/platform/models` | No request-selected file paths or verification flag. | `{models, count, sources, warnings, verification_requested:false}`; listing does not hash/load every checkpoint. |
| `/api/platform/models/{sha}` | Exact 64-character lowercase hexadecimal SHA. | One model row plus `sources`, `warnings`; hashes the requested checkpoint evidence with a bounded stat-identity hash cache. Missing/invalid model key returns 404. |
| `/api/platform/runs/{run_id}/evaluations` | Existing provider-resolved run identity. | `{run_id, benchmarks, replays, warnings}`; canonical benchmark/replay run IDs must agree. No new games. |
| `/api/platform/context` | Optional `run_id`, `replay_path`, integer `ply=0`, full `model_sha`. | `AnalysisContext` described below, validated from original provider replay evidence and Rust transitions. |

A model row includes `id=sha256`, `aliases`, flattened `metadata`, `architecture`,
`output_semantics`, `conflicts`, raw source `evidence`, `checkpoints`, `registry`,
`agent_specs`, exact-SHA `run_refs`, capability flags, `deployment_evidence`,
`requires_isolated_research_catalog`, `actual_load_verified` and per-flag `reasons`.
Names/aliases are display evidence, never identity joins. Conflicting metadata
or one alias mapping to multiple SHAs blocks eligibility rather than guessing.
Local evidence resolves only within explicitly approved roots; request parameters
cannot supply arbitrary catalog/checkpoint paths or remote artifact URLs.

### Capability flags are not operational proof

- `available`: at least one approved local checkpoint file exists.
- `sha_verified`: a requested local hash matches the exact SHA; listing normally
  leaves this false. It is not deserialization or inference verification.
- `analyzable`: static nonconflicting available CNN/MLP, supported
  `dqn_obs_v1`/`dqn_obs_cnn_v1`, `dqn_action_v1`, and 209-action metadata.
- `playable`: static analysis eligibility plus an available SHA-pinned DQN
  AgentSpec with `top_k`, `k=1`; not an arbitrary-checkpoint session capability.
- `deployed`: requires trusted explicit observed deployment evidence and no
  blockers. Registry aliases and AgentSpec presence alone never establish it.
- `actual_load_verified`: always false in this evidence index; a separately
  observed successful Lab call must be reported as that call, not promoted to
  an all-model index claim. Reasons explicitly include static-only/load-unverified
  notes, missing SHA verification and any blockers.

DQN/DDQN outputs remain Q-values; policy-distillation outputs remain
`policy_logits`. A Lab request's historical `algorithm:"dqn"` dispatch does not
turn a policy checkpoint's logits into Q-values. Win rate and score rate
(`(wins + draws/2)/games`) and replay samples versus aggregate results remain distinct.

### AnalysisContext and exact identity semantics

[Context resolution](../training-dashboard/backend/platform_context.py) returns:

- `schema_version:"analysis_context_v1"`;
- `position:{pawns:[[x,y],[x,y]], walls_remaining:[n,n], walls:[{x,y,orientation}], turn:1|2}`;
- `position_hash`, `source`, `actual_action`, `recorded_model`, `analysis_model`,
  internal `return_route`, and `warnings`.

`position_hash` is SHA-256 of UTF-8 JSON with sorted keys, compact separators,
no NaN, and canonical wall order. It identifies the complete position including
turn and stocks, not just pawn geometry. `ply` is a **state offset**:
`states[ply]` is before `actions[ply]`; at terminal ply `actual_action=null`.
The entire trajectory must have `states=actions+1`, a matching `game_length`,
explicit initial actor evidence, and every action/snapshot must agree with Rust.
Turn is never guessed from ply parity. Actions use `{type,target:[x,y],player}`;
walls use explicit coordinates/orientation, never unsafe JavaScript 64-bit masks.

Recorded replay `source` carries `{type:"recorded_replay", run:{source,run_id},
evaluation, game, replay_path, ply, evidence}`. Provider identity is `legacy` or
`mlflow` plus the canonical provider run ID; display names are not joins.
Evaluation ID is a canonical content hash of manifest SHA when present,
otherwise exact run plus explicit evaluation ID; absent evidence remains null
with a warning. Game ID hashes canonical states/actions, with recorded game index,
trajectory/opening hashes retained separately. An opaque replay key is not itself
a game content hash. Duplicate index rows with contradictory identity fail closed.

`recorded_model` comes only from agreeing full checkpoint-SHA evidence in the
replay/index/source. It is not inferred from filenames, aliases or paths.
`analysis_model` is the explicitly selected, SHA-verified eligible model, or the
recorded model when resolvable; it can differ without altering recorded evidence.
Missing/unsupported/conflicting chosen model evidence fails closed. With no
replay, only `ply=0` and no `run_id` are allowed; context is the engine's standard
initial position with `source:{type:"standard_initial",ply:0}` and no actual action.
A run requires an explicit replay key; a replay key requires a run.

Lab import validates the position through its engine endpoint but sends **no
automatic scoring/search/neural analysis**. It pins neural requests to the exact
available catalog SHA/ID and never falls back to a name, prefix, default model,
or opening position on an invalid context. Source context remains attached to
exports; edited positions are separate hypothetical analysis.

### Isolated proxy contract

[The gateway](../training-dashboard/platform_entry.py) accepts only a configured
`http://127.0.0.1:<unprivileged-port>` upstream, never a browser-selected URL.
GET/HEAD allowlist: Lab root, `/health`, `/api/meta`, `/api/models`, `/assets/…`.
POST allowlist: `/api/position`, `/api/analyze`, `/api/branch`, `/api/apply`,
`/api/neural/tensor`, `/api/neural/search`, `/api/multi-compare` beneath
`/research-lab`. These are isolated position/analysis operations, not live-game
or registry writes. Other paths/methods fail, redirects are not followed, and
incoming cookies/authorization/Host are not forwarded.

Bounds: path plus query ≤8192 characters (otherwise 414), body ≤1 MiB (413),
response ≤16 MiB (502), eight concurrent proxy slots (busy 429), 120-second
operation deadline (504); transport failure is 502. Lab decoding separately
allows at most 64 KiB decoded context, but the gateway's **8 KiB URL bound is the
effective tighter navigation limit**. Large provenance can therefore make a
context link unusable; there is no server-side context token/store fallback.

## Acceptance evidence

External evidence directory (not in Git):
`/home/jihua/quoridor-data/platform-p1-p2/acceptance/live-e2e-20261022-final`.
Its `results.json`, HTML report and screenshots/JSON attachments were read.
Final real Chromium/Playwright result: **3/3 passed, 75.572 seconds**, zero
skipped, unexpected or flaky tests; all attached browser/page error and force-click
lists were empty. Earlier attempt directories are not the final acceptance.

The [live test](../training-dashboard/frontend/tests/e2e/live-platform.spec.mjs)
uses real already-running services without mocks, replacement responses, service
startup, training, registry writes or generated games. It covers both providers,
exact run metrics (60 points), checkpoint/Arena/Greedy replay navigation,
autoplay and pause, ply reload/back, context import without analysis, a real UI
analysis click, exact exported result, return navigation, reverse model→run
lineage, and persisted light/dark themes. Health checks are GET-only; this does
not claim that an old live-game API was exercised by starting a game.

Exact target: run `40f0edbd60314c7ab822b6dfbd8522ef` (`C-resnet_seed-42`),
checkpoint `update012000.pt`, SHA
`0faaae51ded5b88ed5e0b3d8774a6e5de954bf30a5ec72e4d8007afe344a15fc`.
Live GET additionally confirmed **10 benchmark rows, 260 games and 260 replay
entries** (250 ranked games plus 10 controls). Greedy paired aggregate is
38W / 0D / 12L with 50 replay links, not a newly generated evaluation.

The accepted imported ply-7 position is:

- pawns `[[4,3],[4,5]]`, stocks `[9,8]`, turn `2`;
- horizontal wall `(5,1)`, vertical walls `(6,1)` and `(7,2)`;
- position hash `87961fa44599de0b0fc92867d15908db0f66860f1b9b8f090783bc0e906a7972`;
- recorded next action `{type:"pawn",target:[4,4],player:2}`.

The imported export had `analysis:null`. Only the explicit click produced
`POST /research-lab/api/analyze`, HTTP **200**, the same position and full neural
SHA, **121 legal candidates**, and `output_semantics:"policy_logits"`. This is
actual successful loading/analysis for this model/call only. Its index row still
correctly reports `analyzable:true`, `sha_verified:true`, `playable:false`,
`deployed:false`, `actual_load_verified:false`.

Final focused validations:

| Validation | Result |
|---|---|
| Dashboard backend unittest discovery | 119 tests run, 118 passed / 1 skipped (optional external replay test). |
| Decision Lab Node tests | 38 passed. |
| Dashboard Node tests | 21 passed. |
| React UI Vitest tests | 18 passed. |
| TypeScript checking / Vite production build | Passed. |

Node was **22.22.0** (the “18 UI” figure is a test count, not Node 18).
Node emitted the existing module-type warning; it did not fail tests. Final
Playwright evidence was inspected, not rerun or overwritten by this docs task.
Additional GET checks returned HTTP 200 for 8751 health/model/evaluation APIs
and existing 8741 dashboard, 8744 Lab and 8743 Agent Service health.

## Reproduce build, launch and tests

Use an approved Training environment with engine bindings, PyTorch, dashboard
and Lab dependencies already installed. Node 22 is the verified frontend runtime.
Dependency/browser installation is explicit, never performed by the launcher.
From this worktree:

```bash
cd /home/jihua/workspace/boardgame-ai-arena-platform
npm --prefix training-dashboard/frontend ci
npm --prefix training-dashboard/frontend run typecheck
npm --prefix training-dashboard/frontend run build
```

Set `VITE_PLAY_URL` to the existing Dev frontend's HTTP(S) URL **during build**
if the external Play link is wanted. Start only when isolated ports are free;
the current live supervisor already occupies them, so do not launch a duplicate:

```bash
QUORIDOR_PYTHON=/home/jihua/workspace/boardgame-ai-arena/.venv/bin/python \
  bash scripts/start_unified_platform.sh
```

Optional launcher configuration: `PLATFORM_HOST`, `PLATFORM_PORT`,
`PLATFORM_LAB_PORT`, `PLATFORM_FRONTEND_DIST`, `MLFLOW_TRACKING_URI`,
`TRAINING_DASHBOARD_SOURCE_ROOT`, `TRAINING_ARTIFACTS_DIR`, and
`TRAINING_DASHBOARD_ARTIFACT_ROOTS`. Terminate the foreground supervisor with
Ctrl-C to stop both children; no production rollback/replacement is involved.

```bash
PYTHONPATH=training-dashboard \
  /home/jihua/workspace/boardgame-ai-arena/.venv/bin/python \
  -m unittest discover -s training-dashboard/backend -p 'test_*.py'
npm --prefix decision-lab test
npm --prefix training-dashboard/frontend test
npm --prefix training-dashboard/frontend run test:ui
npm --prefix training-dashboard/frontend run typecheck
npm --prefix training-dashboard/frontend run build
```

Optional backend external replay coverage sets `ARENA_REPLAY_TEST_ROOT` to
`/home/jihua/quoridor-data/experiments/cnn-architecture-20261021-213018/arena-v2`;
the recorded 119-test run above omitted it and had one skip. For a **new** live
acceptance run, keep evidence separate from the final directory above:

```bash
cd training-dashboard/frontend
npx playwright install chromium
E2E_BASE_URL=http://192.168.68.88:8751 \
  E2E_RESULTS_DIR="/home/jihua/quoridor-data/platform-p1-p2/acceptance/live-e2e-$(date -u +%Y%m%dT%H%M%SZ)" \
  npm run e2e
```

[E2E instructions](../training-dashboard/frontend/tests/e2e/README.md) document
the existing host-local Chromium library workaround and optional
`E2E_CHROMIUM_PATH`. Tests are sequential, without retries, and record real
network traffic. Do not add generated reports, screenshots, logs, catalogs,
checkpoints or replay data to Git.

## Limitations and P3/P4 prerequisites

- Trusted LAN only: plain HTTP, no authentication, authorization boundary or TLS.
  The loopback proxy allowlist is not a substitute for user permissions.
- Huge source/run/model listings remain slow. There is no shared listing cache
  or pagination; the checkpoint hash cache does not solve repeated provider scans.
- URL-carried contexts are size-bound and exposed in browser history; no durable
  context/session store, arbitrary checkpoint session creation, or persistent
  session pause exists. Recorded replay autoplay pause is not game-runtime pause.
- `/lab` is a React **transition** to the existing Lab, not a full React migration;
  Play remains a separate external entry. Not all 92 indexed models were loaded,
  inferred, playable in sessions, or observed deployed. Metadata eligibility is
  not tested behavioral compatibility.
- Missing historical trajectories, metric semantics or checkpoint evidence remain
  unavailable/warned; no regenerated game is presented as original evidence.

**P3 dependencies (not completed):** a session runtime that can pin an exact
immutable model SHA and resolved AgentSpec without moving production aliases;
explicit permission/resource controls for session creation and model execution;
and safe coordinate wall DTOs end-to-end (never lossy 64-bit JS masks), with Rust
authoritative for transitions. Do not infer this runtime from `playable` metadata.

**P4 dependencies (not completed):** pause/resume generation/revision checks to
reject stale in-flight actions, isolation of sessions and hypothetical analysis,
and React wrappers around existing analysis APIs preserving their semantics and
explicit execution controls. These require the P3 runtime/identity/permission
boundary; UI wrappers alone do not provide persisted pause or safe concurrency.

## Handoff

Implementation commit: `be5b20096a1b717bb1239f9ec981267017642b07`, pushed normally
to `origin/feature/unified-platform-p1-p2`; no PR created. Dev fetched that
commit through Git into `/home/jihua/workspace/quoridor-platform-p1-p2` on
`review/unified-platform-p1-p2`. Its existing primary checkout remains clean
on `integration/dev-training-20261021` at `69b495f`; production Backend health
remained OK and container `AGENT_SERVICE_URL` remained
`http://192.168.68.88:8743`. Training's feature worktree was clean after the
implementation commit; the final 119-test backend regression still passed
with the same single optional skip. Dev has the source for review, not a new
deployment or a separately verified Dev frontend build.

All implementation, tests, launcher and documentation belong to the isolated
`feature/unified-platform-p1-p2` branch. No PR, shared-baseline promotion,
production replacement, registry mutation, training or new tournament occurred.
Original dirty Training files and clean Dev integration checkout are preserved.
External acceptance/catalog evidence stays under
`/home/jihua/quoridor-data/platform-p1-p2/`, outside Git. The test service is a
foreground supervised process: keep its terminal alive; deployment hardening
and durable service installation need separate approval.