# Unified platform P3-A: checkpoint-backed human play

**Verified:** 2026-10-22. This records an isolated implementation and deployment,
not promotion into the shared integration branch or production replacement.
The [system architecture](../documents/system/design/system-architecture.md)
and [Agent System design](../documents/system/design/agent-system-design.md)
remain authoritative.

## Delivered workflow and source

Training -> run -> Checkpoints -> Play against -> human versus the exact CNN
checkpoint, within the existing React research shell. Models provides the same
full-SHA deep link. The board, state mapping and action translation reuse Dev
frontend components; gameplay uses the existing GameManager and Rust rules.
There is no iframe, second rule implementation, global roster change or alias
fallback. DQN outputs remain Q-values; policy-distillation outputs remain logits.

- Branch: `feature/unified-platform-p3-a`, based on P1/P2 `7ec538d`.
- Training worktree: `/home/jihua/workspace/boardgame-ai-arena-play`.
- Dev worktree: `/home/jihua/workspace/quoridor-play-p3-a`, branch
  `preview/unified-platform-p3-a`, fast-forwarded from the feature branch.
- Implementation: `a934aeb5acc69257bdb29a60229f7efb5cd30c06`.
- Runtime/game safeguards: `5db06536d55383c663325b4ed4c989bdf126d570`.
- Restart preflight: `ccd449be6444901547f68f45dcd3cc4bb61fa91c`.
- Final frontend: `945cda7d2658f41edc1f3270c89b8466b199d3c5`.
- The subsequent handoff commit contains documentation and explicit E2E
  evidence-loading waits only. All source is pushed; no PR was opened.

Acceptance target:

- Run: `40f0edbd60314c7ab822b6dfbd8522ef`, `C-resnet_seed-42`.
- Checkpoint: `update012000.pt`, SHA-256
  `0faaae51ded5b88ed5e0b3d8774a6e5de954bf30a5ec72e4d8007afe344a15fc`.
- Entry: `http://192.168.68.88:8752/training/runs/40f0edbd60314c7ab822b6dfbd8522ef?tab=checkpoints`.
- Direct selection: `/play?model_sha=<full-sha>`; starting requires an explicit
  click and choice of human P1/P2. URL selection never changes an existing game.

## Actual isolated deployment

| Service | Address | Isolation |
|---|---|---|
| React platform and same-origin gateway | `192.168.68.88:8752` | New preview, not 8751 replacement. |
| Dev Gameplay | `192.168.68.102:8774` | `quoridor-gameplay-p3-a`, research routes only. |
| Training research Runtime | `192.168.68.88:8753` | Research endpoints only, CPU inference. |
| Existing isolated Lab | `127.0.0.1:8754` | Reused through `/research-lab`, not production 8744. |

Training runs under transient user unit
`quoridor-play-p3-a-preview-v3.service`, with restart-on-failure, 2 GiB memory
limit and 150% CPU quota. [The launcher](../scripts/start_research_play.sh)
supervises both children, cleans them up on termination and exits if either
exits. Its port probe uses `SO_REUSEADDR`: real TCP checks proved it permits
TIME_WAIT restarts while rejecting an active listener. This is not an installed
reboot-autostart service. The Dev container has restart policy `no`.

Dev reuses backend image
`sha256:0db9353472a736d80890ef563a0f37df68e973928c945cb7b55a5beb070616e3`,
with the separate source worktree mounted read-only at `/research` and
`PYTHONPATH=/research:/research/backend-server`. Command:
`uvicorn backend.main:app --host 0.0.0.0 --port 8764 --log-level info`.
It binds only `192.168.68.102:8774:8764`, forwards to Training 8753 and sets
`QUORIDOR_RESEARCH_ONLY=1`. It uses a read-only filesystem, 64 MiB tmpfs,
no additional capabilities, no-new-privileges, 64 PIDs, 768 MiB memory and one CPU.
The existing Dev 8764 -> Training 8743 chain is unchanged.

Build with Node 22 and the original Training Python environment:

```bash
cd /home/jihua/workspace/boardgame-ai-arena-play/training-dashboard/frontend
npm ci
npm run typecheck
npm run build
cd ../..
export QUORIDOR_PYTHON=/home/jihua/workspace/boardgame-ai-arena/.venv/bin/python
export QUORIDOR_RUNTIME_VERSION="$(git rev-parse HEAD)"
export QUORIDOR_RESEARCH_SERVICE_KEY="$(</home/jihua/quoridor-data/platform-p3-a/private/service-key)"
bash scripts/start_research_play.sh
```

The foreground command is an alternative to the running unit, not a duplicate
launcher. Stop only the isolated unit before launching another instance.
Both Gameplay and Runtime require the same operator-provisioned service key;
never print it or put it in source, URLs, screenshots or handoffs. Transfer it
to Dev through authenticated stdin/environment, not a checked-in config.
The key file is outside Git, mode 0600 under a mode-0700 private directory.
Existing unit restart: `systemctl --user restart quoridor-play-p3-a-preview-v3.service`.
Restart destroys in-memory sessions; it is not game recovery.

Rollback only stops that unit and `docker stop quoridor-gameplay-p3-a` on Dev.
Do not stop production 8740-8744/8764-8765 or the P1/P2 unit. Stopping P3-A does
not require changing production configuration, catalogs or champion aliases.

## Contracts, identity and safeguards

[ResearchSessions](../agent_system/runtime/service/research.py) consumes an
operator-generated approval snapshot from existing catalogs. The observed file is
`/home/jihua/quoridor-data/platform-p3-a/catalogs/7c450dd96e1fcfe9c8aa204904699dceb1d43f380e6c2f597d48c4e47bb759fd.json`.
[Preparation](../scripts/prepare_research_play.py) writes only this external
content-addressed research material, never the production catalog or weights.
`research_playable` is static eligibility, distinct from production `playable`;
it does not assert every listed checkpoint actually loads.

Runtime requires exactly one approved full SHA, an existing file inside approved
roots, matching file hash, CNN encoder `dqn_obs_cnn_v1`, action version
`dqn_action_v1`, 209 outputs, nonconflicting algorithm/configuration and supported
output semantics. It creates an AgentSpec using the existing loader family
`algo_type=dqn`, deterministic `top_k`, `k=1`, CPU. The actual checkpoint
algorithm, SHA, encoder, configuration and output semantics are checked after
loading. Loader dispatch is not relabeling policy logits as Q-values.

Limits: four Runtime instances, four-hour session lifetime, 128 MiB checkpoint,
20 million network parameters, one concurrent inference and one CPU thread per
numerical library. Gameplay holds at most 16 in-memory game records and ends a
game explicitly at 1024 plies as a resource limit, not a fabricated goal/draw.
There is no unbounded per-user model cache. Session tokens bind room and agent
seat; wrong-turn inference is rejected. Missing/mismatched/busy models fail
closed, without substitution.

| Boundary | Contract |
|---|---|
| Runtime `/agent/research/create`, `/action`, `/destroy` | Private Bearer service key; per-instance opaque session/token plus room and seat binding. |
| Gameplay `/api/research/games` | Create `{sha256,human_seat}`; private `X-Research-Gateway` header required. |
| Gameplay `/{game}` | GET authoritative snapshot; per-game Bearer capability required in addition to gateway authorization. |
| Gameplay `/{game}/actions` | POST `{expected_step,type,target:[x,y]}`; stale revision 409, illegal Rust action 422. |
| Gameplay `/{game}/retry`, `/surrender` | Explicit commands; retry does not advance the failed position first. |
| Gameplay `/{game}` DELETE | Release session and remove the game/history; no durable archive. |
| Platform `/play-api/games...` | Fixed allowlisted Dev target and path/method set, same-origin write check, 8 KiB requests, eight concurrent requests, 40 s upstream timeout, 2 MiB responses, no redirects/cookies. |

Browser-visible snapshots include initial and current state, actions, revision,
result, human/agent seats and verified identity: requested/actual full SHA,
architecture, actual algorithm, output semantics, model configuration, resolved
AgentSpec, observation/action versions and Runtime code version. Runtime private
tokens/service credentials are never returned. The browser stores only its
per-game capability in localStorage, not in the URL.

The engine adapter emits explicit wall coordinates/orientations and decimal
strings for 64-bit masks. Bit 63, both coordinate directions, turn, stocks and
P2 mapping were tested. Browser moves are never optimistic: Rust legal actions
drive highlights and Rust transitions decide acceptance. Identity changes in an
agent response are rejected before applying the action. Agent timeout or network
failure leaves the position unchanged and exposes explicit retry; it never invokes
Greedy or another checkpoint. Runtime calls use a 5 s action timeout and a 30 s
load timeout; timeout does not promise cancellation of already-running CPU work.

Refresh/navigation/reconnection reads the same in-memory game using its stored
capability. Polling reconnects without recreating the game. Active games do not
depend on model-directory requests. Server restart, expiry, missing capability
or closing a game cannot be recovered as a new game silently.

Security is bounded research authorization, **not** user login, tenancy or TLS.
Any trusted-LAN visitor can request eligible instances within global limits;
localStorage capabilities require a trusted browser/origin. Public deployment
needs authentication, TLS, per-user quotas and stronger credential management.
Model-directory scans can take tens of seconds; there is no new source-list cache.

## Verified evidence

Final browser evidence: `/home/jihua/quoridor-data/platform-p3-a/acceptance/live-final-v5`.
Six real Chromium tests passed in **184.838 s**, zero skips, unexpected failures
or flaky retries. Real HTTP, actual CNN weights and Rust transitions were used;
no intercepted model/game responses or forced UI clicks.

| Human | Game | Plies / CNN actions | Result |
|---|---|---|---|
| P1 | `1b4a39a6-51f5-4945-9287-133f65fe3374` | 36 / 18 | P2 goal |
| P2 | `663cf4fa-6502-400a-adc1-43e478c67fd8` | 39 / 20 | P1 goal |

Both games retained the exact C-resnet SHA above and Runtime version
`ccd449be6444901547f68f45dcd3cc4bb61fa91c` throughout. The frontend is `945cda7`;
later E2E-only waits and documentation do not change running model code.
Attachments include complete game JSON, network audit, browser-error assertions
and desktop/mobile screenshots. Tests cover wall bit 63/stocks, navigation,
refresh, offline recovery, a changed next-game SHA not changing the current game,
P2 opening, illegal/stale rejection, unsupported models and preserved services.
The three existing P1/P2 tests also passed, including replay ply-7 Lab import and
explicit neural analysis. These games are functional acceptance, not win-rate
benchmarks.

Additional checks actually run:

- Backend/Runtime/AgentSpec pytest: **81 passed, six subtests passed**, with the
  external real-load smoke enabled. D2, A-baseline/B-deep/C-resnet/D-wide seed-42
  and `cnn_observation_2k` all loaded and returned Rust-accepted actions.
- Dashboard: **121 unittest tests run**, one optional external test skipped.
- Lab Node: **38 passed**; Dashboard Node: **21 passed**.
- React UI: **22 passed**; TypeScript and Vite production build passed.
- Startup: real TIME_WAIT reproduction, successful rebind and rejection of an
  active listener; shell syntax check passed.
- Existing Dev Play was exercised with human P1 versus Greedy, a valid move,
  agent response and surrender result. Production health and protected checkpoint,
  catalog, registry and AgentSpec hashes were checked unchanged.

Reproduce browser checks using `npm run e2e` from the Dashboard frontend with
`E2E_BASE_URL=http://192.168.68.88:8752`,
`E2E_RUNTIME_VERSION=ccd449be6444901547f68f45dcd3cc4bb61fa91c`, and a **new**
external `E2E_RESULTS_DIR`. Use the actual version of a newly started Runtime,
not this historical value, after a redeployment.
Python checks use the original explicit venv, `PYTHONPATH=backend-server`, and
`pytest backend-server/tests/units agent_system/runtime/tests agent_system/definition/tests -q`;
set `RESEARCH_PLAY_SMOKE_CATALOG` to the approval snapshot for real-load tests.
Dashboard checks use unittest discovery under `training-dashboard/backend`;
Node checks use `npm test`, React `npm run test:ui`, types `npm run typecheck`.

Earlier failed attempts are retained separately: `live-final` was connection
refused before the TIME_WAIT fix; v2 exposed the catalog/offline error; v3/v4
exposed premature 15 s evidence-loading waits. They are not passing evidence.
`live-play-attempt1`, `p2-loading-check` and `catalog-loading-check` retain earlier
scoped successes. No evidence, replay, weights or private credentials enter Git.

## P3-B / P4 prerequisites, not implemented

Pause/resume needs an explicit server phase and generation/revision contract:
stop scheduling, invalidate or cancel in-flight work, and compare the captured
revision again before applying an agent action. A stopped viewer timer is not
game pause. Resume must not duplicate or accept stale actions.

A complete immutable live-analysis DTO needs game ID, step/generation, pawns,
explicit walls, stocks, active seat/turn, phase/result, initial state, action
provenance, model SHA/config/AgentSpec, Runtime version and canonical position
hash. Restoring a snapshot must be validated by Rust, not direct arbitrary
Python state mutation. This future DTO is not a new P3-A persistence guarantee.

Live Lab analysis should capture an immutable snapshot and bind results to
game/revision/position hash. Analysis runs in a separate branch, cannot mutate
the live game and must label another selected model as hypothetical. Existing
P1/P2 recorded-replay import remains separate; P3-A does not implement a live-game
Lab button or automatic live scoring.

Durable human-game history and restart recovery additionally need authenticated
ownership, durable storage, schema/version migration, capability lifecycle and
replayable engine-validated transitions. None of pause/resume, live analysis,
durable human history or restart recovery is delivered by this phase. No training,
new benchmark, champion change or production service replacement was performed.