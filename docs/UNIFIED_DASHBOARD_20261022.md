# Unified Training Dashboard handoff

Document type: Implementation / deployment record. Scope: Training dashboard,
read-only experiment evidence, replay-to-Decision-Lab integration.

## Source and boundaries

- Branch: `feature/unified-training-dashboard`.
- Worktree: `/home/jihua/workspace/boardgame-ai-arena-unified-dashboard`.
- Based on shared integration `69b495f`, merged existing architecture benchmark
  branch through `a9078f8`; no PR or shared-branch promotion is part of this task.
- The original Training worktree stays on `codex/environment-deployment`.
  Its modified tracked experiment outputs, untracked artifacts, and local
  Decision Lab edit were not overwritten, cleaned, or committed.
- Dev SSH (`jihua@192.168.68.102`) rejected available authentication. Shared
  checked-in Dev/Training instructions were read on Training; no claim is made
  that Dev services or tests were verified remotely.

## Metric investigation

Experiment `9`, `Quoridor / CNN architecture benchmark`, has 12 checkpoint
registration runs without recorded MLflow training metrics or parent tags.
The architecture training runner writes external files but does not log those
curves to Tracking. Checkpoint run `40f0edbd60314c7ab822b6dfbd8522ef`
(`C-resnet_seed-42`) has SHA
`0faaae51ded5b88ed5e0b3d8774a6e5de954bf30a5ec72e4d8007afe344a15fc`.
Training evidence is below
`/home/jihua/quoridor-data/experiments/cnn-architecture-20261021-213018/matrix/C-resnet/seed-42/`.

- 60 original training points: updates 200–12000, policy cross-entropy,
  training batch action agreement, maximum absolute policy logit.
- Final training cross-entropy: `0.7306727170944214`; agreement: `0.75`.
- Held-out agreement: `0.5491049289703369`; cross-entropy:
  `1.6493388414382935`, one snapshot only. Dataset SHA:
  `05ab2d865b60f54620a5405a7b31d293de5b1d6fd75a6e257ea92fe3c586a787`.
- Historical timestamps, missing checkpoints, unknown protocols or absent
  trajectories are not guessed. Policy logits are not Q-values.
- Benchmark parent `3dbe1422000b4475967cc5aeb204bc58` and evaluation child
  `208d8762689b4c828f85836ca8c0f697` are evaluation evidence, not training tasks.
  The child score is `0.626`, whereas its win rate is `132/250 = 0.528`.

## Evaluation/replay evidence

The pilot's original `arena-v2` manifest, report and gzip blocks are mounted
read-only. Exact roster/checkpoint SHA ties evaluations to the training task.
Seed 42 has 250 ranked games (five opponents × 50) and 10 separate controls.
Greedy ranked results: 38W / 0D / 12L. Minimax D2: 1W / 7D / 42L,
win rate 0.02 versus score rate 0.09. These are from the same frozen report;
no evaluation is rerun.

Replay conversion applies recorded actions with Rust, validates opening,
trajectory, manifest and final-state identities, and sends safe coordinate
snapshots to the existing replay board. Full 64-bit wall masks do not pass through
JavaScript as imprecise numbers. Forfeits/cycles retain original outcomes.
An old three-replay export belongs to A-baseline seed 42 and is not attached to
C-resnet merely because checkpoint basenames match.

## Deployment

- Dashboard: `http://192.168.68.88:8741`; API/health on 8740 and through frontend
  proxy. Docker Compose project `training-dashboard`, container
  `training-dashboard-training-dashboard-1`, source is the isolated worktree.
- Preserve legacy host mount using
  `TRAINING_ARTIFACTS_DIR=/home/jihua/workspace/boardgame-ai-arena/agent_system/training/artifacts`.
  External root remains `/home/jihua/quoridor-data`, read-only. Build uses
  `docker compose -p training-dashboard -f training-dashboard/docker-compose.yml up -d --build --wait`.
- Decision Lab remains at 8744. An existing `multiCompare` scoping error stopped
  all page initialization, including imported replay positions; fixed with a
  real module-startup regression test and script cache revision.
- User systemd drop-in
  `/home/jihua/.config/systemd/user/quoridor-decision-lab.service.d/unified-dashboard.conf`
  points `WorkingDirectory`/`ExecStart` at this isolated worktree and sets
  `QUORIDOR_PYTHON` to the original project venv. The existing catalog drop-in
  is unchanged. This avoids overwriting the original dirty source. To roll back
  this source selection, remove only that new drop-in and reload/restart the
  user service; doing so also restores the old broken frontend until separately
  fixed. No Agent Service, roster, champion alias or MLflow service was changed.

## Validation

Commands use the configured Training venv, not system Python:

- `PYTHONPATH=training-dashboard ARENA_REPLAY_TEST_ROOT=/home/jihua/quoridor-data/experiments/cnn-architecture-20261021-213018/arena-v2 /home/jihua/workspace/boardgame-ai-arena/.venv/bin/python -m unittest discover -s training-dashboard/backend -p 'test_*.py'`.
- Frontend: `npm run typecheck`, `npm test`, `npm run build`.
- Decision Lab: `npm test`.
- Docker build, health checks, frontend proxy/API requests, browser acceptance
  and `git diff --check`.

Final results: **69 backend tests passed** with external evidence enabled;
**16 frontend tests**, TypeScript checking and production build passed;
**10 Decision Lab tests passed**. Container rebuild/health passed. API acceptance
listed 101 tasks (51 Legacy / 50 MLflow), 60 pilot training points, one held-out
snapshot, 260 pilot replay entries and a checkpoint with matched benchmark.
Browser acceptance exercised legacy playback/autoplay and cross-source Compare
including explicit refresh; earlier acceptance exercised pilot curves, Arena
playback and an actual Decision Lab imported P2 position with engine analysis.

Legacy `cnn_curriculum_observation_2k_001` has 32 valid replay entries out of 40.
Eight old episode-500 index entries point to overwritten files for a different
checkpoint and are explicitly unavailable. The original writer used configured
`agent_id` as both replay identities; this exact alias is accepted only with
matching recorded DQN config. Self-contained payload display metadata wins over
stale indexes after identity validation, with warnings, never on-disk repairs.
PyYAML is an explicit container dependency, not an assumed host-only package.

Original external replay validation covers 780 C-resnet games across seeds
42/43/44 (627 goals, 151 deterministic cycles, 2 ply limits). No models were
trained and no games were generated. See the final handoff Git commit/output for
the immutable source revision; source changes are committed on the named branch.

## Remaining limitations

- HTTP-only MLflow has no uploaded training history for this pilot: keep its
  exact external training evidence mounted to see curves. Other historical
  runs lacking both history and readable evidence truthfully show missing data.
- Unknown metric semantics/step units intentionally prevent Compare overlays;
  different evaluation protocols remain separate records, not strength claims.
- Historical incomplete metadata/trajectories cannot be repaired by rerunning
  and pretending those are original records.
- Dev authentication is needed for future Dev-side synchronization/verification.