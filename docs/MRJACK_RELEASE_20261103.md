# Mr. Jack and Quoridor release candidate

User scope: Quoridor + original Mr. Jack base game only. No additional games.
Branch: codex/multi-game-rl-platform. Executable source: 475bc4e.
This supersedes MULTIGAME_RELEASE_20261103.md. No live deployment was performed.

## Scope correction
Removed the unrequested Poker game, its adapter/registry, solver and tabular policy
runtime, job entries, CLI, worker profile, Dockerfile and dependency declaration.
Shared tests now exercise Mr. Jack. The evaluator is mrjack_policies.py.
Unsupported historical project records are no longer exposed as active projects;
stored evidence is retained rather than destructively erased.

The research and worker images were rebuilt without OpenSpiel and checked for
absence of pyspiel. Only Quoridor and Mr. Jack are registered.

## Mr. Jack continuation
The board now accepts mouse clicks and keyboard activation on server-provided
legal move destinations. Legal destinations have green borders; spectator,
replay and busy states do not accept moves. Skill phases have Chinese labels.
Move legality and the eight characters' rules remain exclusively server-side.

Training, private observations, role-scoped replay/recovery and exact-policy
evaluation remain available. This continues the memoryless Maskable PPO baseline;
it does not claim strong play.

## Checks
- 67 games/Jobs tests, 11 Gameplay tests, 80 UI tests, 8 migration tests passed.
- TypeScript, production build, project checks and diff whitespace checks passed.
- Chromium passed with direct board interaction, watch restrictions, reload and replay.
- Offline read-only platform image completed a policy game without OpenSpiel.
- Worker dependency check confirms SB3-Contrib imports and pyspiel is absent.
- Real Docker training/evaluation/recovery/cancellation evidence:
  /tmp/mrjack-only-acceptance/acceptance.json.
- Browser evidence: /tmp/mrjack-only-browser/results.json.
- Image evidence: /tmp/mrjack-only-image-acceptance.json.

## Current candidate images
Local immutable image IDs; not published registry digests.

| Component | Tag | ID |
|---|---|---|
| Platform/Gameplay | quoridor-mrjack-platform:dev-20261103 | sha256:1c47369506241489d2230701cd541fea8041b449e07041247d9657676477b641 |
| Gateway | quoridor-mrjack-gateway:dev-20261103 | sha256:91ed26a2a62b3ede52ceb5be3f7ef1a825bbb394d518ede19ad8088a882417d8 |
| CPU worker | quoridor-mrjack-worker:dev-20261103 | sha256:6a17d64a58c70496c39004c0edea1394b75cc7487107a592be199fec506f0f71 |

Jobs uses JOBS_NEURAL_IMAGE for Mr. Jack. No optional Poker profile remains.
The worker runs a fixed Job source snapshot independently of its dependency image.
Projects -> Mr. Jack -> train_mrjack_ppo -> policies -> evaluate/play is the path.

Old multi-game candidate images are superseded and should not be deployed.
A future coordinated rollout must update Jobs, platform, Gameplay and gateway
while preserving current data and maintaining a single Jobs writer.
Existing research replay and gameplay databases remain separate from legacy
Quoridor data. No roster, production service or champion alias was changed.
