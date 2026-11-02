# Multi-game RL platform
Status: first implementation slice complete; remaining design phases pending
Owner: Codex
Branch: codex/multi-game-rl-platform
Base: feature/unified-rl-platform @ 772e88f (fetched by Git from Training)
Design: documents/system/design/multi-game-rl-platform.md

## Scope and compatibility
Implement the accepted design incrementally on Dev. Begin with executable game adapters, a general match/replay contract, Quoridor parity, and persistent project-scoped Jobs. Existing production services, historical hashes, model bytes and v1 records remain unchanged. No deployment or production training.

## Acceptance
- Existing Quoridor rules remain exclusively in the Rust engine.
- Game selection is explicit; unsupported games/versions fail closed.
- Sequential and chance transitions do not assume alternating players.
- Historical jobs remain in the legacy Quoridor project; new jobs carry validated scope.
- Same experiment name in two projects must not mix job listings.
- Focused tests and existing affected suites pass before commits.

## Implementation and verification
Completed:
- Explicit versioned game registry, Rust-backed Quoridor adapter, optional OpenSpiel Kuhn Poker adapter.
- Local decision/chance runner, independent agent RNG, private replay integrity checks and player-view reconstruction.
- Arena uses Quoridor adapter with a golden trajectory hash captured from unmodified base 772e88f.
- Persistent immutable projects, additive legacy Job migration, validated project/game metadata, scoped Jobs API and UI.
- Gate follow-up and model registration carry project provenance. New MLflow experiments use project namespaces; legacy names remain unchanged.
- Project creation uses the existing platform write-token authorization.

Verified on Dev, Python 3.12 / OpenSpiel 2.0.2 / Node 22.23.4:
- Games, Jobs and Arena matrix integration pytest suites: 41 passed.
- npm run test:ui: 74 passed.
- npm run typecheck and npm run build: passed.
- git diff --check: passed.
- Real CLI smoke: Kuhn terminal after 3 decisions, returns +2/-2; Quoridor correctly truncated at 30 decisions. Both verified per-player replay reconstruction.
- Existing Dev checkout and service environments were not changed. Dependencies installed only into this worktree.

## Remaining phases and limits
This is an initial foundation, not completion of the full design.
- Kuhn Poker currently runs locally through CLI/tests; Jobs entrypoints remain Quoridor-only and explicitly reject other game projects.
- Add a supported generic environment/training adapter and worker capability declaration before exposing non-Quoridor training.
- Mr. Jack rules, hidden-state observations, role-specific policies and benchmark protocols remain to be implemented.
- Project scoping currently covers Jobs and new MLflow experiments, with model provenance metadata. Existing experiment/model/ladder/play pages are not fully project-scoped.
- Private replay files include hidden state and seeds; no public replay API was added.
- Browser behavior was tested with jsdom; no live browser/server deployment or new training benchmark was performed.
- Production image dependency changes and any service rollout require a separate deployment step.

## Deployment
Independent worktree /home/jihua/worktrees/quoridor/multi-game-rl-platform. No service switch.
