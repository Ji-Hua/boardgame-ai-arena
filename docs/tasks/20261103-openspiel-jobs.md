# OpenSpiel research jobs
Status: implemented and isolated-container acceptance passed; not deployed
Branch: codex/multi-game-rl-platform
Base: e61cc63

## Scope
Continue the multi-game design with an optional CPU OpenSpiel worker profile and
Kuhn Poker end-to-end research Jobs. Reuse OpenSpiel tabular Q-learning and CFR
rather than implementing optimizers. CFR is a solver reference, not RL.
The initial RLlib suggestion remains deferred until a neural/memory workload
justifies it; this smaller backend exercises the same platform boundaries.

## Acceptance
Explicit game/profile compatibility; disabled without an image; validated bounded
configs; project-aware UI; policy probabilities exported with hash and provenance,
reloaded for matches; exact exploitability metrics; CPU container smoke in an
isolated data root. Legacy Quoridor Jobs, models and services remain unchanged.
No deployment or service switch.

## Implementation
Source commit f5ead7252f347b26b4494da82d2a203fad3cf160.
Optional JOBS_OPENSPIEL_IMAGE, resolved to immutable local image ID at submission.
Entrypoints filter by project GameRef. UI disables unconfigured profiles.
Research metrics preserve solver iterations or episodes. Exported JSON policies
carry package versions, source/image/config hashes and game/project identity.
Controller validates SHA and provenance before success; no legacy .pt registration.
Exact exploitability and NashConv use OpenSpiel, not a local reimplementation.

## Validation
- Python Games / Jobs / Arena matrix suites: 47 passed (final run).
- React: 75 tests passed; TypeScript and Vite production build passed.
- Docker build succeeded; isolated real Jobs passed CFR and tabular Q-learning,
  controller reconstruction/reconciliation and cancellation.
- Both algorithms ran for 100 progress units, seed 42. These are functional smoke
  checks, not equal-budget comparative research or a strength claim.
- Real acceptance evidence on Dev:
  /tmp/quoridor-openspiel-acceptance-20261103-b/acceptance.json
- Image ID:
  sha256:1193ba92514a1ebed862e6148cbde332ddef80c15c93e1cd512c68c26b060f11
- Initial acceptance attempt a pinned the preceding e61cc63 before the new
  source commit existed and failed with unknown entrypoint, as expected for an
  incompatible snapshot. Attempt b explicitly used f5ead72 and passed.
- python3 scripts/project.py check fails because deploy/compose.dev.yaml does not
  exist in the inherited e61cc63 baseline. This task did not alter release tooling.
- No production service, Training job, MLflow server or roster was changed.

## Remaining
No RLlib/recurrent neural backend, generic immutable policy catalog, policy download
endpoint, public hidden-information replay or live multi-game Play UI is claimed.
Mr. Jack rules/version and its research protocol remain subsequent work.
The optional CPU image was built only on Dev, not published or installed on Training.
