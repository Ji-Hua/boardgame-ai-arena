# Research policy registry and evaluation
Status: implemented; isolated real-Docker acceptance passed; not deployed
Branch: codex/multi-game-rl-platform
Base: 70fecfa

## Scope
Complete policy reuse for the optional OpenSpiel profile: immutable byte-SHA
storage, project-scoped provenance, authenticated policy download, registered
policy evaluation Jobs and project UI. Legacy .pt registration stays separate.
No service deployment. This is not a generic neural-policy runtime or Mr. Jack.

## Acceptance
Idempotent registration, byte-integrity checks, cross-project rejection,
evaluation revalidates mounted policy bytes, fixed paired-seat protocol,
exact exploitability and reload equivalence, API/UI tests, real Docker round-trip.
No private replay download endpoint.

## Changes
- Atomic no-overwrite byte-SHA policy storage and idempotent project/job catalog.
- Registration occurs only on successful OpenSpiel training finalization.
- Project metadata listing, token-protected policy download and UI deep links.
- evaluate_research_policy validates both input identities in the project, then
  revalidates mounted bytes in its worker. Fixed paired-role protocol, exact
  exploitability, seeded reproducibility and private full-state replay records.
- Controller checks report integrity/provenance before success and on reads.
- No JSON-as-.pt registration, Quoridor season publication or production promotion.

## Verification
Implementation commit: a000d71e2a667076523d7dcae9ce732b52c7f995.
- PYTHONPATH=.:backend-server:engine .venv/bin/python -m pytest apps/jobs/tests
  agent_system/games/tests agent_system/evaluation/arena/tests/integration/test_matrix.py -q:
  50 passed.
- npm run test:ui: 77 passed; npm run typecheck and npm run build passed.
- git diff --check passed.
- Real Docker acceptance: two training jobs succeeded, both policies registered,
  evaluation against random succeeded, evaluation against a registered opponent
  succeeded, and long-running training cancellation succeeded. Controller
  reconstruction/reconciliation remains covered.
- External Dev evidence:
  /tmp/quoridor-policy-registry-20261103-a/acceptance.json
- Optional CPU image reused without dependency changes:
  sha256:1193ba92514a1ebed862e6148cbde332ddef80c15c93e1cd512c68c26b060f11
- Tiny 10-pair evaluations are functional checks, not strength benchmarks.
- API and UI behavior verified with FastAPI TestClient and jsdom respectively;
  no live browser deployment or production MLflow write was performed.
- The previously recorded inherited project.py Compose-path issue remains outside
  this change; no claim that repository-wide deployment checks pass.

## Remaining
No historical job backfill, policy promotion, generic neural bundle/runtime,
live multi-game Play or public hidden-information replay. Mr. Jack rule/version
selection and information-isolation contracts remain subsequent work.
