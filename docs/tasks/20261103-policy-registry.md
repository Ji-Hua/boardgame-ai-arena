# Research policy registry and evaluation
Status: implementing
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
