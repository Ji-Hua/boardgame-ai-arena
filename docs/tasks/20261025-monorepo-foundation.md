# Monorepo migration foundation
Status: verified (foundation only)
Owner: Codex
Branch: codex/monorepo-migration-20261025
Design: documents/system/design/monorepo-deployment.md

## Scope
Preserve old directories, integrate P3 and preserved Training fixes, establish agent-native documentation and image-only host deployment contracts.
Do not switch live services, delete old paths, merge main or move experiment data.

## Acceptance
Archive source comparison passes on both hosts. New migration branch has no conflicts.
Release preparation rejects mutable image tags, incomplete identity and missing contracts.
Both Compose models render without creating containers.
Merged domain regression and frontend checks pass.

## Implementation
Merge 18340a2 combines P3 and bc49843. The sole document conflict preserves historical P1/P2 context and Training correction evidence.
The expiry regression used monotonic timestamp zero, incorrectly assuming host uptime exceeds four hours; use an explicitly expired timestamp relative to the same clock. Production expiry semantics are unchanged.

## Deferred
Review old dirty Lab patch and historical backend_ai before semantic porting.
Runtime image construction, model-load acceptance, actual deployment and app-path relocation remain subsequent milestones.

## Observed verification
See ../MIGRATION_STATUS.md for commands/outcomes and archive checksums. No live deployment was performed.
