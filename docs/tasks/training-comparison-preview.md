# Compact training comparison and checkpoint navigation
Status: implementing
Branch: feature/agent-ladder-preview
Design: monorepo-deployment.md; existing metric semanticKey/comparisonGroups contracts.

## Scope
Preview-only frontend. Reuse existing MLflow/provider APIs and LineChart; do not alter training records, checkpoints, registry or services. Select up to five Runs using URL state. Compare only curves with matching semantic/unit/scope/progress-axis metadata. Unknown metadata is not silently aligned.

## Behavior
- Compact selectable Run list, curve selector and shared-domain overlays, nested configuration difference table, checkpoint list and raw evidence.
- Single-Run Agent detail links to the same training workspace.
- SHA-addressed checkpoint page discovers exact matching Agents in completed evaluation batches, retaining separate policies/batches; opponent selection leads to real match records, replay and exact-Agent Play.
- No guessed checkpoint associations by name or filename. Missing/failed evidence is explicit and retryable.
- Replay evidence fills the inspector body and includes the complete API response plus participant/evaluation/view context. Download retains the same data.

## Acceptance
Typecheck, React regression tests (selection, stale responses, semantic grouping/config diff, exact SHA), real-browser comparison/config/checkpoint/evaluation/replay/play and evidence sizing at desktop/mobile. Snapshot all existing containers and replace only 8790 gateway. Keep rollback image/config.
