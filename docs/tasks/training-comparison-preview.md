# Compact training comparison and checkpoint navigation
Status: verified
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

## Observed results
- TypeScript and production build passed. React: 40 tests passed, including seven new training/SHA cases.
- Existing five real-browser Agent/replay/play flows passed on the training UI build. Two new training browser flows passed on the final build: actual MLflow Runs 40f0edbd60314c7ab822b6dfbd8522ef and 9c1b3ed4a5004b5c933bb4bf77b594fe; shared-axis overlays and nested config diffs; exact checkpoint -> Arena matchups -> replay -> exact CNN Play; mobile selection persistence and layout.
- Browser assertions verify full replay trajectory length, participant SHA and manifest identity in displayed evidence, width >95% and height >80% of the inspector body, JSON download, and mobile document width.
- Real sparse matrices contain null diagonal cells. Added null handling and a regression test after the first browser run exposed this case. Checkbox navigation is verified after the router commits state, rather than assuming a synchronous checked property update.
- These MLflow-registered CNN Runs have 60 curve points supplied by existing local-training-file fallback; visible source labels preserve that distinction. No metric history was invented or written back to MLflow.

## Deployment and preservation
- Runtime UI commit: 2612c09, feature/agent-ladder-preview; not pushed.
- Gateway image: sha256:e9978fc1ccdad87d792fb2ca83db8724a15dec865f3b52bc410a6a5721dacab8.
- Only preview gateway on Training 8790 replaced. Preview platform/Lab and seven non-preview Training containers retained IDs, images and start times after each update. Dev 8765, Training 8741 and MLflow 8742 returned HTTP 200. No Dev writes.
- Pre-task rollback config: /home/jihua/quoridor-releases/agent-ladder-preview/ui-refinement-20261025T214923Z. Restore only its GATEWAY_IMAGE into current release.env and run manage.sh up -d --no-deps gateway; platform/Lab remain running.
- Final new-flow evidence: /home/jihua/quoridor-backups/training-comparison-preview-accepted; previous-flow acceptance and initial failure traces: /home/jihua/quoridor-backups/training-comparison-preview-evidence. Screenshots use a Linux browser without CJK fonts; DOM labels and layout were checked.

## Boundaries
Checkpoint navigation discovers exact SHA matches in completed Arena manifests and preserves each AgentSpec/protocol separately. Missing SHA or no eligible manifest Agent means no inferred Play target. Legacy training records and their complete provider evidence remain browsable; no new historical replay format or agent registration workflow was added. Full evidence means all fields returned by existing APIs, not historical diagnostics that were never recorded.
