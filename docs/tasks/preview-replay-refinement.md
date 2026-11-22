# Preview replay and analysis refinement
Status: verified
Branch: feature/agent-ladder-preview
Design: documents/agent/analysis/decision-analysis-interface.md

## Scope
Restore replay transport, display immutable participant identities, and keep per-position analysis in a compact inspector. Continue existing read-only MLflow/provider training records. No production change, data migration, new tracking backend or game API change.

## Acceptance
- First/last, previous/next, autoplay/pause, speed, numeric seek and keyboard controls; stop at end and on branch analysis.
- Replay URL preserves the selected ply. Seeking cancels stale analysis without resetting selected Agent.
- Both seats show type/version, player control and current turn; play setup supports swapping and direct navigation.
- Board remains visible beside bounded inspector; candidate, search, network and source views are selected rather than stacked.
- Original versus recomputed evidence stays explicit; score bounds and checkpoint identity preserved.
- Real MLflow training metrics still load. Only preview gateway is replaced; platform/lab and all non-preview containers retain identity/start time.

## Validation plan
Typecheck, React tests for replay lifecycle and stale requests, production build, real browser acceptance on 8790 including mobile, training and exact Agent play. Preserve previous preview image/config for rollback.

## Observed results
- npm run typecheck: passed. npm run test:ui: 33 passed. npm run build: passed.
- Final browser run: 5 passed (real MLflow-backed training metrics, neural activation, heuristic/heuristic play, exact CNN/human play and reload, replay transport/search/seek, selected analyst retention, seat swap, desktop/mobile layout). Added 1366x768 transport visibility assertion.
- One earlier browser run used an overly narrow tensor-image selector; restored the existing spatial/vector selector and reran successfully. No product fallback was added.
- UI runtime source: 354f275. Branch remains feature/agent-ladder-preview; not pushed.
- Gateway image: sha256:64ec80efe7e239fc2226d98e93fba5f41d98b6e20e27e6e462f0174855f69d97.
- Only preview gateway on 8790 replaced. Preview platform/Lab and all seven non-preview Training containers retained IDs, images and start times. Existing Training 8741, MLflow 8742 and Dev 8765 responded 200. No Dev mutations.
- Evidence: /home/jihua/quoridor-backups/agent-ladder-refinement-final. Linux screenshots lack CJK fonts; layout, DOM labels and real interactions verified.
- Pre-task preview config: /home/jihua/quoridor-releases/agent-ladder-preview/ui-refinement-20261025T211456Z. Intermediate version also retained at ui-refinement-20261025T211931Z.

## Rollback
Restore only GATEWAY_IMAGE from the pre-task release.env into the active preview release.env, then run the existing preview manage.sh up -d --no-deps gateway. Retain PREVIEW_IMAGE and LAB_MODEL_CATALOG. Both previous gateway images remain local. Do not restart platform/Lab or touch production entries.

## Unchanged boundaries
Training continues through the existing MLflow/read-only legacy providers. No database, checkpoint, model registry or tracking backend migration. Advanced raw diagnostics remain available on demand. Training comparison expansion and new TensorBoard instrumentation are not part of this UI refinement.
