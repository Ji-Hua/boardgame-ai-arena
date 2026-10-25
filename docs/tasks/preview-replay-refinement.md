# Preview replay and analysis refinement
Status: implementing
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
