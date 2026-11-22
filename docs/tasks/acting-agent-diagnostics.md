# Acting-Agent diagnostics and visual refinement
Status: verified
Branch: feature/agent-ladder-preview
Scope: 8790 gateway UI only. Keep existing platform, Lab, MLflow and production services running.

## Contract
Default analysis follows the actual side-to-move Agent, including checkpoint SHA / heuristic depth. Other Agents require explicit comparison mode; comparison persists across plies while follow mode resolves the new actor. Position, actor/spec, mode or comparison changes invalidate pending results. No automatic expensive analysis during playback.
Recorded actions/times are historical evidence. Existing Lab returns current deterministic Top1 scoring; do not claim historical RNG reproduction or silently apply Top1 as an original stochastic policy. Expose original policy, recomputed choice, actual action, candidate scores/bounds, legal mask and available heuristic/network evidence. Hierarchical route eligibility is distinct from legality.
Tensor cells use three significant figures / scientific notation, full-value tooltips and toggleable labels; nonnegative slices use sequential color, signed slices a zero-centered diverging range, with visible slice-local legend.

## Acceptance
React tests for actor following, explicit comparison, cancellation, actual-action visibility, stochastic/mismatch/bound semantics and tensor labels. Real preview browser acceptance on neural and heuristic plies, comparison mode, heatmap controls and mobile/dark layouts. Preserve gateway rollback config and verify other container identities unchanged.

## Observed acceptance
- TypeScript and production build passed; 45 React tests passed (five new acting-Agent/tensor tests plus existing cancellation, training and replay checks).
- Seven existing real browser flows passed on the updated preview. The new acting-Agent browser flow exposed a network-tab persistence bug after moving to a heuristic ply; fixed by returning to decision evidence when subject/position changes, then the complete new flow passed.
- Real test starts at A-baseline_seed-44 ply 14, verifies exact checkpoint payload and historical-action row, neural activation short labels/full titles, number toggle, dark theme, ply 15 Greedy following and path-delta basis, explicit neural comparison and return to actor mode. Mobile width and JavaScript errors checked.
- Raw scores retain Q/logit/heuristic meanings; minimax bound rows are labelled and no exact difference is computed from bounds. Hierarchical ineligible actions do not produce a misleading Top1 gap. Missing mask evidence is not labelled illegal.

## Release
- UI source: 27f7f34, feature/agent-ladder-preview; not pushed.
- Gateway image: sha256:0e184a6182b6c0aeeb5f841082bcc49b4997ef8cb13b7b693e7b6e0a94d58e99.
- Only Training 8790 gateway replaced. Preview platform/Lab and seven non-preview Training containers kept IDs, images, start times. Existing Training 8741, MLflow 8742, Dev 8765 returned HTTP 200; no Dev writes.
- Pre-task gateway rollback configuration: /home/jihua/quoridor-releases/agent-ladder-preview/ui-refinement-20261025T221259Z. Restore its GATEWAY_IMAGE only and run manage.sh up -d --no-deps gateway. Keep platform/Lab running.
- Evidence: /home/jihua/quoridor-backups/acting-agent-preview-evidence (existing flows, screenshots and initial failure); /home/jihua/quoridor-backups/acting-agent-preview-final (new flow after fix).
- Linux screenshots lack CJK fonts; underlying Chinese DOM labels, actual interactions, compact number rendering and theme colors verified.

## Limits
No historical scores, random states or search traces were backfilled. No policy is sampled by the diagnostic UI. Unsupported/human actors do not silently select another model; explicit comparison remains available. Analysis is user-triggered and does not run per animation frame.
