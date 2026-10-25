# Acting-Agent diagnostics and visual refinement
Status: implementing
Branch: feature/agent-ladder-preview
Scope: 8790 gateway UI only. Keep existing platform, Lab, MLflow and production services running.

## Contract
Default analysis follows the actual side-to-move Agent, including checkpoint SHA / heuristic depth. Other Agents require explicit comparison mode; comparison persists across plies while follow mode resolves the new actor. Position, actor/spec, mode or comparison changes invalidate pending results. No automatic expensive analysis during playback.
Recorded actions/times are historical evidence. Existing Lab returns current deterministic Top1 scoring; do not claim historical RNG reproduction or silently apply Top1 as an original stochastic policy. Expose original policy, recomputed choice, actual action, candidate scores/bounds, legal mask and available heuristic/network evidence. Hierarchical route eligibility is distinct from legality.
Tensor cells use three significant figures / scientific notation, full-value tooltips and toggleable labels; nonnegative slices use sequential color, signed slices a zero-centered diverging range, with visible slice-local legend.

## Acceptance
React tests for actor following, explicit comparison, cancellation, actual-action visibility, stochastic/mismatch/bound semantics and tensor labels. Real preview browser acceptance on neural and heuristic plies, comparison mode, heatmap controls and mobile/dark layouts. Preserve gateway rollback config and verify other container identities unchanged.
