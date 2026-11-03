# OpenSpiel research jobs
Status: implementing
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
