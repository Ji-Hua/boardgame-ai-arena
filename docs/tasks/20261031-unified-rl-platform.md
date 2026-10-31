# Unified RL platform: single entry for training, diagnosis and play
Status: proposed
Owner: Ji Hua (decisions), Claude (design)
Branch: feature/unified-rl-platform (from feature/agent-ladder-preview bbc6e28)
Design/contract: documents/system/design/unified-rl-platform.md
## Scope and non-goals
Scope: one entry (Training 8790 first) covering experiments, training, Jobs, ladder, models, Lab and Play; Jobs subsystem for submitting GPU training/evaluation/diagnosis; convergence to one Runtime, one Lab and one Gameplay.
Non-goals: user accounts/TLS, rule changes, rewriting historical evidence, multi-GPU scheduling, production roster or champion changes.
## Expected behavior and compatibility
The running 8790 preview and all other existing services stay unchanged until each phase passes acceptance. Old API paths remain proxied until consumers migrate.
## Acceptance criteria
Per phase, see section 11 of the design.
## Implementation
Not started. This commit only adds the design and this task record.
## Verification commands and observed results
Documentation only; no tests run.
## External artifacts and deployment impact
None.
## Deferred decisions
See section 12 of the design: entry port, Jobs write token, GPU image vs host venv, MLflow sub-path proxy, Classic feature retention, legacy artifacts migration.
