# Monorepo and distributed deployment
Status: Accepted design, 2026-10-25. Implementation status is recorded in docs/MIGRATION_STATUS.md.

## Authority
This design supersedes conflicting earlier deployment and repository assumptions. Keep historical acceptance records as historical evidence. Engine rule authority and canonical AgentSpec semantics remain unchanged. New implementation begins with a documented contract, acceptance conditions and migration impact.

## Boundaries
- Engine owns legality and transitions, never sessions or networking.
- Application owns authoritative live game orchestration and revision checks.
- Agent System owns definition, runtime, training, evaluation, analysis and promotion.
- Research Arena runs local instances against Engine; service-chain acceptance separately tests Gameplay and Runtime.
- Decision Lab is independently runnable. Analysis cannot mutate live state.
- Q values, policy logits, state values and search statistics retain distinct semantics.

## Source and deployment
One GitHub repository, Ji-Hua/boardgame-ai-arena. main is the reviewed integration branch. Host names are not long-lived source branches. Training is the migration reference host.
Development clones and temporary worktrees are editable. Stable services run images pinned by digest, never bind-mounted development source. A release records source commit, component image digests and contract versions. Each host records its actual release independently; coordinated upgrades must validate compatibility and preserve rollback.
Dev runs Web/Gateway and Gameplay; Training runs Runtime, research evidence, Lab, MLflow and bounded jobs. Current physical source paths remain during incremental migration. No directory move is required merely to satisfy a target diagram.

## Target source map
engine/ remains the rule crate.
agent_system/{definition,runtime,training,evaluation,analysis,deployment}/ owns agent lifecycle.
apps/{web,gameplay,research_api,jobs}/ is the eventual application-entry layout.
packages/{contracts,board_ui}/ hosts shared wire contracts and rendering.
deploy/ owns deployment definitions; experiments/ holds reusable configurations only.
documents/ owns normative designs and decisions; docs/ owns operational facts.
Do not create empty placeholder implementations or move modules together with behavioral changes.

## Identities and data
AgentSpec hash includes algorithm, policy, parameters and model identity. Checkpoint SHA identifies bytes, not complete agent behavior.
Run and Replay references are source-namespaced. Evaluation records protocol, opening set, seeds and participants. Missing historical provenance stays unknown.
Training data remains under /home/jihua/quoridor-data. Source archives and machine settings remain external. MLflow owns its database. Research services read evidence; they do not write the MLflow SQLite file directly.
Registration, research eligibility and production promotion are separate operations.

## Delivery gates
Source checks -> immutable images -> release manifest -> isolated acceptance -> per-host deployment -> observed versions -> rollback retention.
No live switch or old-directory deletion is part of the initial migration.
Pause/resume, durable games, unified Jobs and complete UI integration remain separate contracts and milestones.
