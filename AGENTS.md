# Quoridor AI Agent Guide

This is the shared entry point for Copilot, Codex, human developers, and other
automation working in this repository.

## Read first

1. Read [README.md](README.md) for module navigation.
2. Read [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for the current
   integrated baseline and open risks.
3. Read [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) before using Dev,
   Training, MLflow, model storage, or services.
4. Follow [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) for
   branches, validation, synchronization, experiment provenance, and handoff.
5. Read the relevant authoritative design or governance document in
   [documents/](documents/) before changing behavior or boundaries.

## Authority and scope

- `documents/system/design/` defines target architecture and module
  responsibilities; `documents/system/governance/` defines contribution and
  test rules.
- `docs/` records shared Dev/Training operational facts. Update it when those
  facts change.
- `.copilot/` is ignored local working material, not the shared project record.
- The Rust engine is the rule authority. Do not recreate legality, path, or
  state-transition logic in Python.

## Non-negotiable safeguards

- Never run `git reset --hard`, `git clean`, a force push, or overwrite another
  worktree's uncommitted work.
- Never add model weights, replay data, experiment outputs, MLflow databases,
  or generated run logs to Git. Use `/home/jihua/quoridor-data/` on Training.
- Preserve output semantics: DQN outputs are Q-values; policy-distillation
  outputs are policy logits. Do not label one as the other.
- Do not change a production roster, champion alias, or deployed service while
  doing research unless the task explicitly authorizes it.
- Start from a feature/integration branch, make focused commits, run the
  smallest relevant test suite, and record the result in the handoff.

## End-of-task handoff

Before handing work to another person or agent:

1. Leave source changes committed on a named branch or explicitly state why
   they are intentionally uncommitted.
2. Record the branch, commit, tests run, and any remaining artifact paths.
3. Update the relevant `docs/` status or research record when shared facts have
   changed.
4. Do not report unverified training, benchmark, or MLflow results as facts.
