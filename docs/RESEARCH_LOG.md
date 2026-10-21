# Research Log

This is a concise index of verified research provenance and conclusions. It is
not a replacement for experiment manifests, MLflow artifacts, or the detailed
documents under `documents/agent/training/`.

## 2026-10-21 — source integration and provenance baseline

- Preserved CNN research source on Training in commit `9737338` and pushed it
  as `codex/environment-deployment`.
- Integrated Training commits into Dev on
  `integration/dev-training-20261021` without rewriting either source branch.
- Verified the MLflow server, persistent SQLite backend, artifact store, and
  registry mechanism. Details: [ENVIRONMENTS.md](ENVIRONMENTS.md).
- No new model, checkpoint, champion alias, or training run was created.

## Historical D2 policy distillation

| Field | Verified record |
|---|---|
| Training data | `/home/jihua/quoridor-data/experiments/cnn-distillation-20261020/shared-data/shared.pt` |
| Dataset SHA-256 | `8ffb562e2fbaa602629371c69d32b606d1feea2233ece3e4f807e8d5899a576a` |
| Dataset metadata | 2,000 games; 124,493 training positions; no separate same-source validation split. |
| Frozen checkpoint | `.../02-policy-d2/checkpoints/update012000.pt` |
| Checkpoint SHA-256 | `09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4` |
| Output meaning | Policy logits under legal-action masking; not Q-values. |

The historical report documents 473 wins in 800 games against Greedy for the
frozen D2 policy CNN on its own frozen opening protocol. That is historical
context only. It is not evidence of superiority to Minimax D2 and must not be
compared directly with a new holdout or architecture benchmark.

## Architecture-comparison guardrails

The planned A/B/C comparison is **architecture-only**:

- Keep the seven-plane `[7, 9, 9]` observation, 209 logits, legal-action mask,
  D2 labels, cross-entropy, reflection augmentation, and frozen data/evaluation
  protocol constant.
- Compare fixed-update learning efficiency separately from a predeclared
  convergence budget; use multiple independent initialization seeds.
- Create a new source-game/opening-group-held-out teacher-labelled validation
  set and a separate fresh paired-opening playing-strength book.
- Report training fit, held-out agreement/equivalent-optimal actions, paired
  W/D/L against Random/Greedy/Minimax D1/Minimax D2/old CNN, reliability, and
  cost.
- Do not add a value head, attention, path features, hierarchical action heads,
  PPO, DQN, DAgger, self-play, MCTS, reward shaping, or a new teacher in this
  comparison.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) and the detailed historical
[policy-distillation document](../documents/agent/training/cnn-policy-distillation.md).
