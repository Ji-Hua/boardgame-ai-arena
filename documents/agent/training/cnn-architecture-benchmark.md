# CNN Architecture Benchmark

This protocol compares only the flat policy CNN body. It keeps the existing
`[7, 9, 9]` observation, 209 policy logits, legal-action mask, D2 teacher,
reflection augmentation, legal-action cross entropy, optimizer, batch size,
and 12,000-update budget. Every run is freshly initialized; no checkpoint is
used as a warm start.

| ID | Body | Conv widths | Dense width | Parameters |
|---|---|---:|---:|---:|
| A | Existing three-layer CNN | 32/64/64 | 256 | 1,438,545 |
| B | Three two-convolution blocks, no skip | 64/64/64 | 230 | 1,429,565 |
| C | B with residual skips | 64/64/64 | 230 | 1,430,077 |
| D | Wider existing CNN | 56/112/112 | 512 | 4,925,745 |

B and C differ only in their residual projection and skip additions. A/B/C
are within 0.63% parameters. D changes only the widths of A's original body.

## Training-side execution

Use an isolated source worktree at a named commit and a new external directory:

```bash
RUN=/home/jihua/quoridor-data/experiments/cnn-architecture-YYYYMMDD-HHMMSS
PYTHONPATH=.:backend-server .venv/bin/python scripts/collect_cnn_architecture_holdout.py \
  --output "$RUN/heldout" \
  --training-state-keys /home/jihua/quoridor-data/experiments/cnn-distillation-20261020/shared-data/training-state-keys.json
PYTHONPATH=.:backend-server .venv/bin/python scripts/run_cnn_architecture_benchmark.py \
  --output "$RUN" \
  --dataset /home/jihua/quoridor-data/experiments/cnn-distillation-20261020/shared-data/shared.pt \
  --holdout "$RUN/heldout/heldout.pt" \
  --book-manifest /home/jihua/quoridor-data/arena/matrix-20261020-v2/manifest.json \
  --seeds 42 43 44
```

The holdout generator starts a separate source-game family and excludes every
training symmetry key, including both reflections. Its output stores the D2
first-max label for strict agreement and all D2 equal-best labels for the
equivalent-optimal rate. It is not a split of training positions.

The runner records the source commit plus hashes for the shared data, holdout,
and development book in its parent manifest. It delegates each training run to
the existing training runner, preserves the existing checkpoint diagnostics,
and writes learning curves, initialization hashes, parameter counts, and D2
holdout metrics to each external run directory.

After the matrix completes, register immutable checkpoint copies in a new
catalog, log the training and Arena results to MLflow, then run the existing
Arena matrix on a fresh locked book against Random, Greedy, Minimax D1/D2, and
the frozen D2 CNN. Do not move a champion alias or modify the online roster.
