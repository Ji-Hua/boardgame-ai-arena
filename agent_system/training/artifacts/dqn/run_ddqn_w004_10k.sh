#!/usr/bin/env bash
# Phase 17B: Double DQN 10k run — distance_delta, weight=0.04, [256,256]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1

RUN_DIR="agent_system/training/artifacts/dqn/dqn_double_dqn_distance_delta_w004_10k_001"
mkdir -p "$RUN_DIR/checkpoints"

uv run python scripts/train_dqn.py \
  --episodes 10000 \
  --buffer-capacity 100000 \
  --warmup-size 5000 \
  --batch-size 128 \
  --lr 1e-4 \
  --gamma 0.95 \
  --grad-clip-norm 10.0 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 200000 \
  --target-sync-interval 1000 \
  --checkpoint-interval 500 \
  --eval-interval 500 \
  --eval-games 200 \
  --seed 42 \
  --log-interval 100 \
  --device auto \
  --opponent random_legal \
  --reward-mode distance_delta \
  --distance-reward-weight 0.04 \
  --distance-delta-clip 1.0 \
  --algorithm double_dqn \
  --hidden-layers 256,256 \
  --agent-id dqn_double_dqn_distance_delta_w004_10k_001 \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  2>&1 | tee "$RUN_DIR/train.log"
