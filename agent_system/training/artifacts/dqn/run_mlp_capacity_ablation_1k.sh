#!/usr/bin/env bash
# Phase 16A: MLP Capacity Ablation — 1000-episode runs
#
# Run three architectures sequentially:
#   A: [256, 256]      — baseline (matches pre-16A default)
#   B: [512, 512]      — 2x width
#   C: [512, 512, 256] — 2x width + funnel third layer
#
# All runs use distance_delta reward (w=0.04), eval_games=100, device=auto.
#
# Usage:
#   cd /home/jihua/workspace/boardgame-ai-arena
#   bash agent_system/training/artifacts/dqn/run_mlp_capacity_ablation_1k.sh 2>&1 | tee ablation_1k.log

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== Phase 16A MLP Capacity Ablation 1k ==="
echo "Start: $(date)"

# ---------------------------------------------------------------------------
# A: [256, 256] — baseline
# ---------------------------------------------------------------------------
echo ""
echo "--- Run A: hidden_layers=[256, 256] ---"
uv run python scripts/train_dqn.py \
  --episodes 1000 \
  --buffer-capacity 50000 \
  --warmup-size 1000 \
  --batch-size 128 \
  --hidden-layers 256,256 \
  --reward-mode distance_delta \
  --distance-reward-weight 0.04 \
  --eval-games 100 \
  --eval-interval 250 \
  --device auto \
  --epsilon-decay-steps 50000 \
  --checkpoint-dir agent_system/training/artifacts/dqn/dqn_mlp256x2_1k_001/checkpoints \
  --checkpoint-interval 250 \
  --agent-id dqn_mlp256x2_1k_001 \
  --seed 42

# ---------------------------------------------------------------------------
# B: [512, 512] — 2x width
# ---------------------------------------------------------------------------
echo ""
echo "--- Run B: hidden_layers=[512, 512] ---"
uv run python scripts/train_dqn.py \
  --episodes 1000 \
  --buffer-capacity 50000 \
  --warmup-size 1000 \
  --batch-size 128 \
  --hidden-layers 512,512 \
  --reward-mode distance_delta \
  --distance-reward-weight 0.04 \
  --eval-games 100 \
  --eval-interval 250 \
  --device auto \
  --epsilon-decay-steps 50000 \
  --checkpoint-dir agent_system/training/artifacts/dqn/dqn_mlp512x2_1k_001/checkpoints \
  --checkpoint-interval 250 \
  --agent-id dqn_mlp512x2_1k_001 \
  --seed 42

# ---------------------------------------------------------------------------
# C: [512, 512, 256] — 2x width + funnel
# ---------------------------------------------------------------------------
echo ""
echo "--- Run C: hidden_layers=[512, 512, 256] ---"
uv run python scripts/train_dqn.py \
  --episodes 1000 \
  --buffer-capacity 50000 \
  --warmup-size 1000 \
  --batch-size 128 \
  --hidden-layers 512,512,256 \
  --reward-mode distance_delta \
  --distance-reward-weight 0.04 \
  --eval-games 100 \
  --eval-interval 250 \
  --device auto \
  --epsilon-decay-steps 50000 \
  --checkpoint-dir agent_system/training/artifacts/dqn/dqn_mlp512x512x256_1k_001/checkpoints \
  --checkpoint-interval 250 \
  --agent-id dqn_mlp512x512x256_1k_001 \
  --seed 42

echo ""
echo "=== Ablation complete: $(date) ==="
