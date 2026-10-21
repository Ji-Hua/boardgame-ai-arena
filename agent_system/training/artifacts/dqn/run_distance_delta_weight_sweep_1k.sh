#!/usr/bin/env bash
# Phase 15C: Distance-delta reward weight sweep — 5 weights × 1000 episodes
# Runs sequentially. Each run ~29 min on RTX 3080 Ti; total ~2.5 h.
set -e

cd /home/jihua/workspace/boardgame-ai-arena
source .venv/bin/activate

run_one() {
    local RUN_ID="$1"
    local WEIGHT="$2"
    echo ""
    echo "========================================================"
    echo "STARTING  ${RUN_ID}  weight=${WEIGHT}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u scripts/train_dqn.py \
      --episodes 1000 --buffer-capacity 100000 --warmup-size 5000 --batch-size 128 \
      --lr 1e-4 --gamma 0.95 --grad-clip-norm 10.0 \
      --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-steps 100000 \
      --target-sync-interval 1000 \
      --checkpoint-dir "agent_system/training/artifacts/dqn/${RUN_ID}/checkpoints" \
      --checkpoint-interval 500 --agent-id "${RUN_ID}" \
      --eval-games 100 --eval-interval 500 --seed 42 --log-interval 100 \
      --device auto --opponent random_legal \
      --reward-mode distance_delta \
      --distance-reward-weight "${WEIGHT}" \
      --distance-delta-clip 1.0 \
      2>&1 | tee "agent_system/training/artifacts/dqn/${RUN_ID}/train.log"

    echo ""
    echo "========================================================"
    echo "FINISHED  ${RUN_ID}  weight=${WEIGHT}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"
}

run_one dqn_distance_delta_w002_1k_001 0.02
run_one dqn_distance_delta_w004_1k_001 0.04
run_one dqn_distance_delta_w006_1k_001 0.06
run_one dqn_distance_delta_w008_1k_001 0.08
run_one dqn_distance_delta_w010_1k_001 0.10

echo ""
echo "ALL 5 SWEEP RUNS COMPLETE"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
