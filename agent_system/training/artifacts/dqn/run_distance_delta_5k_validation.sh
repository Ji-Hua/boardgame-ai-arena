#!/usr/bin/env bash
# Phase 15D: Distance-delta reward weight validation — 5 000 episodes
# Runs A (w=0.04) and B (w=0.05) sequentially.
# Optional run C (w=0.06): set RUN_C=1 to enable.
set -e

RUN_C=0   # set to 1 to also run optional w006

cd /home/jihua/workspace/boardgame-ai-arena
source .venv/bin/activate

run_one() {
    local RUN_ID="$1"
    local WEIGHT="$2"
    local CMD

    echo ""
    echo "========================================================"
    echo "STARTING  ${RUN_ID}  weight=${WEIGHT}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    CMD="PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u scripts/train_dqn.py \
  --episodes 5000 \
  --buffer-capacity 200000 \
  --warmup-size 5000 \
  --batch-size 128 \
  --lr 1e-4 \
  --gamma 0.95 \
  --grad-clip-norm 10.0 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 600000 \
  --target-sync-interval 1000 \
  --checkpoint-dir agent_system/training/artifacts/dqn/${RUN_ID}/checkpoints \
  --checkpoint-interval 500 \
  --agent-id ${RUN_ID} \
  --eval-games 200 \
  --eval-interval 500 \
  --seed 42 \
  --log-interval 100 \
  --device auto \
  --opponent random_legal \
  --reward-mode distance_delta \
  --distance-reward-weight ${WEIGHT} \
  --distance-delta-clip 1.0"

    echo ""
    echo "COMMAND: ${CMD}"
    echo ""

    eval "${CMD}" 2>&1 | tee "agent_system/training/artifacts/dqn/${RUN_ID}/train.log"

    echo ""
    echo "========================================================"
    echo "FINISHED  ${RUN_ID}  weight=${WEIGHT}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"
}

# Run A: weight=0.04
run_one dqn_distance_delta_w004_5k_001 0.04

# Run B: weight=0.05
run_one dqn_distance_delta_w005_5k_001 0.05

# Optional Run C: weight=0.06
if [ "${RUN_C}" -eq 1 ]; then
    run_one dqn_distance_delta_w006_5k_001 0.06
else
    echo ""
    echo "Optional Run C (w006) skipped (RUN_C=0)."
fi

echo ""
echo "ALL PHASE 15D VALIDATION RUNS COMPLETE"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
