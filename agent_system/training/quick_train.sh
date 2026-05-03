#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="agent_system/training/artifacts/dqn/stabilization_smoke_001"

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/checkpoints"

PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u scripts/train_dqn.py \
  --config agent_system/training/configs/dqn/stabilization_smoke_001.yaml \
  2>&1 | tee "$RUN_DIR/train.log"