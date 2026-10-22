#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QUORIDOR_PYTHON:-/home/jihua/workspace/boardgame-ai-arena/.venv/bin/python}"
: "${QUORIDOR_RESEARCH_SERVICE_KEY:?Supply an isolated service credential}"
: "${QUORIDOR_RUNTIME_VERSION:?Supply the committed runtime version}"
export QUORIDOR_RESEARCH_SERVICE_KEY QUORIDOR_RUNTIME_VERSION
export QUORIDOR_DATA_ROOT="${QUORIDOR_DATA_ROOT:-/home/jihua/quoridor-data}"
export MLFLOW_TRACKING_URI="http://127.0.0.1:8742"
export TRAINING_DASHBOARD_SOURCE_ROOT="/home/jihua/workspace/boardgame-ai-arena"
export TRAINING_ARTIFACTS_DIR="$TRAINING_DASHBOARD_SOURCE_ROOT/agent_system/training/artifacts/dqn"
export TRAINING_DASHBOARD_ARTIFACT_ROOTS="$TRAINING_ARTIFACTS_DIR:$TRAINING_DASHBOARD_SOURCE_ROOT/agent_system/training/artifacts"
export PLATFORM_GAMEPLAY_UPSTREAM="http://192.168.68.102:8774"
export PLATFORM_LAB_UPSTREAM="http://127.0.0.1:8754"
export QUORIDOR_RESEARCH_ROOTS="$QUORIDOR_DATA_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export QUORIDOR_RESEARCH_CATALOG
QUORIDOR_RESEARCH_CATALOG="$("$PYTHON" "$ROOT/scripts/prepare_research_play.py")"
"$PYTHON" -c 'import socket; sockets=[]
for port in (8752,8753):
    listener=socket.socket(); listener.bind(("0.0.0.0",port)); sockets.append(listener)'
[[ -f "$ROOT/training-dashboard/frontend/dist/index.html" ]] || { printf 'Build frontend first\n' >&2; exit 1; }
RUNTIME_PID=""; PLATFORM_PID=""
cleanup() {
  trap - EXIT INT TERM HUP
  [[ -z "$RUNTIME_PID" ]] || kill "$RUNTIME_PID" 2>/dev/null || true
  [[ -z "$PLATFORM_PID" ]] || kill "$PLATFORM_PID" 2>/dev/null || true
  wait || true
}
trap cleanup EXIT INT TERM HUP
QUORIDOR_RESEARCH_ONLY=1 "$PYTHON" -m uvicorn agent_system.runtime.service.server:app \
  --app-dir "$ROOT" --host 0.0.0.0 --port 8753 &
RUNTIME_PID=$!
"$PYTHON" -m uvicorn platform_entry:create_app --factory \
  --app-dir "$ROOT/training-dashboard" --host 0.0.0.0 --port 8752 &
PLATFORM_PID=$!
printf 'P3-A platform: http://192.168.68.88:8752; isolated Runtime: 8753; Dev Gameplay: 8774\n'
wait -n "$RUNTIME_PID" "$PLATFORM_PID"