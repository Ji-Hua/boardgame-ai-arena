#!/usr/bin/env bash
# Foreground supervisor. The caller may detach this whole process with its own
# log redirection; no production service, catalog, alias, or port is changed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_ROOT="/home/jihua/workspace/boardgame-ai-arena"
PYTHON="${QUORIDOR_PYTHON:-$ORIGINAL_ROOT/.venv/bin/python}"
export QUORIDOR_PYTHON="$PYTHON"
HOST="${PLATFORM_HOST:-0.0.0.0}"
PORT="${PLATFORM_PORT:-8751}"
LAB_PORT="${PLATFORM_LAB_PORT:-8754}"
export PLATFORM_FRONTEND_DIST="${PLATFORM_FRONTEND_DIST:-$ROOT/training-dashboard/frontend/dist}"

if [[ ! -x "$PYTHON" || "$PYTHON" != /* ]]; then
  printf 'QUORIDOR_PYTHON must name an existing absolute Python executable: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$PLATFORM_FRONTEND_DIST/index.html" ]]; then
  printf 'Built frontend missing: %s/index.html\nRun npm run build in %s/training-dashboard/frontend; install dependencies explicitly if needed. No npm install/build is run by this launcher.\n' "$PLATFORM_FRONTEND_DIST" "$ROOT" >&2
  exit 1
fi
export QUORIDOR_DATA_ROOT="${QUORIDOR_DATA_ROOT:-/home/jihua/quoridor-data}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:8742}"
export TRAINING_DASHBOARD_SOURCE_ROOT="${TRAINING_DASHBOARD_SOURCE_ROOT:-$ORIGINAL_ROOT}"
ARTIFACTS="${TRAINING_ARTIFACTS_DIR:-$ORIGINAL_ROOT/agent_system/training/artifacts}"
export TRAINING_DASHBOARD_ARTIFACT_ROOTS="${TRAINING_DASHBOARD_ARTIFACT_ROOTS:-$ARTIFACTS/dqn:$ARTIFACTS}"

# Explicit original sources keep the platform index independent of the child-only
# merged Lab catalog. Existing operator-supplied approved manifests are retained.
if [[ -z "${PLATFORM_MODEL_CATALOGS:-}" ]]; then
  export PLATFORM_MODEL_CATALOGS
  PLATFORM_MODEL_CATALOGS="$("$PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['QUORIDOR_DATA_ROOT'])
print(json.dumps([
    {'name': 'baseline', 'path': os.environ.get('PLATFORM_MODEL_BASELINE_CATALOG', str(root / 'catalog.json'))},
    {'name': 'lab', 'path': os.environ.get('LAB_MODEL_CATALOG', str(root / 'catalog-cnn-alphazero-lite-20261021.json'))},
    {'name': 'architecture', 'path': os.environ.get('PLATFORM_MODEL_ARCHITECTURE_CATALOG', str(root / 'experiments/cnn-architecture-20261021-213018/architecture-catalog.json'))},
]))
PY
)"
fi

# Fail before starting either child if a port is occupied/protected or invalid.
"$PYTHON" - "$HOST" "$PORT" "$LAB_PORT" <<'PY'
import socket, sys
host, platform_port, lab_port = sys.argv[1:]
ports = [int(platform_port), int(lab_port)]
protected = {8740, 8741, 8742, 8743, 8744, 8764, 8765}
if len(set(ports)) != 2 or any(p < 1024 or p > 65535 or p in protected for p in ports):
    raise SystemExit('Use distinct unprivileged isolated ports, never production ports.')
sockets = []
try:
    for address, port in [(host, ports[0]), ('127.0.0.1', ports[1])]:
        sock = socket.socket()
        sockets.append(sock)
        sock.bind((address, port))
finally:
    for sock in sockets:
        sock.close()
PY

CATALOG="$("$PYTHON" "$ROOT/training-dashboard/platform_entry.py" \
  --generate-lab-catalog "$QUORIDOR_DATA_ROOT/platform-p1-p2/catalogs")"
export PLATFORM_LAB_UPSTREAM="http://127.0.0.1:$LAB_PORT"
printf 'Platform: http://%s:%s; isolated Lab: %s\nResearch catalog: %s\n' "$HOST" "$PORT" "$PLATFORM_LAB_UPSTREAM" "$CATALOG"

LAB_PID=""
PLATFORM_PID=""
cleanup() {
  trap - EXIT INT TERM HUP
  [[ -z "$PLATFORM_PID" ]] || kill "$PLATFORM_PID" 2>/dev/null || true
  [[ -z "$LAB_PID" ]] || kill "$LAB_PID" 2>/dev/null || true
  [[ -z "$PLATFORM_PID" ]] || wait "$PLATFORM_PID" 2>/dev/null || true
  [[ -z "$LAB_PID" ]] || wait "$LAB_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Assignment is scoped to this child only, never the integrated dashboard API.
LAB_MODEL_CATALOG="$CATALOG" "$PYTHON" -m uvicorn agent_system.analysis.app:app \
  --app-dir "$ROOT" --host 127.0.0.1 --port "$LAB_PORT" &
LAB_PID=$!
"$PYTHON" -m uvicorn platform_entry:create_app --factory \
  --app-dir "$ROOT/training-dashboard" --host "$HOST" --port "$PORT" &
PLATFORM_PID=$!

# Supervise both children; a failed/exited Lab must not leave an orphan gateway.
if wait -n "$LAB_PID" "$PLATFORM_PID"; then
  exit 1
else
  exit_code=$?
  exit "$exit_code"
fi