#!/bin/sh
# Isolated loopback stack for browser verification of this worktree on Training.
# Never touches the live 8790 preview: Platform 18793, Lab 18794, gateway 18795 on 127.0.0.1.
# Usage: scripts/verify_stack.sh start|stop  (EVIDENCE_DIR overrides the log directory)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=/home/jihua/quoridor-data
PY=${VERIFY_PYTHON:-/home/jihua/workspace/boardgame-ai-arena/.venv/bin/python}
EVIDENCE=${EVIDENCE_DIR:-/home/jihua/quoridor-backups/verify-stack}
GATEWAY_IMAGE=${VERIFY_GATEWAY_IMAGE:-sha256:7dbe0cad93934fb6fac1638c471bf8d86a7cdcdf5c611bd5fdcc7f70aef66f07}
LAB_CATALOG=${LAB_MODEL_CATALOG:-$DATA/platform-p1-p2/catalogs/research-catalog-17a0fc10c1260ac107745604a5836f580087ab3936465e520b804ce1ed338e46.json}
ARTIFACTS=/home/jihua/workspace/boardgame-ai-arena/agent_system/training/artifacts
NAME=quoridor-verify-gateway

stop() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  for pid in platform lab; do
    [ -f "$EVIDENCE/$pid.pid" ] && kill "$(cat "$EVIDENCE/$pid.pid")" 2>/dev/null || true
    rm -f "$EVIDENCE/$pid.pid"
  done
}

case "${1:-}" in
start)
  stop
  mkdir -p "$EVIDENCE"
  export PYTHONPATH="$ROOT:$ROOT/backend-server:$ROOT/engine" OMP_NUM_THREADS=1
  LAB_MODEL_CATALOG="$LAB_CATALOG" nohup "$PY" -m uvicorn agent_system.analysis.app:app \
    --app-dir "$ROOT" --host 127.0.0.1 --port 18794 >"$EVIDENCE/lab.log" 2>&1 &
  echo $! >"$EVIDENCE/lab.pid"
  PLATFORM_MODEL_CATALOGS=discover QUORIDOR_DATA_ROOT=$DATA MLFLOW_TRACKING_URI=http://127.0.0.1:8742 \
  TRAINING_DASHBOARD_ARTIFACT_ROOTS="$ARTIFACTS/dqn:$ARTIFACTS" PLATFORM_FRONTEND_DIST="$ROOT/apps/web/dist" \
  TRAINING_DASHBOARD_SOURCE_ROOT="$ROOT" PLATFORM_LAB_UPSTREAM=http://127.0.0.1:18794 \
    nohup "$PY" -m uvicorn platform_entry:create_app --factory --app-dir "$ROOT/training-dashboard" \
    --host 127.0.0.1 --port 18793 >"$EVIDENCE/platform.log" 2>&1 &
  echo $! >"$EVIDENCE/platform.pid"
  sed -e 's/listen 8080;/listen 127.0.0.1:18795;/' -e 's/${CLASSIC_UPSTREAM}/127.0.0.1:18793/g' \
      -e 's/${PLATFORM_UPSTREAM}/127.0.0.1:18793/g' "$ROOT/deploy/gateway.conf.template" >"$EVIDENCE/gateway.conf"
  docker run -d --name "$NAME" --network host -v "$EVIDENCE/gateway.conf:/etc/nginx/conf.d/default.conf:ro" \
    -v "$ROOT/apps/web/dist:/usr/share/nginx/html:ro" --entrypoint nginx "$GATEWAY_IMAGE" -g 'daemon off;' >/dev/null
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:18795/health >/dev/null && curl -sf http://127.0.0.1:18794/health >/dev/null && {
      echo "verify stack ready: http://127.0.0.1:18795 (logs: $EVIDENCE)"; exit 0; }
    sleep 1
  done
  echo "verify stack did not become healthy; see $EVIDENCE" >&2; exit 1 ;;
stop) stop; echo "verify stack stopped" ;;
*) echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
