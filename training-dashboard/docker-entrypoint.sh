#!/bin/bash
# docker-entrypoint.sh — starts the Training Dashboard backend and frontend.
# Requires bash 4.3+ for `wait -n` (Debian 12 / bookworm ships bash 5.x).
set -euo pipefail

echo "[training-dashboard] ========================================"
echo "[training-dashboard]  Quoridor Training Dashboard"
echo "[training-dashboard] ========================================"
echo "[training-dashboard]  Backend  → http://localhost:8740"
echo "[training-dashboard]  Frontend → http://localhost:8741"
echo "[training-dashboard]  Health   → http://localhost:8740/health"
echo "[training-dashboard] ========================================"

# ---- Backend ----
echo "[training-dashboard] Starting backend..."
cd /app
uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port 8740 \
    --log-level info &
BACKEND_PID=$!

# ---- Frontend ----
echo "[training-dashboard] Starting frontend..."
cd /app/frontend
npm run dev &
FRONTEND_PID=$!

echo "[training-dashboard] Backend  PID: $BACKEND_PID"
echo "[training-dashboard] Frontend PID: $FRONTEND_PID"

# ---- Forward SIGTERM/SIGINT to children ----
_shutdown() {
    echo "[training-dashboard] Shutting down..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}
trap _shutdown SIGTERM SIGINT

# ---- Wait for either process to exit ----
# wait -n exits as soon as any child changes state.
wait -n
EXIT_CODE=$?
echo "[training-dashboard] A process exited with code $EXIT_CODE. Stopping container."
kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
exit $EXIT_CODE
