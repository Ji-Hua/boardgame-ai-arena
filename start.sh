#!/bin/sh
# start.sh — local development launcher (non-Docker)
# For Docker, use: docker compose up --build

# Resolve project root (directory containing this script)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Use the project venv's Python 3.12
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: .venv not found at $SCRIPT_DIR/.venv"
  echo "Run: uv venv && uv pip install fastapi 'uvicorn[standard]' pydantic httpx"
  exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Start agent service on port 8090
PYTHONPATH="$SCRIPT_DIR" \
  "$PYTHON" -m uvicorn agents.agent_service.server:app \
    --host 0.0.0.0 --port 8090 &
AGENT_PID=$!

# Start backend on port 8764
AGENT_SERVICE_URL="http://localhost:8090" \
PYTHONPATH="$SCRIPT_DIR/engine:$SCRIPT_DIR/backend-server" \
  "$PYTHON" -m uvicorn backend.main:app \
    --host 0.0.0.0 --port 8764 --log-level info &
BACKEND_PID=$!

# Start frontend preview server on port 8765
# Discover 'node' — try PATH first, then well-known locations
if command -v node >/dev/null 2>&1; then
  NODE_BIN="node"
elif [ -x "$HOME/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036/server/node" ]; then
  NODE_BIN="$HOME/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036/server/node"
else
  echo "ERROR: 'node' not found. Install Node.js or ensure it is on PATH."
  kill "$AGENT_PID" "$BACKEND_PID" 2>/dev/null || true
  exit 1
fi
VITE_BIN="$SCRIPT_DIR/frontend/node_modules/.bin/vite"
if [ ! -x "$VITE_BIN" ]; then
  echo "ERROR: vite not found at $VITE_BIN. Run: cd frontend && npm install"
  kill "$AGENT_PID" "$BACKEND_PID" 2>/dev/null || true
  exit 1
fi
cd "$SCRIPT_DIR/frontend"
"$NODE_BIN" "$VITE_BIN" preview --host 0.0.0.0 --port 8765 &
FRONTEND_PID=$!

echo "Agent service running on port 8090 (PID $AGENT_PID)"
echo "Backend running on port 8764 (PID $BACKEND_PID)"
echo "Frontend running on port 8765 (PID $FRONTEND_PID)"

# Monitor all processes; exit if any dies
while true; do
  if ! kill -0 "$AGENT_PID" 2>/dev/null; then
    echo "Agent service exited. Shutting down..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend process exited. Shutting down..."
    kill "$AGENT_PID" "$FRONTEND_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Frontend process exited. Shutting down..."
    kill "$AGENT_PID" "$BACKEND_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done
