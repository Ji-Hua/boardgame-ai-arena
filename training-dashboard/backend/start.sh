#!/usr/bin/env bash
# Start the Training Dashboard backend server on port 8740.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

PORT="${TRAINING_DASHBOARD_BACKEND_PORT:-8740}"

echo "Starting Training Dashboard backend on port $PORT"
echo "  Repo root: $REPO_ROOT"
echo "  Read-only mode: yes"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

DASHBOARD_DIR="$SCRIPT_DIR/.."

exec uvicorn backend.app:app \
  --app-dir "$DASHBOARD_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload
