#!/usr/bin/env bash
# Start the Training Dashboard frontend dev server on port 8741.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${TRAINING_DASHBOARD_FRONTEND_PORT:-8741}"

echo "Starting Training Dashboard frontend on port $PORT"
echo "  Backend expected at: http://localhost:8740"

if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

exec npx vite --port "$PORT"
