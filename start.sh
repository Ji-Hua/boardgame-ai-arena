#!/bin/sh

# Start backend on port 8764
cd /app
PYTHONPATH="/app/engine:/app/backend-server" \
  uvicorn backend.main:app --host 0.0.0.0 --port 8764 &
BACKEND_PID=$!

# Start frontend preview server on port 8765
cd /app/frontend
npx vite preview --host 0.0.0.0 --port 8765 &
FRONTEND_PID=$!

echo "Backend running on port 8764 (PID $BACKEND_PID)"
echo "Frontend running on port 8765 (PID $FRONTEND_PID)"

# Monitor both processes. Exit container if either dies.
while true; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend process exited. Shutting down..."
    kill "$FRONTEND_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Frontend process exited. Shutting down..."
    kill "$BACKEND_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done
