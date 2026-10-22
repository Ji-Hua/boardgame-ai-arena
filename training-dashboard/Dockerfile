# syntax=docker/dockerfile:1
FROM python:3.12-slim AS engine-builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY engine/Cargo.toml engine/pyproject.toml /build/engine/
COPY engine/src /build/engine/src
RUN pip wheel --no-deps /build/engine --wheel-dir /wheels

FROM python:3.12-slim

# ---------- system: Node.js 20 ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python deps (cache layer) ----------
COPY training-dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=engine-builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# ---------- Frontend npm install (cache layer) ----------
COPY training-dashboard/frontend/package.json training-dashboard/frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci --prefer-offline

# ---------- App source ----------
COPY training-dashboard/backend/ ./backend/
COPY training-dashboard/backend_entry.py .
COPY training-dashboard/frontend/ ./frontend/
# Existing engine/state serializers only; no weights or experiment outputs.
COPY engine/game_manager /runtime/engine/game_manager
COPY backend-server/backend/adapters/engine_adapter.py /runtime/backend-server/backend/adapters/engine_adapter.py
COPY agent_system/evaluation/arena/agents/scorers.py agent_system/evaluation/arena/agents/core.py /runtime/agent_system/evaluation/arena/agents/
COPY agent_system/training/dqn/replay_writer.py agent_system/training/dqn/action_space.py /runtime/agent_system/training/dqn/
ENV PYTHONPATH=/runtime
ENV TRAINING_DASHBOARD_SOURCE_ROOT=/runtime

# Build static frontend assets for the deployed service.
RUN cd frontend && npm run build

# ---------- Entrypoint ----------
COPY training-dashboard/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8740 8741

ENTRYPOINT ["/docker-entrypoint.sh"]
