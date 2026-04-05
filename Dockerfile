# ---- Build Stage: Rust Engine ----
FROM python:3.12-slim AS engine-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

WORKDIR /build/engine
COPY engine/ .
RUN maturin build --release --out /build/wheels

# ---- Build Stage: Frontend ----
FROM node:22-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ---- Runtime ----
FROM python:3.12-slim

# Install Node.js for vite preview
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install engine wheel
COPY --from=engine-builder /build/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Install backend Python deps
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic

# Copy backend source
COPY backend-server/ /app/backend-server/
COPY engine/game_manager/ /app/engine/game_manager/

# Copy built frontend + install prod deps for vite preview
COPY --from=frontend-builder /build/frontend/dist/ /app/frontend/dist/
COPY frontend/package.json /app/frontend/package.json
COPY frontend/vite.config.ts /app/frontend/vite.config.ts
RUN cd /app/frontend && npm install vite @vitejs/plugin-react

# Copy public assets into dist (replay files, etc.)
COPY frontend/public/ /app/frontend/dist/

# Copy start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8764 8765

CMD ["/app/start.sh"]
