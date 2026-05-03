# syntax=docker/dockerfile:1
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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Frontend npm install (cache layer) ----------
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci --prefer-offline

# ---------- App source ----------
COPY backend/ ./backend/
COPY backend_entry.py .
COPY frontend/ ./frontend/

# ---------- Entrypoint ----------
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8740 8741

ENTRYPOINT ["/docker-entrypoint.sh"]
