# Quoridor

A Quoridor board game system with a Rust game engine, Python backend server, and React frontend.

## Architecture

- **Engine** (`engine/`) — Rust rule engine with Python bindings (via PyO3/maturin)
- **Backend** (`backend-server/`) — FastAPI server providing REST + WebSocket APIs
- **Frontend** (`frontend/`) — React/TypeScript UI with live play and game replay

## Ports

| Service  | Port |
|----------|------|
| Frontend | 8765 |
| Backend  | 8764 |

---

## Local Development (without Docker)

### Backend

```bash
# Install Python dependencies
pip install fastapi uvicorn pydantic

# Install the engine wheel (requires Rust toolchain + maturin)
cd engine && maturin develop --release && cd ..

# Start backend on port 8764
PYTHONPATH="engine:backend-server" uvicorn backend.main:app --host 0.0.0.0 --port 8764
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server on port 8765
npm run dev
```

Open http://localhost:8765 in your browser.

---

## Docker (single container)

Build and run both services in one container:

```bash
# Build the image
docker build -t quoridor .

# Run the container
docker run -p 8764:8764 -p 8765:8765 quoridor
```

Open http://localhost:8765 in your browser.

The container:
- Builds the Rust engine wheel
- Builds the frontend (Vite production build)
- Starts backend (uvicorn, port 8764) and frontend (vite preview, port 8765) concurrently

---

## Replay

The frontend includes a replay mode that plays through a full recorded game step-by-step (0.5s per step). Click "Replay Full Game" from the main menu.

---

## License

Apache License 2.0. See LICENSE.

## Trademark Notice

Quoridor is a registered trademark of Gigamic. This is an independent, unofficial reimplementation for research and educational purposes.
