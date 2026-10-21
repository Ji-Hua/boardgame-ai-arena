# Quoridor Training Dashboard

A lightweight, read-only visualization dashboard for Quoridor RL training artifacts.

---

## What This Is

The Training Dashboard reads existing training artifact directories (logs, checkpoints, benchmark results) and presents them as a browseable web UI. It requires no database, no external services, and no configuration beyond pointing it at artifact directories.

## What This Is Not

> **The Training Dashboard is read-only.**
> It does not start training.
> It does not run evaluation.
> It does not replace Arena.
> It visualizes existing training and benchmark artifacts.

---

## Architecture

```
training-dashboard/
  backend/       FastAPI read-only API server      (port 8740)
  frontend/      Vite + React dashboard UI         (port 8741)
  sample_artifacts/  Sample run for testing
  README.md
```

---

## Running

### A. Docker Compose (recommended)

**Requirements:** Docker with Compose plugin (v2).

```bash
cd training-dashboard
docker compose up --build
```

Then open:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8741 |
| Backend health | http://localhost:8740/health |
| Backend runs API | http://localhost:8740/api/runs |
| Backend summary | http://localhost:8740/api/summary |

To run in the background:

```bash
docker compose up --build -d
docker compose logs -f     # tail logs
docker compose down        # stop and remove
```

#### Volume mapping

The compose file mounts the training artifact directory read-only into the container:

| Host path (relative to repo root) | Container path |
|-----------------------------------|----------------|
| `agent_system/training/artifacts` | `/data/training_artifacts` |

`TRAINING_DASHBOARD_ARTIFACT_ROOTS` is set to:
```
/data/training_artifacts/dqn:/data/training_artifacts
```

Both paths are scanned; only those that exist are used.

**Artifacts live somewhere else?** Override the volume in `training-dashboard/docker-compose.yml`:

```yaml
volumes:
  - /absolute/path/to/your/runs:/data/training_artifacts:ro
environment:
  TRAINING_DASHBOARD_ARTIFACT_ROOTS: "/data/training_artifacts"
```

---

### B. No-Docker (local dev)

**Requirements:** Python 3.12+, Node.js 18+, npm.

**Backend:**

```bash
cd <repo-root>
source .venv/bin/activate
cd training-dashboard
uvicorn backend.app:app --host 0.0.0.0 --port 8740 --reload
```

**Frontend** (separate terminal):

```bash
cd training-dashboard/frontend
npm install
npm run dev          # starts on http://localhost:8741
```

Set artifact roots if needed:

```bash
export TRAINING_DASHBOARD_ARTIFACT_ROOTS="/path/to/runs:/another/path"
```

---

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_DASHBOARD_ARTIFACT_ROOTS` | auto-detected from repo | Colon-separated list of directories to scan for training runs |
| `TRAINING_DASHBOARD_BACKEND_PORT` | `8740` | Backend listen port (non-Docker only) |

---

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Dashboard shows no runs | Check `TRAINING_DASHBOARD_ARTIFACT_ROOTS` points to a directory containing run subdirectories. Verify with `curl http://localhost:8740/api/runs`. |
| Port already in use | Kill existing process: `kill $(lsof -ti:8740)` or `kill $(lsof -ti:8741)`. |
| Frontend shows "Failed to fetch" | Backend is not running or proxy is broken. Check `http://localhost:8740/health`. |
| Docker container starts but dashboard is empty | Check that the host volume path exists and is readable. Run `docker compose exec training-dashboard ls /data/training_artifacts` to verify. |
| Mounted directory path is wrong | Edit the `volumes:` block in `training-dashboard/docker-compose.yml`. Paths must be absolute or relative to the compose file's directory. |
| Permission error reading artifacts | The container runs as root by default. If host files are owned by another user with restrictive permissions, make them world-readable: `chmod -R o+rX /path/to/artifacts`. |

---

## Running the Backend

**Requirements:** Python 3.10+, FastAPI, uvicorn (already in project venv).

```bash
cd <repo-root>
source .venv/bin/activate

# Option A: module invocation
python -m uvicorn training-dashboard.backend.app:app --host 0.0.0.0 --port 8740 --reload

# Option B: start script
bash training-dashboard/backend/start.sh
```

The backend will automatically scan known artifact directories on each request.

### Health check

```
GET http://localhost:8740/health
```

---

## Running the Frontend

**Requirements:** Node.js 18+, npm.

```bash
cd training-dashboard/frontend
npm install
npm run dev          # starts on port 8741
```

Or use the start script:

```bash
bash training-dashboard/frontend/start.sh
```

Open `http://localhost:8741` in a browser.

The Vite dev server proxies `/api/*` and `/health` to `localhost:8740`.

---

## Default Ports

| Service  | Port |
|----------|------|
| Backend  | 8740 |
| Frontend | 8741 |

Both ports are in the reserved 8740–8749 range for the Training Dashboard.
They do not conflict with the existing backend (8764) or frontend (8765).

---

## Configuring Artifact Roots

Set `TRAINING_DASHBOARD_ARTIFACT_ROOTS` to a colon-separated list of directories:

```bash
export TRAINING_DASHBOARD_ARTIFACT_ROOTS="/path/to/runs:/another/path"
python -m uvicorn training-dashboard.backend.app:app --port 8740
```

**Default roots** (searched relative to repo root, in order):

- `agent_system/training/artifacts/dqn`
- `agent_system/training/artifacts`
- `artifacts/training_runs`
- `training_runs`
- `runs`
- `outputs`
- `agents/checkpoints`
- `arena/results`

Only roots that actually exist on disk are scanned.

---

## Using the Sample Artifacts

To test with no real training runs:

```bash
export TRAINING_DASHBOARD_ARTIFACT_ROOTS="$(pwd)/training-dashboard/sample_artifacts"
```

The sample directory contains `sample_run_001` with a realistic `train.log` and placeholder checkpoints.

---

## Expected Artifact Directory Layout

A **run directory** is any subdirectory under an artifact root that contains:
- `train.log` — the primary training log (text + trailing JSON), and/or
- `checkpoints/` — directory of `.pt` checkpoint files

```
artifact_root/
  my_run_001/
    train.log            # training log with metrics and summary JSON
    checkpoints/
      ep00500_step9580.pt
      ep01000_step19000.pt
      ...
  my_run_002/
    train.log
    checkpoints/
      ...
    benchmarks/          # optional: JSON benchmark files
      vs_random.json
      vs_greedy.json
```

### `train.log` format

The log contains two sections:

1. **Text header + metric lines:**
   ```
   DQN Training — 2000 episodes | obs_version=dqn_obs_v1 | device=cuda
     obs_size=292, action_count=209
     ...
     ep  100/2000 | steps 1978 | opt_steps 0 | eps 0.991 | ...
     ep  200/2000 | steps 3872 | ...
   ```

2. **Trailing JSON block** (starts with a line containing only `{`):
   ```json
   {
     "total_episodes": 2000,
     "total_env_steps": 40216,
     "periodic_evals": [...],
     "evaluation": {...}
   }
   ```

### Checkpoint filename pattern

```
ep{episode:05d}_step{step}.pt
```

Example: `ep00500_step9580.pt`

### External benchmark JSON

For standalone benchmark files (in `benchmarks/`, `evaluation/`, etc.):

```json
[
  {
    "checkpoint": "ep00500_step9580",
    "checkpoint_episode": 500,
    "opponent": "random_legal",
    "games": 100,
    "wins": 5,
    "losses": 90,
    "draws": 5,
    "win_rate": 0.05,
    "avg_game_length": 450.3,
    "illegal_action_count": 0
  }
]
```

Unknown fields are preserved and displayed.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status |
| GET | `/api/summary` | Aggregate counts across all runs |
| GET | `/api/runs` | List all discovered runs (sorted by modified date) |
| GET | `/api/runs/{run_id}` | Full detail for one run |
| GET | `/api/runs/{run_id}/metrics` | Parsed metric time series |
| GET | `/api/runs/{run_id}/benchmarks` | Benchmark results |
| GET | `/api/runs/{run_id}/diagnosis` | Rule-based training health checks |

---

## Dashboard Views

| View | Description |
|------|-------------|
| Run List | Table of all runs; click to open detail |
| Run Detail | Metadata, config, checkpoints, available files |
| Metrics | Line charts: loss, reward, epsilon, Q-max |
| Benchmarks | Table of evaluation results per checkpoint |
| Compare | Overlay metric curves and win rates for 2–4 runs |

---

## Known Limitations

- The backend parses the custom `train.log` format used by this project's DQN trainer. Runs using a different log format will show empty metrics (with a warning).
- The compare view overlays charts by stacking SVG elements — works best when runs have the same episode range.
- No authentication or access control (intended for local/dev use).
- Frontend development server only; for production use, build with `npm run build` and serve with a static file server.
