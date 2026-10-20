# Quoridor model management

Author: Codex for Ji Hua
Created Date: 2026-10-20
Last Modified: 2026-10-20
Current Version: 1
Document Type: Implementation
Document Status: Active
Document Purpose: Operate checkpoint registration, paired evaluation and online deployment.

## Deployed services

- Gameplay on dev: `http://192.168.68.102:8765`
- Matrix and training dashboard: `http://192.168.68.88:8741/#arena`
- MLflow Tracking and Registry: `http://192.168.68.88:8742`
- Pinned-model Agent Service on training: `http://192.168.68.88:8743`

The dev backend connects to training's agent service through its ignored
`docker-compose.override.yml`. Its original agent container remains available
on port 8090. The training service runs under the existing user systemd manager,
with CPU inference and one PyTorch thread per process. GPU remains available for
training. MLflow runs in a separate Docker environment pinned to 3.17.1.
Services are intended for the existing private network.

Code is in `/home/jihua/workspace/boardgame-ai-arena` on training.
All new model copies, manifests, source snapshots, game records, evaluation
logs and MLflow data live in `/home/jihua/quoridor-data`, outside Git.
Do not add the pre-existing training artifacts or ignored audit reports to Git.

## Checkpoint identity

A model is identified by its full checkpoint SHA256, not by a mutable filename
such as `latest.pt`. `models/<sha256>.pt` is a verified, read-only copy.
The catalog records architecture, observation/action versions, training step,
episode count, source path and the reason for selection. Missing historical
commit and training seed information must remain unknown.

`quoridor-cnn-dqn` and `quoridor-mlp-dqn` are separate registered model names.
Each version contains the original checkpoint plus metadata. These are native
Quoridor checkpoint bundles, not MLflow pyfunc models.
Stable descriptive aliases identify historical versions.
`validation_status=evaluated` means the evaluation ran without an agent error;
it is not a claim that the model plays well.

## Reproduce and resume a matrix

From the repository on training:

```bash
.venv/bin/python model-management/prepare_catalog.py
docker compose -f model-management/compose.yml up -d --build
docker exec -i model-management-mlflow-1 python - < model-management/register_catalog.py
cp /home/jihua/quoridor-data/mlflow/registry.json /home/jihua/quoridor-data/registry.json
.venv/bin/python scripts/run_benchmark_matrix.py \
  --roster /home/jihua/quoridor-data/catalog.json \
  --output /home/jihua/quoridor-data/arena/my-new-benchmark \
  --pairs 25 --workers 8 --seed 20261020
docker exec -i model-management-mlflow-1 python - my-new-benchmark < model-management/log_benchmark.py
```

`--pairs 25` means **50 ranking games per unordered pair**, plus two standard
initial-position controls. Fixed opening samples and seeds are shared by all
matchups. A two-game block is atomic for resume. Re-running the exact command
resumes missing blocks only; changed code, models, protocol or manifest causes
rejection. Choose a new output directory after a change.

Each directory contains the immutable manifest, source archive, compressed
two-game records, and a live `report.json`. `arena/latest.json` selects the run
shown by the read-only dashboard. MLflow stores a parent evaluation run with
the complete evidence and one child run per agent, with per-opponent scores.
It links checkpoint hashes to registry versions.

See [benchmark protocol](../documents/agent/arena/benchmark-matrix.md) for the
opening distribution, draw rules, confidence intervals and Elo assumptions.
Primary ranking is the round-robin score; tied scores share a rank.
Elo is a regularized batch estimate anchored at random_legal=1500.

## Add a newly trained model

1. Save the checkpoint and record its training config, seed, code commit and
   observation/action versions. Include optimizer/replay state separately if
   resuming training is needed; a deployable checkpoint alone is not a complete
   training-resume snapshot.
2. Copy it into the SHA256 store, verify the hash, and create a **new** catalog
   JSON following `catalog.json`: add its metadata to `models` and a canonical
   DQN AgentSpec to `agents`. Use a new agent ID for the new hash. Keep old
   catalogs unchanged. `prepare_catalog.py` is the historical six-model import
   recipe, not a scanner that automatically deploys every checkpoint.
3. Register using `register_catalog.py --catalog /quoridor-data/catalog-v2.json
   --output /data/registry-v2.json` inside the container. Supply that catalog
   to the matrix runner. Run a predeclared evaluation before promoting.
   When logging this evaluation, pass `log_benchmark.py --registry
   /data/registry-v2.json RUN_ID` so model hashes resolve to the matching versions.
4. Export only the intended online roster with
   `export_runtime.py --catalog ... --output .../agent_defs-v2`. Update the
   systemd unit's `QUORIDOR_AGENT_DEFS_DIR` and restart during an idle period.
   The inference loader verifies SHA256 and encoder compatibility on loading.
5. For promotion, define the opponents and sample size in advance, require zero
   illegal/error forfeits, and compare against the incumbent on fresh openings.
   A conservative ±10 percentage-point 95% bound needs 185 pairs / 370 games.
   A suitable initial gate is a lower confidence bound above 50% against the
   incumbent, with separate baseline and cycle-rate checks. Selection followed
   by testing on the same book is not independent confirmation.

Use `challenger` for a nominated version and move `champion` only after the
declared gate passes. Keep the previous champion's version/hash for rollback.
The initial historical benchmark does not automatically appoint a champion.
Future training runs should log metrics to MLflow; the present integration
imports historical checkpoints and records evaluations.

## Online service and rollback

```bash
.venv/bin/python model-management/export_runtime.py
cp model-management/quoridor-agent.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now quoridor-agent
curl -f http://localhost:8743/health
```

The supplied unit uses the deployed paths above. Adjust them on another host.
The runtime environment needs the Rust engine, FastAPI/Uvicorn, PyYAML, NumPy
and PyTorch. The verified environment is Python 3.12.12, Torch 2.11.0+cu126,
NumPy 2.5.4, and PyYAML 6.0.3; inference is explicitly CPU.

To roll the dev backend back to its original agent service, change its
`AGENT_SERVICE_URL` override to `http://agent-service:8090` and run
`docker compose up -d --no-deps --no-build backend` on dev when no game is active.
No database or checkpoint changes are needed.

## Backup and scale

MLflow stores metadata in `mlflow/mlflow.db` and artifacts in `mlflow/artifacts`.
For a consistent local backup, stop the MLflow container when no import/logging
job is active, snapshot the entire `mlflow` directory, then restart it. Back up
the SHA256 model store, catalogs and evaluation directories too. Test restores
before relying on the backups. Move to PostgreSQL and object storage when
multiple concurrent training writers or capacity justify it.

References: [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/),
[registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/).

## Changelog

- Version 1: deploy historical checkpoint registry, paired benchmark evidence,
  dashboard, pinned online agents, and promotion/rollback workflow.
