# Release deployment
These definitions prepare the approved distributed topology; no live rollout is implied.
They contain image references only, with no build or source mounts.
Generate validated nonsecret variables with scripts/project.py release-env. Supply machine paths and service credentials separately in a private environment file.
Use explicit -p project names and explicit --env-file paths. Render docker compose config before any authorized deployment.
Dev and Training are separate Compose applications, not cross-host Compose dependencies.
Keep live and preview project names, ports and state distinct. The delivered ports are current live endpoints; do not start these definitions alongside live services on the same ports.

## Required external inputs
Dev: AGENT_SERVICE_URL (Training runtime URL).
Training: QUORIDOR_DATA_DIR (existing data root), TRAINING_ARTIFACTS_DIR (existing evidence root until migrated).
Never point these variables at a development source root except the transitional read-only training-artifacts subdirectory.
No .env with credentials belongs in Git.

## Image contracts
web: existing frontend image contract, serves port 8765.
gameplay: backend image contract, serves 8764, consumes AGENT_SERVICE_URL.
runtime/lab: runtime-cpu image contract; /app includes agent_system, engine/game_manager, decision-lab and backend-server adapters; torch, numpy, PyYAML and engine wheel required.
research: existing training-dashboard image contract, ports 8740/8741.
mlflow: existing model-management image contract, port 8742 and /data persistence.
Do not substitute the old lightweight Dockerfile.agent image for runtime-cpu: it lacks neural runtime dependencies.

## Rollout and rollback
Archive current resolved configuration privately, image identities, model roster and state schema. Validate candidate endpoints in isolated ports. Deploy compatible Runtime before Gameplay; verify actual version and fixed model identity. Wait for active games or establish tested persistence before restart.
Rollback restores previous image digests and configuration; never restore an old database over new live writes without an approved database recovery plan.
Initial migration only validates manifests and renders Compose. Image build/load and real model acceptance are subsequent gates.
