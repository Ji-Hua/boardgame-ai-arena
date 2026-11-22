#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${QUORIDOR_PYTHON:-$LAB_ROOT/.venv/bin/python}" -m uvicorn \
  agent_system.analysis.app:app --app-dir "$LAB_ROOT" \
  --host "${LAB_HOST:-127.0.0.1}" --port "${LAB_PORT:-8744}" "$@"
