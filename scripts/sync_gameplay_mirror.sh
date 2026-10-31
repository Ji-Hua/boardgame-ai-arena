#!/bin/sh
# Pull the read-only evidence the Dev gameplay service needs from Training (P5).
# Mirrors /home/jihua/quoridor-data at the same absolute path, so catalog checkpoint paths and season
# manifests resolve unchanged. Never touches games/ (owned by Dev gameplay): excluded paths are not deleted.
set -eu
SOURCE=${MIRROR_SOURCE:-training:/home/jihua/quoridor-data/}
TARGET=${MIRROR_TARGET:-/home/jihua/quoridor-data/}
mkdir -p "$TARGET" "$TARGET/games"
exec rsync -a --delete --partial --timeout=120 \
  --exclude=/games/ --exclude=/mlflow/ --exclude='/jobs/job-*/' --exclude='/jobs/*.sqlite*' \
  --exclude='*.tmp' --exclude='.*.swp' \
  "$SOURCE" "$TARGET"
