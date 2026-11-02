#!/bin/sh
# Pull the read-only evidence the Dev gameplay service needs from Training (P5).
# Mirrors /home/jihua/quoridor-data at the same absolute path, so catalog checkpoint paths and season
# manifests resolve unchanged. Never touches games/ (owned by Dev gameplay): excluded paths are not deleted.
# legacy-artifacts/ (historical training runs, read by the Training research API only) is not mirrored.
set -eu
SOURCE=${MIRROR_SOURCE:-jihua@192.168.68.88:/home/jihua/quoridor-data/}
TARGET=${MIRROR_TARGET:-/home/jihua/quoridor-data/}
mkdir -p "$TARGET" "$TARGET/games"
exec rsync -a --delete --partial --timeout=120 \
  --exclude=/games/ --exclude=/mlflow/ --exclude='/jobs/job-*/' --exclude='/jobs/*.sqlite*' --exclude=/legacy-artifacts/ \
  --exclude='*.tmp' --exclude='.*.swp' \
  "$SOURCE" "$TARGET"
