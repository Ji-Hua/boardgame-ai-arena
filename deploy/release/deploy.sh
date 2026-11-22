#!/bin/sh
# Platform release: Job controller from source $COMMIT, Training platform image, Dev entry gateway image.
# The compose files come from the release source snapshot. Lab and gameplay use the same tested source as platform.
set -eu
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"
n=$(active_jobs)
[ "$n" = 0 ] || { echo "refusing: $n Job(s) preparing or running; retry when idle" >&2; exit 1; }

P=$B/previous
if [ ! -f "$P/saved" ]; then
  mkdir -p "$P"
  cp "$UNIT" "$P/quoridor-jobs.service"
  ssh "$DEV" "cat '$DEV_DIR/sync_gameplay_mirror.sh'" >"$P/dev-sync_gameplay_mirror.sh"
  cp "$R/release.env" "$P/training-release.env"; cp "$R/compose.training-api.yaml" "$P/training-compose.yaml"
  ssh "$DEV" "cat '$DEV_DIR/release.env'" >"$P/dev-release.env"
  ssh "$DEV" "cat '$DEV_DIR/compose.dev-entry.yaml'" >"$P/dev-compose.yaml"
  grep -q '^GATEWAY_IMAGE=sha256:' "$P/dev-release.env" && grep -q '^PREVIEW_IMAGE=' "$P/training-release.env"
  touch "$P/saved"
fi

echo "== source snapshot $COMMIT"
if [ ! -d "$SRC" ]; then
  rm -rf "$SRC.partial"; mkdir -p "$SRC.partial"
  git -C "$REPO" archive "$COMMIT" | tar -x -C "$SRC.partial"
  mv "$SRC.partial" "$SRC"
fi
for hook in "$B"/hooks.d/*.sh; do [ -f "$hook" ] && { echo "== hook $(basename "$hook")"; . "$hook"; }; done

echo "== Job controller"
sed -e "s#^WorkingDirectory=.*#WorkingDirectory=$SRC#" \
  -e "s#^Environment=PYTHONPATH=.*#Environment=PYTHONPATH=$SRC#" \
  -e "s#^Environment=JOBS_SOURCE_REF=.*#Environment=JOBS_SOURCE_REF=$COMMIT#" \
  -e "s#^Description=.*#Description=Quoridor main $COMMIT#" \
  -e '/^Environment=JOBS_MLFLOW_URI=/d' -e '/^Environment=JOBS_MODEL_INDEX=/d' -e '/JOBS_SYNC_COMMAND=/d' \
  "$P/quoridor-jobs.service" >"$UNIT.partial"
for pair in ${MLFLOW_URI:+JOBS_MLFLOW_URI=$MLFLOW_URI} ${JOBS_ENV:-}; do
  sed -i "/^Environment=JOBS_MAX_CPU_JOBS=/a Environment=$pair" "$UNIT.partial"
done
if [ -n "${SYNC_COMMAND:-}" ]; then
  sed -i "/^Environment=JOBS_MAX_CPU_JOBS=/a Environment=\"JOBS_SYNC_COMMAND=$SYNC_COMMAND\"" "$UNIT.partial"
fi
mv "$UNIT.partial" "$UNIT"
restart_jobs

echo "== Training platform"
cp "$SRC/deploy/compose.training-api.yaml" "$R/compose.training-api.yaml"
[ -z "${PLATFORM_IMAGE:-}" ] || set_env "$R/release.env" PREVIEW_IMAGE "$PLATFORM_IMAGE"
# The Training API listener and the 8790 redirect run the same gateway image as the Dev entry.
[ -z "${GATEWAY_IMAGE:-}" ] || set_env "$R/release.env" GATEWAY_IMAGE "$GATEWAY_IMAGE"
services="platform gateway redirect lab"
if [ -n "${LAB_CATALOG:-}" ]; then
  for _ in $(seq 1 120); do [ -s "$LAB_CATALOG" ] && break; sleep 1; done
  [ -s "$LAB_CATALOG" ] || { echo "Lab catalog was not written: $LAB_CATALOG" >&2; exit 1; }
  set_env "$R/release.env" LAB_MODEL_CATALOG "$LAB_CATALOG"; services="$services lab"
fi
dc up -d $services
# nginx resolves upstream names once; a recreated platform or Lab gets a new address.
dc restart gateway
wait_ok "$API/health"

echo "== Dev entry gateway"
if [ -n "${GATEWAY_IMAGE:-}" ]; then
  ssh "$DEV" "docker image inspect $GATEWAY_IMAGE >/dev/null 2>&1" \
    || docker save "$GATEWAY_IMAGE" | gzip -1 | ssh "$DEV" "gunzip | docker load" >/dev/null
  ssh "$DEV" "sed -i 's#^GATEWAY_IMAGE=.*#GATEWAY_IMAGE=$GATEWAY_IMAGE#' '$DEV_DIR/release.env'"
fi
ssh "$DEV" "cat > '$DEV_DIR/compose.dev-entry.yaml'" <"$SRC/deploy/compose.dev-entry.yaml"
ssh "$DEV" "cat > '$DEV_DIR/sync_gameplay_mirror.sh.partial' && chmod +x '$DEV_DIR/sync_gameplay_mirror.sh.partial' && mv '$DEV_DIR/sync_gameplay_mirror.sh.partial' '$DEV_DIR/sync_gameplay_mirror.sh'" <"$SRC/scripts/sync_gameplay_mirror.sh"
if [ -n "${PLATFORM_IMAGE:-}" ]; then
  ssh "$DEV" "docker image inspect $PLATFORM_IMAGE >/dev/null 2>&1" \
    || docker save "$PLATFORM_IMAGE" | gzip -1 | ssh "$DEV" "gunzip | docker load" >/dev/null
  ssh "$DEV" "sed -i 's#^PLATFORM_IMAGE=.*#PLATFORM_IMAGE=$PLATFORM_IMAGE#' '$DEV_DIR/release.env'"
fi
ddc up -d gameplay gateway
wait_ok "$ENTRY/health"
"$B/check.sh"
