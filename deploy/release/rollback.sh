#!/bin/sh
# Restore the controller unit, the Training compose and env, and the Dev entry files saved by deploy.sh.
set -eu
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"
P=$B/previous
[ -f "$P/saved" ] || { echo "nothing saved to roll back to" >&2; exit 1; }
n=$(active_jobs || echo 0)
[ "$n" = 0 ] || echo "warning: $n Job(s) active; the controller reconciles them after the restart"
cp "$P/quoridor-jobs.service" "$UNIT"
restart_jobs
cp "$P/training-release.env" "$R/release.env"; cp "$P/training-compose.yaml" "$R/compose.yaml"
dc up -d platform gateway redirect lab
# nginx resolves upstream names once; a recreated platform or Lab gets a new address.
dc restart gateway
wait_ok "$API/health"
ssh "$DEV" "cat > '$DEV_DIR/release.env'" <"$P/dev-release.env"
ssh "$DEV" "cat > '$DEV_DIR/compose.dev-entry.yaml'" <"$P/dev-compose.yaml"
ssh "$DEV" "cat > '$DEV_DIR/sync_gameplay_mirror.sh'" <"$P/dev-sync_gameplay_mirror.sh"
ddc up -d gateway
wait_ok "$ENTRY/health"
echo "rolled back: $(systemctl --user show quoridor-jobs -p Description --value)"
