# Shared settings for a platform release (run on Training). release.env names the source commit and images:
#   COMMIT           Job controller source snapshot (git archive of the shared repository)
#   PLATFORM_IMAGE   Training research API image (compose service "platform"); empty keeps the current one
#   GATEWAY_IMAGE    gateway image of the Dev entry and of the Training API listener / redirect; empty keeps them
#   MLFLOW_URI       tracking server the Job controller logs finished Jobs to; empty disables logging
#   JOBS_ENV         extra controller environment, space-separated KEY=VALUE (e.g. JOBS_MODEL_INDEX=1)
#   SYNC_COMMAND     run by the controller after each model store refresh (one command line; may contain spaces)
#   LAB_CATALOG      Lab model catalog path (Training release.env LAB_MODEL_CATALOG); empty keeps the current one
B=$(cd "$(dirname "$0")" && pwd)
. "$B/release.env"
DEV=jihua@192.168.68.102
DEV_DIR=/home/jihua/quoridor-releases/p5-dev
R=/home/jihua/quoridor-releases/agent-ladder-preview
REPO=/home/jihua/workspace/boardgame-ai-arena
UNIT=$HOME/.config/systemd/user/quoridor-jobs.service
SRC=$HOME/quoridor-releases/jobs/src-$COMMIT
JOBS=http://172.17.0.1:18796
ENTRY=http://192.168.68.102:8790
API=http://192.168.68.88:8792

dc() { docker compose -p quoridor-agent-preview --env-file "$R/release.env" -f "$R/compose.yaml" "$@"; }
ddc() { ssh "$DEV" "cd '$DEV_DIR' && docker compose -p quoridor-entry --env-file release.env -f compose.dev-entry.yaml $*"; }
active_jobs() {
  curl -sf "$JOBS/api/v1/jobs" | python3 -c 'import json,sys; c=json.load(sys.stdin)["counts"]; print(c.get("preparing",0)+c.get("running",0))'
}
wait_ok() {
  for _ in $(seq 1 60); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done
  echo "not healthy: $1" >&2; return 1
}
restart_jobs() { systemctl --user daemon-reload; systemctl --user restart quoridor-jobs; wait_ok "$JOBS/health"; }
set_env() {  # set_env FILE KEY VALUE: replace or append KEY=VALUE
  if grep -q "^$2=" "$1"; then sed -i "s#^$2=.*#$2=$3#" "$1"; else echo "$2=$3" >>"$1"; fi
}
