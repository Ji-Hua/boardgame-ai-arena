#!/bin/sh
# Checks through the Dev entry after a release. Extra checks for this release live in checks.d/*.sh.
set -eu
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"
fail=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1" >&2; fail=1; }
code() { curl -s -o /dev/null -m 60 -w '%{http_code}' "$1"; }
for path in / /jobs /api/v1/jobs /api/v1/jobs/system /api/v1/models /api/v1/experiments /api/v1/play /api/v1/research/runs /api/v1/seasons; do
  c=$(code "$ENTRY$path"); if [ "$c" = 200 ]; then ok "$path 200"; else bad "$path $c"; fi
done
c=$(code "$API/health"); if [ "$c" = 200 ]; then ok "Training API health"; else bad "Training API health $c"; fi
for f in "$B"/checks.d/*.sh; do [ -f "$f" ] && . "$f"; done
echo "controller: $(systemctl --user show quoridor-jobs -p Description --value)"
exit $fail
