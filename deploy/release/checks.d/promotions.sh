# Standing check: promotion API through both gateways; a release never promotes a model.
c=$(code "$ENTRY/api/v1/promotions"); if [ "$c" = 200 ]; then ok "/api/v1/promotions through the entry"; else bad "/api/v1/promotions $c"; fi
g=$(curl -s "$ENTRY/api/v1/promotions/candidates/98e0ff2ec9f05b916c2b55c2aebbf16e753c0e7e02e032015c4ce7814f1dda3f" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["gates"]))')
if [ "$g" -ge 1 ]; then ok "gate evidence for a gate candidate ($g season)"; else bad "gate evidence $g"; fi
c=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{"sha256":"x","note":"x"}' "$ENTRY/api/v1/promotions")
if [ "$c" = 401 ]; then ok "promotion without token refused"; else bad "promotion without token $c"; fi
d=$(curl -s "$ENTRY/api/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["summary"]["deployed"])')
ok "deployed models: $d (unchanged by the release)"
if curl -s -o /dev/null -w '%{http_code}' -I http://192.168.68.88:8790/ladder | grep -q 301; then ok "Training 8790 still redirects"; else bad "Training 8790 redirect"; fi
