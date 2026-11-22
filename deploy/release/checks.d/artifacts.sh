# Standing check: historical training runs come from the verified legacy copy, not the original checkout.
m=$(docker inspect quoridor-agent-preview-platform-1 --format '{{range .Mounts}}{{if eq .Destination "/data/training_artifacts"}}{{.Source}}{{end}}{{end}}')
if [ "$m" = /home/jihua/quoridor-data/legacy-artifacts/training-artifacts ]; then ok "training artifacts from $m"; else bad "training artifacts mount $m"; fi
n=$(curl -s "$ENTRY/api/v1/research/runs" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["runs"]))')
if [ "$n" -ge 100 ]; then ok "research runs $n"; else bad "research runs $n"; fi
