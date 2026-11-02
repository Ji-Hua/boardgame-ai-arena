# Standing check: verified model store and the Lab reading the merged index.
v=$(curl -s "$ENTRY/api/v1/models" | python3 -c 'import json,sys; s=json.load(sys.stdin)["summary"]; print(s["sha_verified"], s["total"])')
if [ "${v% *}" -gt 0 ] && [ "${v% *}" = "${v#* }" ]; then ok "models sha_verified $v"; else bad "models sha_verified $v"; fi
lab=$(curl -s "$ENTRY/api/v1/lab/models" | python3 -c 'import json,sys; m=json.load(sys.stdin)["models"]; print(len(m), sum(x["available"] for x in m))')
want=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["models"]))' "$LAB_CATALOG")
if [ "${lab% *}" = "$want" ] && [ "${lab#* }" = "$want" ]; then ok "Lab lists the merged index ($lab of $want)"; else bad "Lab models $lab, index $want"; fi
