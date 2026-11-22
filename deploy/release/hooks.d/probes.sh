# Standing hook: publish the versioned probe suites (write-once; an existing file must be identical).
(cd "$SRC" && PYTHONPATH="$SRC:$SRC/backend-server:$SRC/engine" /home/jihua/workspace/boardgame-ai-arena/.venv/bin/python \
  -m agent_system.evaluation.probes --data-root /home/jihua/quoridor-data)
