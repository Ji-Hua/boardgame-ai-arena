# Supported games

Quoridor retains the Rust rule authority. Mr. Jack London base game uses the
headless Python authority documented in [MRJACK_RULES.md](MRJACK_RULES.md).
These are the only registered games. No additional games are enabled implicitly.

Mr. Jack training uses the pinned SB3-Contrib Maskable PPO backend.
Run an isolated private replay:

    PYTHONPATH=.:backend-server:engine python scripts/run_game_match.py --game mr_jack --output /tmp/private-mrjack.json

Full-state research records are private. Player and spectator API projections
must never return them.
