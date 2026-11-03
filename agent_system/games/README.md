# Game adapters and first multi-game slice

Quoridor delegates to the existing Rust engine and existing action/state codecs.
The Arena matrix now uses this adapter for restoration, observations and actions;
the legacy opening book, protocol, identities and output hashes remain unchanged.

Kuhn Poker delegates to OpenSpiel and is a small imperfect-information contract
check. It is NOT an implementation of Mr. Jack. The optional OpenSpiel research
Job backend now supports CFR and tabular Q-learning; see
[backend contract](../training/OPENSPIEL.md).
Its player information state includes only that player's visible history.

Run from the repository root in an isolated environment:

```sh
PYTHONPATH=.:backend-server:engine python -m pytest agent_system/games/tests
PYTHONPATH=.:backend-server:engine python scripts/run_game_match.py --game kuhn_poker --output /tmp/private-kuhn.json
```

The optional adapter was tested with open_spiel 2.0.2 on Python 3.12. Add it to a
separate environment, not the running service environment. Replays produced by
this command are explicitly research_private: they contain the true initial
state, chance events and seeds. Never serve this JSON through a public download.
replay(game, record, viewer) verifies the record and returns only the final
player observation; it is not an authorized HTTP API or full public replay UI.

The runner supports repeated actors, chance nodes, per-agent RNG, bounded
decision/event counts, terminal returns and separate failure/truncation reasons.
It does not impose a draw or a synthetic loss when the protocol truncates.
