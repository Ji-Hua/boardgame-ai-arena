# Agent ladder preview
Status: implementation in progress.
User authorizes an isolated preview only. Preserve all running service containers/configs/images and old data.

## Product contract
Two top-level entries: Ladder and Play. Agent versions are scoped to an immutable evaluation manifest and exact AgentSpec, not names. Read reported ranks/ratings, never invent Elo or pool different protocols. Agent detail provides pair results, recorded games, exact-SHA training Run links, metrics/config/checkpoints. Heuristic and neural types are distinct; random remains a separate type. Replay reconstructs recorded actions through Rust; per-step recorded facts remain separate from explicit recomputation, with candidates, branch exploration and diagnostics. Direct Play supports both player and Agent seats, materialized from the selected manifest only, with verified checkpoint bytes. No production roster or registry changes.

## Preview boundaries
Build separate immutable images; use a new Docker project/network, bounded CPU/memory, read-only external model/evidence mounts, distinct LAN preview port. No stop/recreate of existing services. Preview live sessions are bounded, in-memory and not ranked. Historical replay evidence is read-only. No public GitHub push while previous publication approval remains unresolved.

## Acceptance
Backend: manifest/hash/path isolation, pair filtering, training identity, replay validation, token/revision/legality, actual inference. Frontend: loading/errors, Agent types, exact-context navigation, race cancellation, training records, two-seat Play. Browser: real ladder -> pair -> replay -> per-ply analysis, linked training metrics, human/Agent and Agent/Agent play, mobile width. Observe protected container IDs and health unchanged.
