# Agent ladder preview — 2026-10-25
Status: implemented, isolated preview available for user acceptance. No production replacement.

## Entry and source
- Preview: http://192.168.68.88:8790/ladder
- Direct Play: http://192.168.68.88:8790/play
- Source: /home/jihua/worktrees/quoridor/agent-ladder-preview
- Branch: feature/agent-ladder-preview
- Running application commit: fc8c61523f44e47854d6def3bf92b3f7147a46fa. Later commits record acceptance/tests/docs only.
- Existing Training and Dev public entries remain on the prior release.

## Implemented
Two top-level routes: Ladder and Play. Real immutable Arena reports supply season-scoped Elo/rank, W/D/L, AgentSpec and pair results. Four completed batches are discoverable; default latest CNN architecture batch contains 17 Agents. Reports are not pooled. Reported ranking limitations remain available in the UI.
Heuristic, neural and random Agents have distinct types. Agent detail links pair records and exact SHA training evidence. Training records are also available as a global index, including legacy runs not associated with a ranked Agent. Run source/status, actual metrics, configuration/provenance, checkpoints and notes are available. C-resnet seed42 resolves to Run 40f0edbd60314c7ab822b6dfbd8522ef and 60 training metric points; source-file fallback is disclosed when a registration run has no MLflow history.
Recorded games are read from original Arena gzip blocks. Manifest, opening, trajectory, final state and winners are verified using existing Rust reconstruction. Pure heuristic games now use the same validator without inventing a checkpoint SHA. Per-ply actual action and recorded decision duration are shown separately from explicit recomputation. Historical candidate scores/search trees are absent and labeled unrecorded; game-level diagnostic totals are not passed off as per-step facts.
Per-step Lab supports explicit heuristic/neural analysis, candidates, engine-derived branches, return to original position, input planes, activation slices, search trace and candidate sensitivity. Agent changes/step navigation discard stale asynchronous results. Q-values, policy logits and heuristic scores retain their semantics.
Direct Play and detail-origin Play use the same two-seat setup. Each seat accepts a human or a manifest-scoped Agent. The preview materializes the canonical AgentSpec via the existing factory, verifies neural SHA, and delegates legality/transitions to Rust. Agent/Agent can advance stepwise or automatically. Sessions are capability-token protected, revision checked, limited to four games, four hours and 500 plies; they are in memory, unranked, and not durable history. Model loading errors never select a replacement Agent. Live analysis uses an isolated position snapshot.

## Deployment
Separate Docker project/network: quoridor-agent-preview.
- Gateway publishes Training 8790 only.
- Platform API and preview Gameplay share the isolated platform process; no existing Dev Gameplay or Runtime is called.
- Lab is a separate private container using the existing accepted runtime image.
- External model/evidence and historical training artifacts are mounted read-only.
- Platform: 3 GiB / 2 CPUs; Lab: 2 GiB / 1 CPU; gateway: 128 MiB / 1 CPU.
- Immutable image IDs and saved Compose file are in /home/jihua/quoridor-releases/agent-ladder-preview.
- Existing production roster, champion aliases, MLflow database, source worktrees and old service configurations are unchanged.

To stop this preview only:
```sh
/home/jihua/quoridor-releases/agent-ladder-preview/manage.sh stop
```
To resume the same preview images:
```sh
/home/jihua/quoridor-releases/agent-ladder-preview/manage.sh up -d
```
Stopping platform loses preview sessions; it does not touch historical records or production games.

## Observed validation
- Backend: 129 tests run, 1 optional external-evidence test skipped, successful. New coverage includes manifest/path rejection, exact training association, pure heuristic replay, corruption rejection, capability/revision checks, illegal-action immutability and capacity.
- React: 29 tests passed, including new navigation and stale-analysis checks. Legacy shell tests target retained LegacyApp explicitly; they are not claimed as new UI acceptance.
- TypeScript and production image build passed; 21 Node contract tests passed.
- Three real-browser tests passed against 8790: real ladder/Run metrics/pair replay/per-ply neural diagnostics; heuristic Agent/Agent play plus snapshot analysis; exact CNN/human play and reload.
- Additional API acceptance: two different exact CNNs loaded, both SHA identities checked, six engine-legal plies, test game closed.
- Older-season heuristic/heuristic and neural/heuristic raw games reconstructed successfully.
- Windows browser loaded real ladder/detail successfully; Chinese text and compact layout visually inspected. Browser mobile layout passed at 390px.
- Seven previously running Training containers retained the same IDs, images and StartedAt values. Both old public entry health endpoints returned 200. No Dev write/deployment operation was performed for this preview.
Evidence: /home/jihua/quoridor-backups/agent-ladder-preview-evidence (logs, screenshots, browser traces/results). Release directory has protected-before/after.json.

## Limits and next decisions
This is a trusted-LAN preview without user accounts/TLS. Training is read-only evidence browsing, not job submission. Rank is report/protocol scoped, not a new global tournament. Unranked catalog models remain outside season rosters; no guessed merges. Unsupported/unavailable models fail explicitly at load time. Preview games are not persisted or added to ladder scores. No production switch, main promotion, GitHub push or old-directory cleanup is part of this delivery.
