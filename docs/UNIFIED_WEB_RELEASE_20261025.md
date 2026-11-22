# Unified frontend release — 2026-10-25

## Verified outcome
The unified frontend is live at http://192.168.68.102:8765 and http://192.168.68.88:8741 (8740 also retains API access). Both hosts serve identical local JS/CSS assets from immutable gateway image IDs. Training remains the reference source; deployment is distributed.

- Training source: `/home/jihua/worktrees/quoridor/monorepo-migration-20261025`, branch `codex/monorepo-migration-20261025`.
- Dev source: `/home/jihua/worktrees/quoridor/unified-web-20261025`, branch `codex/unified-web-dev-20261025`.
- Released application source: `414cc483ce9f83e306a4d4501fa9315667f54b63`; later documentation commits do not change the running image.
- Code synchronized privately with Git fetch/fast-forward, not source copying. GitHub main was not merged or pushed; no new repository created.

## Application layout and behavior
`apps/web` is the sole active React build for Play, Training, Arena, Replay, Lab and Models. `packages/board-ui` owns reusable board source, coordinate translation and player colors. Compatibility symlinks preserve former frontend build/public paths. Classic controllers remain in `frontend/src`, imported into the same build; this is deliberate source reuse, not a second active frontend.
Native React Lab replaces iframe/external navigation for normal research flows: exact replay ply and model SHA, edited or imported positions, explicit heuristic/neural analysis, candidate branches, tensor/layer views, sensitivity and capability-gated search. Play supports an isolated frozen snapshot and return to the same game; it does not pause the live game. Model identity is never silently substituted.
Root npm workspace commands, AGENTS instructions, task documentation and a frontend CI workflow support agent development. The workflow is committed but has not run on GitHub.

## Running deployment and preserved dependencies
| Host | Component | Address / dependency |
|---|---|---|
| Dev | Unified static gateway | 8765; `quoridor-unified-entry` |
| Dev | New research Gameplay | 8776; `quoridor-unified-gameplay`; Training runtime 8783 |
| Dev | Existing Classic Gameplay | 8764; unchanged; `/classic-api` and `/ws` proxy here |
| Training | Unified static gateway | 8741 and 8740; `quoridor-unified-entry` |
| Training | Platform API | 8781; `quoridor-unified-platform-1` |
| Training | Research runtime | 8783; `quoridor-unified-runtime-1` |
| Training | Research Lab | container-network `lab:8744`; `quoridor-unified-lab-1` |
| Training | Existing Agent / standalone Lab / MLflow | 8743 / 8744 / 8742; unchanged and healthy |

The Platform still mounts the old Training main checkout's `agent_system/training/artifacts` read-only for historical evidence. That directory must remain until a separate artifact migration is approved and verified. The old Classic backend source/configuration is also an active dependency. No historical checkout may be removed just because the new frontend is live.

Runtime/Lab use `/home/jihua/quoridor-data` read-only. New immutable research catalog JSON files were added for this release; selected catalog files are world-readable for the non-root containers. Existing model/data bytes, MLflow database and production agent definitions were not replaced. Training jobs/GPU execution are outside this frontend release.

## Immutable images
- Runtime / Lab / research Gameplay: `sha256:5746cf19c0c0eb126759cfe24a77d60953bdf6d4d64fa3c06409db5055a4ee7f`, source `fae4fa19837634f68a592c2a36cf121c4e6bb4e2`.
- Platform / web build: `sha256:d81e21d6738705952ad5d56b84bc491852588993f86e11fe7558af0bcb025482`, source `414cc48`.
- Gateway including static assets: `sha256:6b6d0b69867eab0dc8a76d454c8428abdcadcc07d04c5c02322e93915550b288`, source `414cc48`.
- Final JS SHA256: `4837298750224169d53d1037184d13358937049f0808dfa4990163df860e63e2`.
- Final CSS SHA256: `724238252d976b3bf2666ccf2ea0f90a8efbe6696e3b7479a873afeed5d15d47`.

## Verification
- TypeScript, production build and 27 React UI tests passed in the final image build; 21 Node contract tests passed during this stage.
- Nine candidate real-browser tests passed, including complete P1/P2 games, offline recovery, illegal/stale actions, exact replay-to-Lab neural diagnostics and Classic WebSocket.
- Live HTTP revealed `crypto.randomUUID` was unavailable outside a secure context. Both entries were rolled back successfully. Replaced it with cryptographic `getRandomValues` for the local snapshot key, and added a regression case with randomUUID disabled.
- Six browser flows passed against the corrected LAN HTTP candidate, including exact neural analysis and snapshot round-trip. Three affected native/Classic flows then passed on the final live Dev entry.
- All six workspace routes, health, Classic API, static asset hashes and missing-asset 404 behavior checked across both live hosts (including Training 8740).
- Original tracked/untracked status matched the initial backup manifest for all 13 recorded Git directories, including the unfinished Dev merge and Training dirty source.
- Local mobile screenshot inspected. Preview gateways are stopped; no old source directory was deleted.
Evidence: `/home/jihua/quoridor-backups/unified-web-evidence-20261025` on Training. Test results are observed local runs, not a claim of GitHub CI or exhaustive validation of every historical algorithm.

## Preservation and executable rollback
Original full source/Git/ignored-file archives remain:
- Dev: `/home/jihua/quoridor-backups/20261025T162123Z`.
- Training: `/home/jihua/quoridor-backups/20261025T162124Z`.

Additional private pre-switch container/image/config archives, with SHA256SUMS:
- Dev: `/home/jihua/quoridor-backups/pre-unified-web-20261025T193651Z`.
- Training: `/home/jihua/quoridor-backups/pre-unified-web-20261025T193650Z`.

Old containers are stopped and retained: Dev `quoridor-frontend-1`, Training `training-dashboard-training-dashboard-1`. Their original images and configs remain. Backups are same-host copies, not independent disaster recovery.

On either host, execute:
```sh
/home/jihua/quoridor-releases/unified-web-20261025/rollback-entry.sh
```
This stops only the new public gateway and starts that host's old frontend container. It was actually exercised successfully during LAN acceptance. It does not restore or modify databases, restart old Gameplay, or terminate the new research services. To re-enable the same accepted release after rollback, stop that host's old frontend container and run `docker start quoridor-unified-entry`.

Private release directory on each host: `/home/jihua/quoridor-releases/unified-web-20261025`. It contains start/rollback scripts, immutable image receipt, container inspection, service environment and keys. Do not commit these private files. Training also has a saved Compose file and `manage-training.sh`; manage active compute with `up -d runtime lab platform`, not an indiscriminate command that would re-enable the preview gateway.

## Remaining decisions / limits
Public GitHub publication still needs explicit consent: a previous automatic approval review rejected pushing operational environment information into the public repository. This does not block private deployment or Git synchronization between the hosts.
Main-branch promotion, historical unique-code review, old directory deletion, GPU images, Jobs, durable sessions and genuine live pause/resume remain separate tasks. This release unifies the frontend; it does not claim every historical source tree has been merged.
