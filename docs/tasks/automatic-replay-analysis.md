# Paused replay automatic acting-Agent analysis
Status: implementing
Scope: preview gateway 8790 only; no backend/service/data changes.
Behavior: debounce paused replay positions for 350 ms, follow the current Agent, cancel/invalidate on playback or seeking, never auto-run comparison models. Reuse up to 16 successful position/Agent results per mounted replay; explicit Run analysis refreshes. Terminal/unsupported actors do not auto-run. Existing live snapshot analysis stays manual. Abort prevents stale UI updates but cannot guarantee stopping already-running backend computation.
Acceptance: request-count tests for playback, pause, rapid seeks, caching, errors without retry loops, late responses; real-browser replay and existing regression flows. Preserve gateway rollback and all other container identities.
