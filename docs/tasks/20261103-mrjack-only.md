# Mr. Jack scope correction

Status: complete.
User explicitly requested removing Kuhn Poker and continuing Mr. Jack while
retaining Quoridor. Source implementation: 475bc4e.

Removed game adapter, registry, solver, tabular runtime, job/profile, dependency,
CLI and dedicated tooling. Shared coverage now uses Mr. Jack. No extra game is
registered. Historical records are not exposed as supported projects.

Continued Mr. Jack with server-authorized board destination clicks, keyboard
support and Chinese skill phases. Tests: 67 games/Jobs + 11 Gameplay + 80 UI +
8 migration; typecheck/build; Chromium direct move/reload/replay; real Docker
three train + six evaluation jobs (120 games), cancellation and recovery.
Both new worker and platform were verified without OpenSpiel.
See [current release evidence](../MRJACK_RELEASE_20261103.md).
No live deployment or destructive data cleanup.
