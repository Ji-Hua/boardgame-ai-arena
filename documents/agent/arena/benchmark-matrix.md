# Quoridor Benchmark Matrix

Author: Codex for Ji Hua
Created Date: 2026-10-20
Last Modified: 2026-10-20
Current Version: 1
Document Type: Implementation
Document Subtype: Reproducible evaluation and model management
Document Status: Active
Document Authority Scope: Agent evaluation and deployment implementation
Document Purpose: Specify how existing AgentSpec definitions and learned realizations are compared reproducibly without confusing repeated deterministic games with independent evidence.

## Evaluation protocol

Use the canonical AgentSpec -> Scorer -> Policy -> AgentInstance pipeline for
all participants. DQN scoring loads a checksum-pinned checkpoint and selects
the checkpoint's own observation encoder. Networks need not share architecture.
Only legal actions receive scores. NaN/Inf Q values are errors.

The first benchmark contains 25 paired opening samples. Each sample is a legal,
nonterminal position after four random plies: choose a pawn move with probability
0.7, otherwise a legal wall; choose uniformly within that action family. Draw
openings independently and report the number of unique states. Store the
complete opening actions and seeds before running.
Every matchup uses the same book. Each sample is played twice with the agents
exchanging P1/P2, for 50 games per unordered pair. RNG streams follow agent
identity across the seat swap. This measures strength on this explicit opening
distribution, not the standard initial position alone.

Also play one seat-swapped pair from the standard initial position per matchup.
These two control games are reported separately and excluded from the sampled
opening ranking. Do not count repeated deterministic initial games as new trials.

Use a 500-ply cap, consistent with existing Arena, and count capped games as
draws. A repeated exact state can terminate as a draw only if both agents have
deterministic, memoryless policies. An illegal action or runtime error forfeits
the game and must remain visible in the report. Record every game's agents,
hashes, opening, seeds, P1/P2, outcome, reason, actions and decision time.

Write outputs outside Git. Resume only when the pinned manifest, source digest,
opening book and completed game records agree; never append a different
experiment to an existing result directory.

## Scores, confidence and ratings

The matrix entry is (wins + 0.5 draws) / games for the row agent. Show W/L/D,
both seat splits and termination reasons alongside it. Rank by equally weighted
round-robin score against the declared roster. Do not conceal cycles or
nontransitive pairwise results behind one rating.

The two seat-swapped games are one sampling block. For a fixed sample size,
use a conservative 95% Hoeffding interval on the mean of block scores in [0,1]:
half-width = sqrt(log(2/0.05)/(2N_pairs)), clipped to [0,1]. This avoids a
zero-width bootstrap interval after all wins/losses. It assumes sampled openings
represent the declared distribution; it says nothing about arbitrary human
positions. Accidental duplicate samples are retained as draws from that distribution.
Intervals are marginal, not simultaneous claims across all matchups.

With 25 pairs the bound is about ±27 percentage points, so this is a screening
benchmark. A predeclared ±10-point bound requires 185 pairs / 370 games.
Any follow-up benchmark chooses its sample size before it starts; repeatedly
checking ordinary fixed-sample intervals is not an adaptive stopping rule.

Also fit batch Elo using the logistic expected-score model (400-point scale),
draws worth 0.5, and a zero-centered 400-point Gaussian regularizer. Anchor
random_legal at 1500. Batch fitting avoids dependence on arbitrary execution
order. Ratings are relative to this roster/protocol and are provisional.
Equal total scores share a rank. Greedy and depth-1 Minimax can produce
identical records; this roster deliberately includes both as an implementation
comparison, and their presence weights this style twice in the overall score.
For a future live ladder, update a complete seat-swapped pair atomically with
K=20 using its mean score and the pre-pair ratings. Store the event sequence
and protocol ID. A new checkpoint hash is a new entrant.

## Model management

Use MLflow Tracking and Model Registry on training, with a SQLite metadata
database and persistent local artifact storage. Keep it separate from the
training virtualenv. Model versions contain immutable checkpoint copies,
SHA256, original run directory, model/observation/action metadata and provenance.
Unknown historical training commit/seed information stays unknown.

Maintain separate registry names for CNN and MLP DQN, versions for checkpoints,
and aliases such as challenger and champion. Evaluation IDs bind model hashes
to rules, code digest, roster, openings and complete game evidence. Promotion
requires compatible encoders, zero illegal/error forfeits and a predeclared
benchmark gate; highest round-robin score alone is not approval for promotion.
Only explicitly selected versions enter the runtime roster. The inference loader
uses the Quoridor checkpoint format; MLflow registry entries are not advertised
as generic pyfunc models.

Start with SQLite because there is one evaluation coordinator and one training
host. Back up both metadata DB and artifact directory consistently. Move metadata
to PostgreSQL and artifacts to object storage when concurrent writers or storage
growth justify it. Git contains code/configuration only.

Operational commands and promotion/rollback procedures are in
[model-management README](../../../model-management/README.md).

References:
- [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [MLflow registry aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

## Changelog

Version 1 (2026-10-20)
- Define seat-balanced matrix, uncertainty, Elo, reproducibility and registry workflow.
