# Mr. Jack London base-game ruleset

User confirmed the original base game, excluding expansions, on 2026-11-03.
The implementation pins `mr_jack/london-2016-base/v1`, based on the publisher's
current French London base-game booklet (2016 PDF metadata), not Pocket/New York.

Sources:
- Publisher: https://hurrican-edition.com/jeux/mr-jack-london/
- Rules and FAQ: https://hurrican-edition.com/wp-content/uploads/hurrican-games-mr-jack-london-regles.pdf
- Independently inspected implementation: https://github.com/Aghajari/Mr.Jack
  (`Mr.Jack/graphic/MrJack_Map.c`, `characters/`, `gameplay/`, Apache-2.0).
  Board cell numbering and factual layout were cross-checked with the publisher's
  setup diagram (PDF page 7). No SDL implementation, artwork, branding assets or AI
  was imported. The candidate GUI AI can access the secret Jack identity, so it
  was rejected as a research opponent. This is an independent headless rules engine.

## Implemented semantics
Eight rounds; D/J/J/D activation on odd rounds and J/D/D/J on even rounds.
Four character cards sampled without replacement on odd rounds, complementary
four on even rounds. Hidden identity and alibis use independent chance events.
The environment records actual events, not just seeds.

All eight characters, before/after powers, sewer travel, Stealthy traversal,
Gull swaps, Watson reorientation after any displacement, Holmes private alibis,
witness elimination, numbered lamp expiry, accusations, escape and survival are
implemented in one Python authority. Frontend geometry only renders state and
server-provided legal actions; it never determines legality.

Goodley distance is shortest street distance without sewers; each chosen
displacement must end strictly nearer, as specified by the FAQ. Intermediate
occupied cells are allowed; occupied destinations and special powers are not.

Explicit edge conventions pinned by this version:
- Crossing from an exit cell out of the district costs one movement point.
- If no legal Goodley pull remains, unused whistle movement is forfeited.
- If all alibis have been drawn, Holmes finishes without drawing.
- Identity is not disclosed automatically to spectators even at game end.
- Staged skill decisions count as RL decisions; character activations and rounds
  are separate concepts. Action IDs and encoder versions are immutable contracts.

Tests cover setup, all eight skills, private information, both victory modes,
round order/expiry, sewers, special movement, deterministic replay and sampled
complete games. This is research validation, not publisher certification.

## Learning baseline
SB3-Contrib MaskablePPO 2.9.2 + Stable-Baselines3 2.9.2 + Gymnasium 1.2.3.
A shared 64x64 tanh actor plays either role using a visible-summary encoder;
no true state, unknown alibis or opponent private information enters the encoder.
This is a memoryless baseline, not a claim of a sufficient information state.
Opponent baseline is random or a deliberately weak visible-information heuristic.
Training resets that terminate before the learner gets a decision are discarded;
evaluation always runs whole games without this conditioning.

Export is actor-only NumPy arrays, with SHA-256 dependency verification and
`allow_pickle=False`. SB3 training probability parity is checked on five
seeded decision states, tolerance 1e-5. Optimizer resume is not supported.
Mr. Jack reports paired-role mean returns and standard errors, not exploitability.
Evaluation pairs share environment seed; divergent action paths can consume
different chance events, so this is common-seed pairing, not identical deals
after diverging Holmes activations.

## Operational boundary
Research policies and eligibility never change the legacy Quoridor roster.
Private full-state trajectories remain on the research worker/data root;
public/player views, replay and analysis contexts are server-projected.
Mirror exclusions and missing-file/hash checks prevent partial bundles from
loading. A project catalog must be present and match the requested game.
