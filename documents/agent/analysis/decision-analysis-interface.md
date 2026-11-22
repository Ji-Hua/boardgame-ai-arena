# Decision Analysis Interface

Author: Ji Hua / Codex
Created Date: 2026-10-20
Last Modified: 2026-10-20
Current Version: 2

Document Type: Interface
Document Subtype: Developer Analysis API and Evidence
Document Status: In Development
Document Authority Scope: Agent System / Decision Analysis
Document Purpose:
Define the standalone HTTP interface, in-process observation boundary, position
format, and evidence semantics implemented by Decision Lab.

---

## 1. Entry points and dependencies

`agent_system.analysis.app:app` exposes a standalone FastAPI service. Its same-origin
frontend is served from `decision-lab/web/`. It imports the Rust Engine, canonical
Agent Spec model, and the public scoring functions of the existing heuristic agents.
Neural routes lazily import PyTorch and reuse `runtime.inference.dqn.DQNInference`.
It has no operational dependency on gameplay Backend, training, Arena, MLflow, or
Agent Service.

## 2. Position and action representation

```json
{
  "pawns": [[4, 0], [4, 8]],
  "walls_remaining": [10, 10],
  "walls": [],
  "turn": 1
}
```

Coordinates are `[x, y]`, zero-based. `a1 = [0,0]` is bottom-left in the analysis
board. P1's goal is `y=8`; P2's goal is `y=0`.
This component does not use the legacy frontend's row/column rendering convention.
Walls are objects `{"x":3,"y":4,"orientation":"horizontal"}` with anchor coordinates
0–7. A horizontal wall blocks the upward edges at `(x,y)` and `(x+1,y)`.
A vertical wall blocks the rightward edges at `(x,y)` and `(x,y+1)`.

Validation rejects out-of-range or non-integer coordinates, coincident pawns,
simultaneous winners, duplicate/overlapping/crossing walls, inaccessible goals, and
inventories where the two stocks plus placed walls do not total 20.
Each stock is 0–10. Unknown fields are rejected.
Nonterminal wall layouts are reconstructed using Rust wall placements.
Terminal layouts undergo geometry and Rust path validation without attempting
new actions after game over. Historical reachability is not certified.

Actions use the existing wire shape:
`{"player":1,"type":"pawn","target":[4,1]}`. Types are `pawn`, `horizontal`, `vertical`.
The Engine validates actual moves, pawn jumps, stocks, turn, and wall legality.
No large bitset integer crosses the JavaScript boundary.

## 3. HTTP contract

| Endpoint | Request | Result |
|---|---|---|
| `GET /health` | — | status and component name |
| `GET /api/meta` | — | presets, provenance, resource limits |
| `GET /api/models` | — | configured model-management catalog, availability, optional registry versions |
| `POST /api/position` | Position | validated snapshot |
| `POST /api/analyze` | `{position, algorithm, depth, model_id?}` | complete candidate evidence and chosen action |
| `POST /api/branch` | `{position, algorithm, depth, model_id?, action}` | branch diagnostics and after snapshot |
| `POST /api/neural/tensor` | `{position, model_id, kind, layer, channel?, input_channel?, offset?}` | bounded activation or weight slice |
| `POST /api/apply` | `{position, action}` (algorithm/depth optional) | Engine-produced after snapshot |
| `GET /` | — | independent developer UI |
| `GET /assets/*` | — | frontend modules/styles |

`algorithm` is `greedy`, `minimax`, or `dqn`; default `greedy`. `depth` is an integer 1–3,
default 2, counting half-moves including the root action. Greedy reports effective
depth 1. Its depth parameter is not used for scoring.
DQN requires `model_id`, ignores search depth and reports effective depth 1.
It performs direct Q inference, not lookahead search.

The `dqn` API adapter also accepts compatible CNN policy checkpoints with
`model_config.algorithm=policy_distillation`. These return unnormalized policy
logits, not Q/return estimates. `neural.model.output_semantics` is `policy_logits`
for those models and `action_values` for historical DQN models. The UI uses the
corresponding logit/Q labels, including candidate scores and input gradients.
Legacy field names (`q`, `margin`, and diagnostic `max_abs_legal_q`) remain for
API compatibility; their numeric interpretation follows model metadata.

Snapshots include canonical `position`, P1/P2 `distances`, display `paths`,
`winner` (1/2/null), and `legal_actions`. The display BFS ignores the other pawn and
checks each projected path length against Rust's distance helper.

Invalid inputs/actions and exhausted search budgets return HTTP 422 with an error.
At most two requests enter the analysis computation section simultaneously;
additional requests receive 429. No partial recommendation is returned.

Minimax search limits are 75,000 recursive visits or six seconds, checked on entry
to recursive nodes. This is a cooperative limit, not a hard process deadline.
The complete result is required before an action is recommended.

## 4. Candidate and score semantics

Heuristic candidate order is the Engine's enumeration order. DQN candidates are
sorted by encoded action index, matching Runtime/Arena. Each candidate includes:

- `id` (`type:x:y`), `order`, action and algebraic label;
- `value` and `bound` (`exact`, `upper`, `lower`);
- before/after distances in **root self, opponent** order;
- `delta_self`, `delta_opp`, `greedy_score`, and immediate-win flag;
- node and cutoff diagnostics.

Greedy uses `delta_opp - delta_self`. It has no separate terminal override.
Minimax nonterminal leaves use `distance(opponent) - distance(self)`; terminal
leaves use ±infinity. JSON encodes these as strings `"+inf"` and `"-inf"`.
Response bodies contain no nonstandard numeric Infinity or NaN.

Root Minimax search shares alpha across candidates, matching serving. After an
earlier candidate raises alpha, another candidate can fail low and return only
an upper bound. The UI must label this with `≤`; it must not claim that sorting
root returns yields a precise ranking of all inferior candidates.
When candidates tie, heuristic Top1 preserves Engine order; DQN selects the smallest
legal action index. DQN `bound=exact` indicates the actual forward output only,
not exact game value; the frontend labels it Q.

Each analyzed response embeds the canonical `AgentSpec` serialized through the
existing definition model: `greedy`, `minimax_dN`, or the selected DQN catalog ID,
matching parameters and `top_k: k=1`. DQN parameters include the pinned path, SHA256,
and CPU device. There is no independent agent registry. Stochastic selection,
arbitrary browser-supplied paths, and checkpoint uploads are unsupported.

## 5. Shared implementation and observer contract

Public in-process entries:

- `greedy_agent.score_position(state)` returns ordered score/evidence rows.
- `minimax_agent.score_position(state, depth, observer=None)` returns ordered
  serving-search rows and `SearchContext`.
- `minimax_agent.inspect_branch(state, action, depth, observer=None)` returns
  the value from a fresh `[-∞,+∞]` search window and `SearchContext`.

Serving `make_action` consumes the same scoring entries. The optional observer
receives events in recursion order:

| Event | Data |
|---|---|
| `root_action` | candidate Engine action |
| `enter` | RawState, remaining depth, incoming alpha/beta, maximizing flag |
| `edge` | action about to be traversed |
| `exit` | returned value, leaf flag, count of unvisited direct children after cutoff |

Observers are diagnostic consumers, not action selectors. A resource-limit exception
may stop an analysis request; no observer is attached during ordinary serving.
The hooks do not alter legal action order, scoring, shared alpha, or policy selection.

## 6. Exact branch, PV, and trace

For Minimax, clicking a candidate performs an independent full-window search. `bound=exact`
means its minimax value at this depth and heuristic, **not** solved game value.
The returned principal variation (PV) is replayed through Rust and contains the
root move followed by best responses until the horizon or a terminal state.

Trace nodes record id, parent, incoming action, resulting position, MAX/MIN role,
remaining depth, original alpha/beta window, score/bound, leaf status, recorded
children, visited child count, and unvisited direct children after a cutoff.
Bounds refer to the original incoming window; leaf evaluations are exact.
All scores remain in the root player's perspective.

Trace storage retains the first 1,200 visited nodes. `recorded`, `visited`, and
`omitted` report storage truncation explicitly. Storage truncation does not prune
the search or alter its value/PV; it is distinct from alpha-beta cutoffs.
The UI can preview recorded positions and PV steps without changing the research root.

Global analysis `stats.nodes` counts recursive visits. Greedy counts candidates.
Per-candidate Minimax `nodes` retains the legacy runtime counter, which includes
one extra root-candidate increment. These counts must not be summed and mislabeled
as recursive visits.

## 7. Reproducible cases

`quoridor-decision-analysis-v1` identifies heuristic evidence;
`quoridor-neural-analysis-v1` identifies DQN evidence. Provenance includes
the diagnostic/scoring source SHA-256, installed Engine binary hashes, and tie/policy
semantics. The position receives a canonical JSON SHA-256.

Browser exports use `quoridor-decision-case-v1` and include position, timestamp,
analysis/Agent Spec, selected branch, optional comparisons, current neural tensor slice,
and provenance. DQN evidence additionally pins checkpoint SHA256, PyTorch version,
CPU device, and source hashes covering model construction, loading, encoding,
action mapping, shared inference, and diagnostic code.
Exported cases are downloaded to the user's chosen location; the service does not
write artifacts into Git or register models.

Import accepts a Position or an exported object's `position`. Stored scores and
traces are not trusted as live analysis: importing validates the position and
recomputes with current code. Reproducing historical scores requires the matching
source and Engine fingerprints; fingerprints do not bundle historical source.

## 8. Neural diagnostics

Hierarchical CNN checkpoints add a two-stage category/action decision without changing
the 209-action encoding. Both category Q values, category availability, conditional
selection and the corresponding gradients are exposed; see the
[hierarchical CNN contract](../training/cnn-hierarchy.md#decision-lab-contract).

`GET /api/models` reads `LAB_MODEL_CATALOG` (default `~/quoridor-data/catalog.json`).
Only IDs in its `models` list resolve to checkpoint paths. Optional `registry.json`
in the same directory joins registry metadata by SHA256, never by a mutable alias.
Missing catalogs return an empty list and a configuration message. Missing files,
unknown IDs, hash mismatches, incompatible encoders and nonfinite outputs are errors.
The shared inference object exposes checkpoint metadata without retaining optimizer state.

Models run in eval mode on CPU with one Torch thread, at most two cached networks,
and serialized per-model forward hooks/gradient requests. File inode, size and
modification time participate in cache identity; changed files reload and revalidate
the hash. Limits are 512 MB per checkpoint and 20M loaded parameters. These are
resource bounds, not a hard inference deadline. Standard two-request admission limits apply.
Neural dependency import failure returns 503 while heuristic routes remain usable.

The `neural` field in an analysis includes:

| Field | Meaning |
|---|---|
| `model` | catalog identity, verified checkpoint metadata, architecture, actual parameter count |
| `input` | full encoded values, tensor shape, semantic input views, root-player perspective and axes |
| `outputs` | all 209 action indices, wire actions, labels, raw values, legality and masked values |
| `layers` | leaf-module names/types, output shapes, activation statistics, optional weight shapes/statistics/bias |
| `raw_best` | unmasked argmax, including whether it is legal |
| `legal_margin` | highest minus second-highest legal Q, or null if fewer than two legal actions |
| `illegal_above_selected` | count of illegal outputs strictly higher than legal Top1 |

DQN candidate `order` equals `action_index`: pawn 0–80, horizontal wall 81–144,
vertical wall 145–208, using the canonical action encoder. Illegal `masked_value`
is the JSON sentinel `"-inf"`. No softmax or probability calibration is introduced.
Distance deltas remain rule diagnostics and do not enter Q selection.
Terminal positions have no candidate or selected action but retain raw network diagnostics.

`POST /api/branch` for DQN validates a legal action with Rust, returns its unchanged
root-position Q and after snapshot, and a one-step preview rather than a search PV.
`neural.sensitivities` contains input gradients of `Q(a)` and, if another legal action
exists, `Q(a)-Q(b)` where b is the highest-Q other legal action. Each includes the
objective value, reference index/label, complete gradient tensor and semantic views.
These are local derivatives, not causal attributions, attention weights, or legal
perturbation results. Parameters and `.grad` buffers remain unchanged; hooks are removed
in a finally block. All gradients are required to be finite.

`POST /api/neural/tensor` accepts `kind=activation|weight` and an observed leaf-layer
name. Activation slices select a spatial output channel; vector activations use offset.
Conv weights select output `channel` and `input_channel`; Linear weights select output
`channel` then input-vector `offset`. Indices default to zero and are strictly checked.
Responses include model hash, position hash, kind/layer/indices, source shape, layout,
values, full selected-channel/row statistics, and `window={offset,total,shown}`.
Vector windows contain at most 256 elements in display rows of 16. Spatial views keep
encoder `[x,y]` indexing; kernels use matrix row/column indices. Padded final display
rows omit nonexistent elements. Statistics and 20-bin histograms cover the full
selected channel or row, not only its window. The UI labels scope and color scale.

The original CNN input is `[7,9,9]`, with current pawn, opponent pawn, horizontal
and vertical wall heads, normalized stocks broadcast to the board, and current goal row.
Legacy MLP input is `[292]`: two 81-value pawn blocks, two 64-value wall-head blocks,
and two normalized stock scalars. Both use original engine coordinates without flipping;
the MLP has no goal-direction feature. Layer activations are spatial only where the
network retains spatial axes; fully connected vectors are not board maps.

`stats` reports min, max, population mean/std and zero fraction. Computations use
float64 for summaries so large finite model outputs do not overflow their variance.
Zero activation in one position does not establish a permanently inactive unit.
Q values across different reward definitions or model scales are not directly comparable.

## Changelog

Version 2 (2026-10-20)
- Added catalog-backed CNN/MLP analysis, raw/masked Q, activations, weights and input gradients.
- Specified neural identity, resource limits, tensor coordinates and interpretation boundaries.

Version 1 (2026-10-20)
- Defined standalone analysis API, validated positions, faithful score/bound semantics,
  exact branch inspection, trace limits, and exported-case provenance.


## Policy/value search inspection

CNN checkpoints may declare `cnn_value_head=true` and an outcome-calibrated or trained `value_head_status`. Neural analysis then includes a bounded current-player `neural.value` and captures the raw linear `value_head` layer before tanh. `POST /api/neural/search` accepts a model ID, position, 1–200 simulations and positive c_puct up to 10. It returns the same PUCT action used by Arena, normalized visits, masked priors, parent-view edge Q, terminal root values and measured search counters. It never mutates weights or enables root noise. No calibrated head means HTTP 422. Search gradients are not exposed; candidate gradient inspection continues to differentiate the policy logits. See [the benchmark contract](../training/alphazero-lite.md).
