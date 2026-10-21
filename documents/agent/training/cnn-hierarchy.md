# Two-stage CNN decisions

Set `cnn_hierarchical: true` to choose an action in two stages:

1. Compare two category Q values: **pawn** versus **wall**.
2. Compare action Q values only within the selected category's legal actions.

The CNN shares its convolutional encoder and 256-unit hidden layer. The existing
209-action head is retained and a two-output `action_type_head` is added: 514 extra
parameters, for 1,439,059 total. Horizontal and vertical walls share one category.
Both heads output unnormalized Q estimates, not probabilities. An action with the
largest raw Q across all 209 outputs can be rejected by the category decision.

## Legality and exploration

The category mask is computed from the engine's legal action mask. If no wall can be
placed, the wall category is unavailable even if its raw category Q is larger.
Within-category selection still uses the complete engine mask. Ties select the lower
category index (pawn) and then the lower action index, consistently across consumers.

With epsilon exploration, choose uniformly among **available categories**, then
uniformly among legal actions of that category. Initially this gives about 50% pawn
and 50% wall exploration, compared with 128/131 = 97.7% wall exploration under a flat
uniform legal-action sampler. This addresses exploration imbalance. Flat DQN argmax
does not itself vote by action count; changing the hierarchy is not a guarantee of
better decisions or wall placement.

## Training objective and comparison

For a completed learner transition, both the executed action Q and its category Q
are trained toward the actual reward plus the Double-DQN bootstrap. The online
network selects the next action using both stages; the target network evaluates
that selected action. Category and action TD losses are averaged so introducing a
second head does not simply double the overall TD-loss weight.

When demonstration supervision is enabled, the same actual Monte Carlo return
supervises both executed heads. Expert supervision compares the teacher's category
against the other available category, then compares the expert action against other
legal actions **in the teacher's category**. These two margins are also averaged.
The category estimates depend on the evolving lower-level policy; they are learned
approximations, not a guaranteed maximum over perfectly known action values.

The October 20 comparison uses the existing 69,923 training transitions and 6,504
validation transitions from 1,500 completed demonstration games. Flat and hierarchical
models start with identical shared weights, use minibatch seed 777, and train for
6,000 updates with diagnostics at 2,000/4,000/6,000. The hierarchy changes both action
selection and the required category/conditional losses; this is an architecture
comparison, not an ablation claiming to isolate every loss term.

The separate online comparison starts each architecture from its 2,000-update
checkpoint, retains the same demonstration dataset, and trains for 1,000 Greedy
episodes at learning rate 1e-5, epsilon .20→.05. Each 500-episode checkpoint plays
50 Greedy games. Episode budgets match; actual transition/update counts can differ
because the policies produce different game lengths.

Configs:

```bash
.venv/bin/python scripts/train_cnn_repair.py \
  --config agent_system/training/configs/cnn_hierarchy/01_flat_control.json
.venv/bin/python scripts/train_cnn_repair.py \
  --config agent_system/training/configs/cnn_hierarchy/02_two_stage.json
.venv/bin/python scripts/train_cnn_repair.py \
  --config agent_system/training/configs/cnn_hierarchy_online/01_flat_online.json
.venv/bin/python scripts/train_cnn_repair.py \
  --config agent_system/training/configs/cnn_hierarchy_online/02_two_stage_online.json
```

Use new external output paths when rerunning these configs. The [repair workflow](cnn-repair.md)
documents provenance, per-checkpoint diagnosis and immutable registration.

## Decision Lab contract

Checkpoint `model_config.cnn_hierarchical` controls loading; missing means flat.
Runtime, Arena, `DQNPolicy`, training evaluation and Decision Lab use the same routing
helper. Diagnostics replay actual moves and assert that both-stage selection matches.

For hierarchical models, `neural.hierarchy` exposes both raw category Qs, availability,
legal-action counts, the best action in each category, and the selected category.
All legal candidates remain inspectable. `route_eligible` distinguishes actions in
the selected category from legal actions excluded by stage one. `masked_value` is
the engine-only mask; `routed_value` additionally excludes the unselected category.
Neither operation changes the recorded raw action Q.

The page shows the category decision above the conditional action decision. It
sorts selected-category candidates first, marks other candidates as `类别未选`, and
shows the post-routing mask in the full-output table. Gradients are available for
action Q, within-category action margin, category Q and category margin. Category
head activations and weights use the existing tensor viewer. These gradients
describe local sensitivity, not causal explanations.

## Validation

Every new checkpoint plays the fixed 50-game development book. A candidate is locked
within each architecture using score, tactical finish tests, then smaller legal Q
magnitude. Each selected model subsequently plays 200 games against Greedy on the
same 100 fresh, unique opening states with seats swapped. Prior development and
holdout opening states are excluded. Standard-start controls are reported separately.
One training seed is used; confidence bounds describe the opening sample and do not
measure variability across independent training runs.

For the offline comparison, use:

```bash
.venv/bin/python scripts/evaluate_cnn_repair.py \
  /home/jihua/quoridor-data/experiments/cnn-hierarchy-20261020 \
  --candidate-prefix cnn_hierarchical --baseline-prefix cnn_flat_control \
  --pairs 100 --seed 2026102102 \
  --exclude-book /home/jihua/quoridor-data/experiments/cnn-repair-20261020/holdout/selection.json
```

The diagnostic report separates `pawn_selected`, `wall_selected`, and
`missed_win_wrong_type`. A high category accuracy alone is insufficient: a model
that nearly always walks can still miss strategically necessary wall placements.

## Results: October 20, 2026

Training ran on training's RTX 3080 Ti. Every saved checkpoint completed its 50-game
Greedy diagnostic before training continued. Development scores are wins out of 50;
all games below ended at a goal, with no draws or illegal actions.

| Experiment | Checkpoint | Flat CNN | Two-stage CNN |
|---|---:|---:|---:|
| Offline demonstrations | 2,000 updates | 18 | 18 |
| Offline demonstrations | 4,000 updates | 19 | 18 |
| Offline demonstrations | 6,000 updates | 19 | 18 |
| Online with demonstration retention | 500 episodes | 19 | 18 |
| Online with demonstration retention | 1,000 episodes | 16 | 11 |

The frozen selection rule chose flat update 4,000 and hierarchical update 2,000 for
the offline holdout, and the 500-episode checkpoint for both online branches.

| Independent test | Flat CNN wins / games | Two-stage CNN wins / games | Two-stage minus flat |
|---|---:|---:|---:|
| Offline holdout, seed 2026102102 | 76 / 200 (38%) | 76 / 200 (38%) | 0 percentage points |
| Online holdout, seed 2026102103 | 75 / 200 (37.5%) | 74 / 200 (37%) | −0.5 percentage points |

Each row uses its own set of 100 unique openings, paired by seat. Conservative 95%
Hoeffding intervals for the paired difference are [−27.16, +27.16] and
[−27.66, +26.66] percentage points. These results show **no demonstrated strength
gain from this hierarchy**. They do not establish equivalence, and the two rows
must not be treated as a comparison on identical openings. All four selected models
passed 18/18 immediate-finish probes and missed no available immediate win in their
holdout games. Each won one of the two separate standard-start controls.

The hierarchy materially changed behavior: in the offline holdout, the flat CNN
placed 301 walls in 1,888 decisions (15.9%), versus 10 in 1,667 (0.6%) for the two-stage
CNN. After online fine-tuning, these counts were 318/1,763 (18.0%) and 41/1,669 (2.5%).
Less wall placement is an observation, not a measure of better play.

### Category diagnostic

The demonstration teacher labels a wall in 11,979/69,923 training states (17.1%)
and 1,236/6,504 validation states (19.0%). On the fixed validation split, the offline
two-stage CNN chose a wall on only 425 states. It selected the wall category on
259/1,236 teacher-wall states (21.0% recall), compared with 410/1,236 (33.2%) for the
flat control. Online two-stage fine-tuning raised that recall to 383/1,236 (31.0%).
Agreement with Greedy is only a teacher diagnostic, not strategic ground truth.

A real offline holdout loss (opening 0, P1, replay ply 16) makes both failure levels
inspectable. P1 at f5 and P2 at f2, with the recorded five walls, gives category Qs
`pawn=-0.793226`, `wall=-0.989660`. The gate selects pawn and the action head chooses
e5, after which P2 wins with f1. Legal walls e1h and f1h prevent that immediate finish,
but are excluded by the gate. Even if the gate were forced to wall, this checkpoint's
highest wall Q is d2v, which does not block f1. Thus this example needs improvement
in both category choice and specific wall choice. A delaying wall does not imply
that the position is ultimately winnable.

Next experiments should add targeted defensive and useful-wall positions with
search-checked labels, report category precision/recall and conditional wall quality
separately, and compare category-balanced sampling against the present objectives.
Category Q currently regresses returns of executed walls, many chosen by exploration;
it can learn the average consequence of those sampled walls instead of the value of
the best wall the lower policy could choose. A separate action/category-target ablation
is needed to test that mechanism. Keep flat and hierarchical controls, multiple training
seeds, and fresh paired evaluations; do not infer that a larger model alone fixes this.

## Records and verification

The ten hierarchical-comparison checkpoints (six offline and four online) are
registered as `quoridor-cnn-dqn` versions 18–27 in
[Quoridor / CNN hierarchy](http://192.168.68.88:8742/#/experiments/4).
[Decision Lab](http://192.168.68.88:8744/) can inspect both stages for any
`cnn_hierarchical_*` checkpoint. Experiment files live outside Git under
`~/quoridor-data/experiments/cnn-hierarchy-20261020/` and
`~/quoridor-data/experiments/cnn-hierarchy-online-20261020/`; category probes and
the recorded case are in the former's `category-probe/` directory.

Validation: 801 training/analysis/runtime Python tests passed, plus six Decision Lab
JavaScript tests. Dedicated hierarchy tests cover legal-category routing,
balanced category exploration, checkpoint round trips, runtime/analysis agreement,
category gradients and DDQN bootstrap targets. Browser checks also exercise
the category cards, excluded candidates and category sensitivity view.
