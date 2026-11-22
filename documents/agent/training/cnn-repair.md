# CNN repair experiments and checkpoint diagnostics

This workflow trains CNNs on **training** and evaluates every saved checkpoint against
the exact runtime Greedy implementation. It uses the existing CNN observation and
209-action mapping. MLP training and the online agent roster are outside this experiment.

## Baseline and experiment scope

The frozen baseline is `cnn_observation_2k`, SHA256
`d823125353ffb5e902ddbfba8d6086b88c87354375839682783fd38c86e0abcf`.
The network still has convolution widths 32/64/64 and a 256-unit dense head,
1,438,545 parameters. New experiments use LeakyReLU(0.01); old checkpoints without
activation metadata continue to load as ReLU, with identical weights and behavior.

The October 20 experiments use one training seed, 42. They are a first repair round,
not a multiple-seed estimate of training reliability or a claim of general Quoridor
strength. A configuration may change several things relative to the historical run;
do not attribute its entire gain to one architectural or optimizer change.

## Training changes

- At a sampling limit, finish an outstanding opponent reply before storing the learner
  transition. This can exceed the requested ply cap by one. Nonterminal `next_obs` and
  `next_mask` always belong to the learner. A cutoff keeps bootstrap enabled; an actual
  win disables it. Every incomplete episode is counted, regardless of seat/parity.
- `cnn_activation` is saved with `dense_width` in the checkpoint. Defaults remain
  backward compatible. `warm_start_checkpoint` initializes weights only: optimizer,
  replay, exploration counters and episode counts start fresh.
- `greedy` is a training opponent using the same scorer and tie breaking as runtime.
- `train(..., checkpoint_callback=...)` blocks after each save until diagnostics finish.
  A diagnostic error aborts the experiment instead of silently continuing training.
- Optional `demonstration_dataset` retains the **training split** of a fixed demonstration
  dataset during online updates. It adds Monte Carlo regression and a legal-action
  expert margin; `demonstration_weight` controls their contribution. Default training
  does not use this extra objective. `metrics.jsonl` includes the auxiliary loss.

## Demonstration experiment

`agent_system/training/dqn/demonstrations.py` generates new, randomized Greedy games.
Random opening prefixes and occasional pawn/wall exploration diversify the states.
The teacher prioritizes a directly winning legal move; otherwise it uses Greedy's
action score and tie breaking. These are **action labels**, never numerical Q targets.

The dataset stores actual two-ply learner transitions and discounted returns from
completed games. Incomplete games are excluded and counted, not assigned invented
terminal rewards. Validation uses held-out episodes and removes states whose geometry,
including reflections and reversed goal direction, occurs in training.

The offline objective combines one-step Double DQN, regression to actual discounted
returns, and a 0.1 expert-action margin over legal actions. This is inspired by
[learning from demonstrations](https://arxiv.org/abs/1704.03732), but is not a full DQfD
implementation: it uses uniform sampling and full-game returns, without prioritized
replay or the paper's n-step implementation. Inference always uses the ordinary masked
CNN argmax. It does not call Greedy, search, or apply an immediate-win override.

## Running on training

From `/home/jihua/workspace/boardgame-ai-arena`:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/train_cnn_repair.py \
  --config agent_system/training/configs/cnn_repair/01_warmstart.json
```

The numbered configs record the actual experiments:

| Config | Initialization and learning |
|---|---|
| `01_warmstart.json` | Historical 2k weights, LeakyReLU, conservative DDQN, 300 episodes |
| `02_fresh_ddqn.json` | Fresh CNN, same size, 2,000 Greedy episodes, diagnose every 500 |
| `03_demonstrations.json` | 1,500 new demonstration games, 6,000 gradient updates, diagnose every 2,000 |
| `04_demo_finetune.json` | Initialize from demonstration update 2,000; 1,000 online episodes |
| `05_demo_retention.json` | Same initialization, seed and budget as 04; retain demonstration loss |

For a rerun, copy a config and change `output` and `training.checkpoint_dir` to a new
external directory. The runner rejects existing outputs and repository-local artifact
directories. It writes source snapshots, configuration, GPU/Torch information, opening
book, checkpoints, metrics, replays and diagnostics outside Git. Warm starts do not
resume an interrupted run exactly. External input hashes are recorded by the current
runner; earlier runs retain their config and immutable parent-checkpoint references.

## Diagnostic protocol

Each checkpoint plays **50 games against Greedy**, using 25 fixed opening instances,
swapping seats once per opening. The historical development book has 24 unique states.
No tactical rule overrides either CNN. Illegal moves/errors fail evaluation; repeated
states in deterministic play are draws. The report includes:

- wins/draws/losses, separate seat outcomes, termination reasons and full action traces;
- immediate wins available and missed in actual games;
- 18 fixed finish-in-one probes, explicitly separate from the win-rate estimate;
- last-hidden-layer all-zero counts, output-equals-bias counts, active units and legal Q range;
- exact checkpoint and opening hashes; paired-score confidence bounds.

The diagnostic pass replays every CNN decision through Decision Lab's shared `Runner`
and asserts that its masked argmax matches the move actually played. This guards
against debugging a different model or inference path than the Arena evaluated.

`scripts/evaluate_cnn_repair.py SUITE --pairs 100` locks a candidate using development
score, tactical completion, then smaller maximum legal Q magnitude. It writes that
selection **before** playing new games. The fresh book excludes development opening
states and contains 100 unique openings. Both baseline and candidate play 200 Greedy
games each, plus two separate standard-start controls. Confidence bounds use opening
pairs as independent units; the two seat-swapped games are not counted as independent.
Do not tune on this holdout and then call it independent confirmation again.

## Inspect and register

`model-management/publish_cnn_repair.py` requires a completed result and a matching
diagnosis for every checkpoint, copies weights into the immutable SHA store, and
creates a **new** catalog including failed candidates. `log_cnn_repair.py` runs inside
the existing MLflow container and registers configurations, source, weights and match
diagnostics under **Quoridor / CNN repair**. It does not move the production champion.

```bash
.venv/bin/python model-management/publish_cnn_repair.py SUITE \
  --catalog /home/jihua/quoridor-data/catalog.json \
  --output /home/jihua/quoridor-data/catalog-NEW.json
docker exec -i model-management-mlflow-1 python - \
  --catalog /quoridor-data/catalog-NEW.json --output /data/registry-NEW.json \
  < model-management/log_cnn_repair.py
```

Point Decision Lab's `LAB_MODEL_CATALOG` at the new catalog and restart that component
to inspect any candidate's values, activations, weights and input sensitivities.
Registry metadata is joined by SHA from the adjacent active `registry.json`.
Training results, datasets and diagnostic evidence stay under `~/quoridor-data/`;
they are not source files to commit.

## Results: October 20, 2026

| Experiment | Checkpoints | Development Greedy wins / 50 |
|---|---|---|
| Historical 2k weight warm start | 300 episodes | 0 |
| Fresh conservative DDQN | 500 / 1,000 / 1,500 / 2,000 episodes | 0 / 0 / 0 / 0 |
| Demonstration-assisted CNN | 2,000 / 4,000 / 6,000 updates | 18 / 18 / 18 |
| Ordinary online continuation | 500 / 1,000 episodes | 1 / 3 |
| Online continuation retaining demonstrations | 500 / 1,000 episodes | 17 / 15 |

The selected checkpoint is `cnn_repair_demo_update002000`, SHA256
`e93e50402ebf04ef3d11da6442bb089e231cec37307177de60ddf2f2e2b1c9ce`,
registered as `quoridor-cnn-dqn` version 11. Its 2,000 count denotes gradient updates
over demonstrations from 1,500 games, not 2,000 newly played training episodes.

On the same 100 fresh, unique opening states (seats swapped, seed 2026102101), the
historical 2k CNN won **6/200 (3%)** and the selected repaired CNN won **67/200 (33.5%)**.
The paired gain is 30.5 percentage points; the conservative 95% Hoeffding interval
is [+3.34, +57.66] points. This is improvement against Greedy for this opening
distribution and one training seed; the repaired CNN still loses most games.

Across those held-out games the historical model had an all-zero final hidden layer
on 1,112/2,014 decisions and missed 15/21 available immediate wins. The repaired model
had no all-zero hidden states in 1,741 decisions and missed 0/67 immediate wins.
It also passed all 18 fixed finish-in-one probes. Standard-start controls were 0/2
for the historical model and 1/2 for the repaired one, separate from the 200-game test.

Changing activations and stabilizing DDQN did not by themselves produce strength.
The successful experiment also introduced demonstrations and actual-return targets,
so the gain cannot be attributed solely to LeakyReLU or the transition correction.
Ordinary online continuation lost most of the learned playing ability; retaining
demonstration supervision limited this regression but did not surpass the selected
offline model. Further decomposition is covered by [two-stage CNN decisions](cnn-hierarchy.md).

All twelve new checkpoints, including failures, are in
[Quoridor / CNN repair](http://192.168.68.88:8742/#/experiments/3), with configs,
source snapshots, input hashes where available, 50-game diagnostics and replays.
Independent evaluation files are under
`~/quoridor-data/experiments/cnn-repair-20261020/holdout/`.

## Direct repair from the original 2k weights

The subsequent `cnn-2k-repair-20261020` suite returns to the ordinary, flat CNN.
All convolution and dense weights are loaded strictly from the immutable
`cnn_observation_2k` checkpoint. Its 32/64/64 convolution widths, 256-unit hidden layer,
input representation and 209-action output remain the same; `cnn_hierarchical` is
false. ReLU is replaced with LeakyReLU(0.01), and the optimizer and update counter
start fresh. This is weight fine-tuning, not an exact optimizer-state resume.

The demonstration training entry point now honors `warm_start_checkpoint`, just as
the online trainer does. Incompatible observation versions and parameter shapes
are rejected. `initialization.json` and each checkpoint's `eval_summary.initialization`
record the parent SHA256, checkpoint ID, episode/update counts and activation change.
An integration test uses a zero learning rate to verify that every saved parameter
really equals its parent, while metadata and the new update count remain correct.

Configs live in `agent_system/training/configs/cnn_2k_repair/`. All three runs use the
same 69,923-transition demonstration training split, the same actual-return/DDQN/
expert-margin objective, AdamW at 1e-4 and a batch size of 256. Each runs 6,000 optimizer
updates, saving and immediately diagnosing updates 2,000, 4,000 and 6,000. Training
seeds 42/43/44 use minibatch seeds 742/743/744. These are repetitions of optimization
from the **same original weights and data**, not independently initialized models
or independent datasets. Each of the nine checkpoints plays 50 Greedy games.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/train_cnn_repair.py \
  --config agent_system/training/configs/cnn_2k_repair/01_seed42.json
```

The fixed development rule selects the primary checkpoint and one checkpoint per
seed before independent games begin. The final evaluation uses 400 unique opening
states, swapping seats for **800 games per model**. It evaluates the original 2k,
the three selected repairs, and the previous fresh-training repair on the same book.
It excludes the development book, all three earlier holdout books, and every opening
geometry/reflection present in either split of the demonstration dataset. Later
states in evaluated games can still overlap training trajectories. The resulting
distribution is conditional on these exclusions and differs from earlier tests.

`evaluate_cnn_repair.py` supports repeatable `--replicate-prefix`,
`--reference-diagnosis`, and `--exclude-dataset` arguments for this protocol.
`holdout/selection.json` freezes identities, opening states, input hashes and evaluator
source before matches. Each model is evaluated once even when it is both the primary
and a seed representative. Primary confidence bounds use paired Hoeffding; all three
seed comparisons additionally get simultaneous 95% Bonferroni-Hoeffding bounds.
Opening-pair uncertainty does not quantify uncertainty across training datasets.

The plan and its revision are stored outside Git with the suite. Before any holdout
games, the sample count was increased from 200 to 400 pairs and demonstration-state
exclusion was added. The acceptance criterion is a positive 95% lower bound on the
primary repair's improvement over 2k, with no illegal moves or evaluation errors.
All model weights, replay traces, datasets, source snapshots and audit output stay
under `~/quoridor-data/`; only reusable source, configs and documentation belong in Git.

### Direct-repair results

Development wins out of 50 for updates 2,000 / 4,000 / 6,000 were:

| Sampling run | Update 2,000 | Update 4,000 | Update 6,000 | Frozen representative |
|---|---:|---:|---:|---|
| seed 42 / batches 742 | 8 | 19 | 19 | update 4,000 |
| seed 43 / batches 743 | 11 | 18 | 19 | update 6,000 |
| seed 44 / batches 744 | 3 | 19 | 18 | update 4,000 |

The primary was locked as `cnn_2k_repair_s42_update004000`, not changed after seeing
holdout results. All five models below played the **same** 400 fresh opening pairs;
68,714 demonstration geometry groups and the previous evaluation books were excluded.
All 4,000 holdout games ended at a goal with no illegal actions/errors/draws.

| Model | Wins / 800 | Win rate | P1 wins / 400 | P2 wins / 400 |
|---|---:|---:|---:|---:|
| Original CNN 2k | 32 | 4.00% | 15 | 17 |
| **2k repair seed 42, update 4,000** | **254** | **31.75%** | 174 | 80 |
| 2k repair seed 43, update 6,000 | 256 | 32.00% | 175 | 81 |
| 2k repair seed 44, update 4,000 | 251 | 31.375% | 173 | 78 |
| Previous fresh-training repair, update 2,000 | 256 | 32.00% | 173 | 83 |

The primary gain over original 2k is **+27.75 percentage points**, with a conservative
paired 95% interval **[+14.17, +41.33]** points. All three repetitions also pass the
simultaneous 95% check: their gain intervals are [+12.28, +43.22], [+12.53, +43.47],
and [+11.90, +42.85] points. Thus this repair exceeds the requested 2k benchmark
under this protocol; it has not demonstrated an improvement over the previous
fresh-training repair, and still loses most games against Greedy.

The original 2k had an all-zero last hidden layer on 4,489/8,189 decisions and missed
99/131 available immediate wins in these games. The primary repair had zero all-zero
hidden states in 7,269 decisions and missed 0/254 immediate wins. LeakyReLU changes
the zero-activation metric by construction; the win and decision improvements are
the relevant behavioral evidence, and cannot be attributed to that activation change
alone. The original model lost both standard-start controls; each repair won one.

Decision Lab's actual HTTP analysis/gradient endpoints were also checked. In the old
failure case (P1 g8, P2 d2, horizontal wall a6), the original model chooses h8 with
zero input gradient and an all-zero hidden layer. The repaired model chooses the
winning g9 and has a nonzero input gradient. This fixes that example through neural
inference alone.

There is still a known regression: the primary passes **17/18** finish-in-one probes.
With P1 e6, P2 c2, no walls placed and P2 to move, it chooses d2 instead of the winning
c1, which the original 2k chose. This case was not used to retune the accepted model.
Input sensitivity is present, so this remaining error cannot be explained by the
old all-zero hidden-layer failure. Keep it as a regression case for the next training
iteration. The other two seed representatives pass 14/18 and 17/18; their absence of
missed finishes in the sampled games does not imply all finish positions are solved.

The primary is `quoridor-cnn-dqn` **version 29**, SHA256
`941b72b2990f09bcb7562288204b9fba5f0890fac9cc1b544e381fbf04619384`:
[model run](http://192.168.68.88:8742/#/experiments/3/runs/68f649e69f904be6a2276ea29daee57f),
[independent evaluation](http://192.168.68.88:8742/#/experiments/3/runs/59e17f08240549129ed66cbc5a234ae8).
All nine checkpoints are registered as versions 28–36 and selectable in Decision Lab.
The suite's `recommended.json` records acceptance and identity; `decision-probes/`
contains the old failure and the remaining regression, with analysis JSON and input
gradients. No production champion alias was moved.
