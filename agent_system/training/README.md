# Training

This folder contains the DQN training workflow for the Quoridor agent.

## Prerequisites

Run these once from the repository root:

```bash
# Build the Rust engine Python bindings
cd engine && maturin develop --release && cd ..
```

You also need `uv` available so `uv run` can start the training script.

## Quick Start

Run the short smoke experiment via the helper script:

```bash
bash agent_system/training/quick_train.sh
```

Or run it directly with the YAML config:

```bash
PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u scripts/train_dqn.py \
  --config agent_system/training/configs/dqn/stabilization_smoke_001.yaml
```

Both write:

- `agent_system/training/artifacts/dqn/stabilization_smoke_001/train.log`
- `agent_system/training/artifacts/dqn/stabilization_smoke_001/checkpoints/`

## Training Artifact Contract

Every training run produces the following standard artifacts under its run directory:

```
<run_dir>/
  config.yaml          # resolved config — all params actually used by this run
  train.log            # human-readable training log
  metrics.jsonl        # structured per-interval training metrics (one JSON object per line)
  eval_results.jsonl   # structured evaluation results (one JSON object per line)
  checkpoints/         # model checkpoint files (ep*_step*.pt)
```

**`config.yaml`** is written at training start and reflects the final resolved config
after applying `CLI explicit value > YAML config value > argparse default`.

**`metrics.jsonl`** is the canonical source for training scalars (loss, reward, epsilon,
buffer size, etc.). It is appended at every `log_interval`.

**`eval_results.jsonl`** is the canonical source for evaluation results (win rate, games,
wins/losses/draws). It is appended at every `eval_interval` and at the final evaluation.

**`train.log`** is human-readable and retained for debugging. It is **not** the primary
metrics API — do not depend on its format for programmatic access.

## Opponent Configuration

DQN training supports the following opponent modes, set via the `opponent` key in YAML (or `--opponent` on the CLI).

### Supported opponent types

| Opponent | YAML/CLI value | Description |
|---|---|---|
| Random legal | `random_legal` | Uniform-random selection from legal moves (default) |
| Pawn random | `pawn_random` | Uniform-random selection from legal **pawn moves only** (no walls). Symmetric in P1/P2 seats — recommended for eval baselines |
| Dummy | `dummy` | Always picks the first legal action — deterministic but **broken as P1** (marches to opponent's goal). Use `pawn_random` for eval instead |
| Minimax | `minimax` | Alpha-beta minimax at a configurable depth |
| Mixed | `mixed` | Per-episode sampling from a weighted combination of the above |

> **Note on `dummy` vs `pawn_random`**: `DummyOpponent` always picks the lowest-ID legal action,
> which causes it to march directly to the corner `(0,0)` in ~11 pawn moves. As P2, `(0,0)` is
> the goal so dummy P2 wins in 22 steps. As P1, `(0,0)` is the *wrong* goal, so dummy P1 can
> never win — it oscillates at the corner indefinitely. This asymmetry makes the reported "50%
> win rate" against dummy meaningless (it is really 0% in the P1 seat and 100% in the P2 seat).
> Use `pawn_random` for evaluation benchmarks.

### Minimax depth

Use `opponent_depth` to select the search depth (valid values: 1, 2, 3):

```yaml
opponent: minimax
opponent_depth: 2   # 1 = greedy, 2 = standard (default), 3 = strong (slower)
```

Or via CLI: `--opponent minimax --opponent-depth 2`

### Mixed opponent

Mixed policy samples one opponent type at the start of each episode according to relative weights. Weights are **normalized internally** and do not need to sum to 1. At least one weight must be positive.

YAML example (all five types):

```yaml
opponent: mixed
opponent_mix_dummy: 0.1          # first-legal (very weak)
opponent_mix_random: 0.3         # random-legal (weak)
opponent_mix_minimax_d1: 0.2     # minimax depth=1 (greedy)
opponent_mix_minimax_d2: 0.3     # minimax depth=2 (standard)
opponent_mix_minimax_d3: 0.1     # minimax depth=3 (strong)
```

CLI equivalent:
```bash
--opponent mixed \
  --opponent-mix-dummy 0.1 \
  --opponent-mix-random 0.3 \
  --opponent-mix-minimax-d1 0.2 \
  --opponent-mix-minimax-d2 0.3 \
  --opponent-mix-minimax-d3 0.1
```

Omitted keys default to 0.0 (i.e., that component is not included in the mix).

### Available opponent config examples

| Config file | Opponent |
|---|---|
| `configs/dqn/stabilization_smoke_001.yaml` | `random_legal` |
| `configs/dqn/smoke_vs_dummy.yaml` | `dummy` |
| `configs/dqn/smoke_vs_minimax_d1.yaml` | `minimax` depth=1 |
| `configs/dqn/smoke_vs_minimax_d2.yaml` | `minimax` depth=2 |
| `configs/dqn/smoke_vs_minimax_d3.yaml` | `minimax` depth=3 |
| `configs/dqn/smoke_vs_mixed_001.yaml` | `mixed` (all five types) |
| `configs/dqn/smoke_schedule_001.yaml` | 3-phase schedule (random → minimax_d1 → dummy) — smoke test |

## Training Opponent Schedule (Curriculum)

The `train_opponent_schedule` key enables **episode-based curriculum scheduling**:
the training opponent changes automatically as training progresses through defined
episode ranges.

`train_opponent_schedule` and `train_opponent` are **mutually exclusive** — using both
in the same YAML config is an error.

### Schedule entry format

Each entry in `train_opponent_schedule` must have:

| Key | Required | Description |
|---|---|---|
| `from_episode` | Yes | First episode (1-indexed, inclusive) for this phase |
| `to_episode` | No | Last episode (inclusive). Omit to mean "until end of training" |
| `opponent` | Yes | Opponent config dict — same format as `train_opponent` |

Episode ranges must not overlap. Episodes not covered by any range keep the last
active opponent (or raise an error if training starts before the first range).

### Single-opponent phase

```yaml
train_opponent_schedule:
  - from_episode: 1
    to_episode: 500
    opponent:
      type: random_legal

  - from_episode: 501
    opponent:          # no to_episode → runs until end of training
      type: minimax
      depth: 1
```

### Mixed-opponent phase

Each phase can itself be a `mixed` opponent (per-episode sampling by weight):

```yaml
train_opponent_schedule:
  - from_episode: 1
    to_episode: 1000
    opponent:
      type: random_legal

  - from_episode: 1001
    to_episode: 2500
    opponent:
      type: mixed
      opponents:
        - name: random_legal
          type: random_legal
          weight: 0.7
        - name: minimax_d1
          type: minimax
          depth: 1
          weight: 0.3

  - from_episode: 2501
    opponent:
      type: minimax
      depth: 2
```

### Metrics

When a schedule is active, each `metrics.jsonl` record includes:

| Field | Description |
|---|---|
| `train_opponent_phase` | Current phase label (e.g. `random_legal`, `minimax(depth=1)`) |
| `train_opponent_name` | Sampled opponent for this interval (same as phase for non-mixed, sampled name for mixed) |

### Evaluation

Evaluation opponents are configured separately via the `evaluation` section and are
**independent** of the training schedule — evaluations run at `eval_interval` against
all configured eval opponents regardless of the current training phase.



Training is configured through YAML files under `agent_system/training/configs/dqn/`.
Pass the config path with `--config`; every key in the file maps directly to the
corresponding CLI argument name (using underscores).

Precedence (highest → lowest):

1. **Explicit CLI argument** — overrides everything
2. **YAML config value** — overrides argparse defaults
3. **Argparse default** — used when neither of the above is present

Example — run the long experiment while overriding the seed:

```bash
PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u scripts/train_dqn.py \
  --config agent_system/training/configs/dqn/long_train_001.yaml \
  --seed 123
```

## Full Experiment

```bash
mkdir -p agent_system/training/artifacts/dqn/long_train_001/checkpoints
PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u scripts/train_dqn.py \
  --config agent_system/training/configs/dqn/long_train_001.yaml \
  2>&1 | tee agent_system/training/artifacts/dqn/long_train_001/train.log
```

## Available Configs

| Config file | Description |
|---|---|
| `configs/dqn/stabilization_smoke_001.yaml` | 300-episode smoke / validation run (random_legal opponent) |
| `configs/dqn/long_train_001.yaml` | 5000-episode standard long run |
| `configs/dqn/smoke_vs_dummy.yaml` | 300-episode smoke vs dummy opponent |
| `configs/dqn/smoke_vs_minimax_d1.yaml` | 300-episode smoke vs minimax depth=1 |
| `configs/dqn/smoke_vs_minimax_d2.yaml` | 300-episode smoke vs minimax depth=2 |
| `configs/dqn/smoke_vs_minimax_d3.yaml` | 300-episode smoke vs minimax depth=3 |
| `configs/dqn/smoke_vs_mixed_001.yaml` | 300-episode smoke vs mixed (all 5 opponent types) |
| `configs/dqn/smoke_schedule_001.yaml` | 30-episode curriculum schedule smoke test (3 phases) |
| `configs/dqn/cnn_mixed_random80_minimaxd1_5k_001.yaml` | 5000-episode CNN DDQN mixed (random 80% / minimax_d1 20%) |
| `configs/dqn/cnn_curriculum_random_to_minimax_5k_001.yaml` | 5000-episode CNN DDQN 4-phase curriculum (random → mixed → minimax_d2) |

## Key Flags

All flags below are also valid YAML keys (use underscores in YAML, e.g. `epsilon_decay_steps`).
`scripts/train_dqn.py --help` shows the full list.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Path to YAML config file |
| `--algorithm` | `dqn` | `dqn` or `double_dqn` |
| `--model-arch` | `mlp` | `mlp` (MLP Q-network) or `cnn` (CNN Q-network) |
| `--obs-version` | `v1` | `v1` (raw coords) or `v2` (board-flip normalization for P2) |
| `--opponent` | `random_legal` | `random_legal`, `dummy`, `minimax`, or `mixed` |
| `--opponent-depth` | `2` | Search depth for `minimax` opponent (1, 2, or 3) |
| `--opponent-mix-random` | `0.0` | Weight for random_legal in `mixed` opponent |
| `--opponent-mix-dummy` | `0.0` | Weight for dummy in `mixed` opponent |
| `--opponent-mix-minimax-d1` | `0.0` | Weight for minimax depth=1 in `mixed` opponent |
| `--opponent-mix-minimax-d2` | `0.0` | Weight for minimax depth=2 in `mixed` opponent |
| `--opponent-mix-minimax-d3` | `0.0` | Weight for minimax depth=3 in `mixed` opponent |
| `--reward-mode` | `terminal` | `terminal` (sparse ±1) or `distance_delta` (terminal + distance shaping) |
| `--distance-reward-weight` | `0.01` | Scale for distance shaping (used with `distance_delta`) |
| `--distance-delta-clip` | `2.0` | Symmetric clip bound for distance advantage delta |
| `--hidden-layers` | `256,256` | Comma-separated MLP hidden layer widths (e.g. `512,512,256`); YAML accepts a list |
| `--cnn-channels` | `32,64,64` | Comma-separated CNN conv-layer output channels; YAML accepts a list |

## Observation Versions

- **v1** (`dqn_obs_v1`): 292-element flat vector — pawn one-hots, wall occupancy, remaining walls. Raw board coordinates for both players.
- **v2** (`dqn_obs_v2`): Same 292-element layout as v1 but P2's coordinates are y-flipped so the network always sees the board from the current player's perspective (goal at y=8).
- **cnn** (`dqn_obs_cnn_v1`): [7, 9, 9] spatial tensor — pawn one-hots, wall occupancy, remaining walls (broadcast), and goal-row indicator. Used automatically when `--model-arch cnn`.

## Notes

- Run commands from the repository root unless you use the provided shell script.
- `scripts/train_dqn.py --help` shows all available flags.
- Mixed-opponent weights are relative; they are normalised internally so they do not need to sum to 1.
- For historical runs without `metrics.jsonl`, the dashboard falls back to parsing `train.log`.