# Quoridor Arena

The Arena is an automated evaluation system that runs matches between
Quoridor agents and aggregates results into a win-rate matrix.

## Structure

```
arena/
  agent_defs/          # YAML agent definitions (one file per agent)
  agents/              # Core abstractions (Scorer, Policy, Agent, AgentInstance)
  experiments/         # YAML experiment definitions
  runner.py            # Game execution and match orchestration
  experiment_loader.py # Parse experiment YAML
  experiment_runner.py # Execute experiments
  aggregator.py        # Win-rate matrix computation
  db.py                # SQLite persistence
  models.py            # Data models (GameRecord, MatchResult, etc.)
  tests/               # Arena tests
scripts/
  run_arena.py         # CLI — round-robin tournament
  run_experiment.py    # CLI — YAML-defined experiments
```

## Agent Definition YAML

Each file in `arena/agent_defs/` defines one agent.

**The file name (without `.yaml`) MUST equal the `id` field.**

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique agent identifier |
| `algo.type` | Yes | Algorithm family: `random`, `greedy`, `minimax` |
| `algo.params` | No | Opaque parameters passed to the scorer (e.g., `depth`) |
| `policy.type` | Yes | Action selection policy: `top_k` |
| `policy.k` | No | Number of top candidates to sample from (default: 1 = argmax) |

### Example

```yaml
# arena/agent_defs/minimax_d3.yaml
id: minimax_d3

algo:
  type: minimax
  params:
    depth: 3

policy:
  type: top_k
  k: 1
```

### Architecture

Each agent follows the pipeline:

```
Scorer (evaluate all legal actions)
  → Policy (select one action from scored candidates)
    → AgentInstance (runtime wrapper with per-game RNG)
```

- **Scorer** enumerates all legal actions and assigns scores.
  It does NOT choose the final action.
- **Policy** selects one action from scored candidates using the
  instance-owned RNG. `top_k` sorts by score descending, takes the
  top-k, and uniformly samples one (`k=1` = deterministic argmax).
- **AgentInstance** is created per-game with an explicit seed,
  ensuring deterministic and reproducible behavior.

## Experiment Definition YAML

Each file in `arena/experiments/` defines one experiment — a set of
explicit matchups between agents.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Experiment identifier |
| `matches` | Yes | List of matchup entries |
| `matches[].agent_1` | Yes | Agent ID (must match a file in `agent_defs/`) |
| `matches[].agent_2` | Yes | Agent ID (must match a file in `agent_defs/`) |
| `matches[].params` | No | Opaque params; currently supports `num_games` (default: 50) |

### Example

```yaml
# arena/experiments/basic.yaml
id: basic_matchups

matches:
  - agent_1: random
    agent_2: greedy
    params:
      num_games: 50

  - agent_1: greedy
    agent_2: minimax_d3
    params:
      num_games: 50
```

Experiments list **explicit pairs only** — there is no automatic
round-robin expansion. Each match entry specifies exactly which two
agents play and how many games.

## How to Run

### Round-Robin Tournament (all agents, all pairs)

```bash
python scripts/run_arena.py
python scripts/run_arena.py --num-games 20 --seed 0 --verbose
```

### YAML Experiment (explicit matchups)

```bash
python scripts/run_experiment.py arena/experiments/basic.yaml
python scripts/run_experiment.py arena/experiments/basic.yaml --verbose
python scripts/run_experiment.py arena/experiments/basic.yaml --very-verbose --seed 0
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--verbose` | Show per-game summary (winner, steps, time) |
| `--very-verbose` | Show per-step actions |
| `--seed N` | Base seed for deterministic execution (default: 42) |
| `--db PATH` | SQLite database path (default: `arena_results.db`) |
| `--agent-dir DIR` | Agent definitions directory (default: `arena/agent_defs/`) |

Verbose flags are **CLI-only** and never appear in YAML.

## Design Principles

- **Config-driven agents** — all agents are defined in YAML; no direct
  class instantiation in the runner. There is no legacy agent path.
- **Canonical identity** — each agent has a deterministic `canonical_id`
  derived from `algo.type + algo.params + policy.type + policy.params`
  (e.g., `minimax/depth=3+top_k/k=1`). This is independent of the
  human-readable `id` and can be used for deduplication and comparison.
- **Explicit experiments** — experiment YAML lists exact matchup pairs;
  no implicit expansion.
- **Separation of scoring and policy** — scorers evaluate actions,
  policies select from scored candidates. This makes agents composable.
- **Deterministic execution** — same agent + same seed = identical
  behavior. RNG is instance-owned; no global random state.
- **Reuse over reimplementation** — scorers wrap existing algorithm
  code (greedy, minimax) without duplicating logic.
