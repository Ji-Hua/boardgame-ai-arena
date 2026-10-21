# Development and Training Workflow

## Start every task safely

```bash
git status --short --branch
git log --oneline --decorate -8
git fetch origin --prune
```

Read [AGENTS.md](../AGENTS.md), [PROJECT_STATUS.md](PROJECT_STATUS.md), and the
relevant authoritative `documents/` material. On Training, also inspect the
external output directory; never infer experiment state from a source checkout.

If a worktree is dirty, first classify every path:

- **Commit:** source, tests, configs, reusable scripts, static documentation,
  service definitions.
- **External only:** checkpoints, datasets, replay buffers, metrics, logs,
  generated replays, MLflow artifacts/databases, and per-run source archives.
- **Stop for review:** a path whose role is unclear, a merge conflict, or any
  operation that could discard data.

Never use `reset --hard`, `clean`, `push --force`, or a source-tree copy to
“synchronize” environments.

## Branch and commit protocol

1. Branch from the agreed integration/development baseline:
   `git switch -c feature/<topic> <base>`.
2. Keep one logical concern per commit. Use the repository convention:
   `Feature: ...`, `Fix: ...`, `Test: ...`, `Documents: ...`, or
   `Infrastructure: ...`.
3. Run the smallest relevant tests, record exact commands and outcomes, then
   push the branch normally (`git push -u origin <branch>`).
4. Use a PR to review and merge shared code. Do not force-push or rewrite
   shared history.

When reconciling environments, create an explicit integration branch, fetch
both named source branches, and merge them one at a time. If Git reports a
conflict, stop before resolving it; identify the conflicting files and obtain a
human decision about intended behavior. A conflict-free merge still requires
tests and review.

## Dev/Training synchronization

### Dev to Training

1. Commit and push the Dev branch.
2. On Training, `git fetch origin --prune`.
3. Create a separate integration branch or worktree and merge the pushed branch.
4. Run training-side tests before using that commit for an experiment.

### Training to Dev

1. On Training, commit and push only source/config/test/doc changes.
2. Leave experimental outputs in `/home/jihua/quoridor-data/experiments/<run-id>/`.
3. On Dev, fetch and merge the named Training branch into an integration branch.
4. Run Dev-compatible tests and review the resulting merge before PR promotion.

Do not “fix” a dirty Training source checkout by deleting outputs. The source
commit and external manifest together preserve reproducibility.

## Training provenance and MLflow

Before launching a run, create a unique external output directory and record:

- Git commit and branch (including dirty-file hashes if a dirty snapshot is
  intentionally used).
- Complete resolved config and its SHA-256.
- Dataset path, dataset SHA-256, split definition, and teacher/version.
- Random seeds, model initialization hash, package/Torch/CUDA/GPU versions.
- Checkpoint SHA-256 and exact evaluation-opening book hash.

Use the Training MLflow server from [ENVIRONMENTS.md](ENVIRONMENTS.md). Log
training parameters and metrics to the parent run, retain checkpoints and
metadata as run artifacts, and create child runs for per-agent benchmark
results where the benchmark tool supports them.

To make a model discoverable:

1. Copy it to the immutable hash-addressed model store under
   `/home/jihua/quoridor-data/models/<sha256>.pt`.
2. Add a new immutable catalog entry with the checkpoint hash, code/config/data
   provenance, action/observation versions, and selection rationale.
3. Use `model-management/register_catalog.py` inside the MLflow container to
   create or find the matching registry version. It tags versions by
   `checkpoint_sha256`.
4. Run the predeclared benchmark on fresh openings. Registration is not
   deployment; do not move a `champion` alias or service roster without an
   explicit promotion task.

Given a model SHA, locate it in the Quoridor catalog/registry JSON, then use
the recorded MLflow model version and `run_id` to open its run in MLflow and
retrieve `runs:/<run_id>/checkpoint`. Given an MLflow version, inspect its
`checkpoint_sha256` tag and resolve the immutable checkpoint/catalog entry.

## End-of-task handoff

Every handoff must include:

- Branch and commit(s), plus whether they were pushed.
- Files intentionally left uncommitted and why.
- Tests/validation actually run and their results.
- For experiments: output path, MLflow run ID, config/data/model hashes, and
  whether results are development-only or held-out.
- A concise update to [PROJECT_STATUS.md](PROJECT_STATUS.md) or
  [RESEARCH_LOG.md](RESEARCH_LOG.md) when a shared fact changes.
