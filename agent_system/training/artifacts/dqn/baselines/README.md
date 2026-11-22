# DQN Baseline Artifacts

**Last Updated:** 2026-04-26
**Phase:** 12D — DQN v0 Baseline Consolidation

This directory contains canonical baseline checkpoint copies for the DQN v0 phase. Source checkpoints remain in their original training artifact directories.

---

## DQN v0 Peak Baseline

| Field | Value |
|---|---|
| File | `dqn_v0_peak.pt` |
| Source checkpoint | `agent_system/training/artifacts/dqn/long_train_005_sync2000/checkpoints/ep03500_step1765553.pt` |
| Win rate vs `random_legal` | **17%** |
| Eval games | 100 |
| Training run | `long_train_005_sync2000` |
| Episode | 3500 of 5000 |
| `target_sync_interval` | 2000 |
| `obs_version` | `dqn_obs_v1` |
| Notes | Highest observed vanilla DQN peak win rate; represents the best single evaluation point in Phase 12C |

---

## DQN v0 Stable Baseline

| Field | Value |
|---|---|
| File | `dqn_v0_stable.pt` |
| Source checkpoint | `agent_system/training/artifacts/dqn/long_train_004_sync1000/checkpoints/ep05000_step2678737.pt` |
| Win rate vs `random_legal` | **11%** |
| Eval games | 100 |
| Training run | `long_train_004_sync1000` |
| Episode | 5000 of 5000 (final) |
| `target_sync_interval` | 1000 |
| `obs_version` | `dqn_obs_v1` |
| Notes | Best stable final checkpoint; sustained late improvement (11% at ep5000); very low avg_loss (0.0008); no divergence throughout |

---

## Superseded Baseline

| Field | Value |
|---|---|
| Source checkpoint | `agent_system/training/artifacts/dqn/long_train_002/checkpoints/ep02500_step1277239.pt` |
| Win rate vs `random_legal` | 11% |
| Training run | `long_train_002` |
| Phase | 12B |
| Notes | Previous best before Phase 12C target-sync ablation; superseded — `long_train_002` diverged at ep3200 and collapsed to 0% at ep4500; the Phase 12C baselines are more stable and equally strong or stronger |

---

## Baseline Usage Notes

When running future experiments:

1. **Compare against both baselines** — use both `dqn_v0_peak` (17%) and `dqn_v0_stable` (11%) as the floor to beat. A new method that beats the peak is a clear win; a method that matches stable but beats peak is an improvement in consistency.

2. **Do not compare only against final ep5000** of unstable runs — the Phase 12B run's ep5000 checkpoint has 0% win rate due to divergence. The Phase 12C sync500 ep5000 checkpoint has only 3% win rate. Final episode ≠ best episode.

3. **Report illegal action count** for all comparisons — should remain 0. Any non-zero count indicates a masking regression.

4. **Include `target_sync_interval` and `obs_version`** in all comparison tables — these affect reproducibility.

5. **Report both peak and final checkpoint win rates** for new training runs — not just final.

6. **Win rate at n=100 games** has standard error ≈ ±3% near 10–20%. Use n≥200 for more precise comparisons or note sampling uncertainty.

---

## Checkpoint Format

Checkpoints are saved via `agent_system/training/dqn/checkpoint.py` (`save_checkpoint` / `load_checkpoint`).

Each `.pt` file contains:
- `checkpoint_id`
- `agent_id`
- `training_step` (env steps)
- `episode`
- `model_state_dict` (QNetwork weights)
- `optimizer_state_dict`
- `obs_version` (must be `dqn_obs_v1` for current checkpoints)
- `action_count` (209)
- `model_config` (hidden_size=256, obs_size=292)
- `train_config` snapshot
- `created_at` timestamp
