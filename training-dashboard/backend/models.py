"""Normalized data models for Training Dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckpointInfo:
    filename: str
    episode: int | None
    step: int | None
    path: str
    modified_at: str | None = None
    has_benchmark: bool = False
    best_win_rate: float | None = None


@dataclass
class RunSummary:
    run_id: str
    algorithm: str
    reward_mode: str | None
    model_arch: str | None
    total_episodes: int | None
    created_at: str | None
    modified_at: str | None
    latest_checkpoint: str | None
    best_checkpoint: str | None
    has_metrics: bool
    has_benchmarks: bool
    has_checkpoints: bool
    has_notes: bool
    has_eval_replays: bool
    best_benchmark_win_rate: float | None
    best_benchmark_opponent: str | None
    latest_benchmark_win_rate: float | None
    available_metrics: list[str]
    available_benchmarks: list[str]
    notes_preview: str | None
    final_avg_loss: float | None
    final_avg_reward: float | None
    final_epsilon: float | None
    total_env_steps: int | None


@dataclass
class RunDetail:
    run_id: str
    algorithm: str
    reward_mode: str | None
    model_arch: str | None
    total_episodes: int | None
    created_at: str | None
    modified_at: str | None
    config: dict[str, Any]
    checkpoints: list[CheckpointInfo]
    metric_files: list[str]
    benchmark_files: list[str]
    notes: str | None
    summary_stats: dict[str, Any]
    has_metrics: bool
    has_benchmarks: bool
    has_notes: bool
    has_eval_replays: bool


@dataclass
class MetricPoint:
    episode: int | None
    step: int | None
    values: dict[str, float | None]


@dataclass
class MetricSeries:
    run_id: str
    points: list[MetricPoint]
    columns: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    run_id: str
    checkpoint: str | None
    checkpoint_episode: int | None
    checkpoint_step: int | None
    opponent: str | None
    opponent_family: str | None
    opponent_depth: int | None
    games: int | None
    wins: int | None
    losses: int | None
    draws: int | None
    win_rate: float | None
    side: str | None
    seed: int | None
    avg_game_length: float | None
    illegal_action_count: int | None
    timeout_count: int | None
    crash_count: int | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisItem:
    level: str   # "info" | "warning" | "positive" | "error"
    message: str


@dataclass
class DiagnosisReport:
    run_id: str
    items: list[DiagnosisItem]


@dataclass
class EvalReplayMeta:
    """Index entry for a saved evaluation replay."""
    episode: int | None
    checkpoint: str | None
    opponent_name: str | None
    opponent_type: str | None
    opponent_depth: int | None
    game_index: int | None
    dqn_player_id: str | None
    winner: str | None
    result_from_dqn_perspective: str | None
    game_length: int | None
    illegal_acts: int | None
    replay_path: str
    created_at: str | None
