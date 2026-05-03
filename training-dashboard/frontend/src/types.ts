// API types matching the backend models

export interface CheckpointInfo {
  filename: string
  episode: number | null
  step: number | null
  path: string
  modified_at: string | null
  has_benchmark: boolean
  best_win_rate: number | null
}

export interface RunSummary {
  run_id: string
  algorithm: string
  reward_mode: string | null
  model_arch: string | null
  total_episodes: number | null
  created_at: string | null
  modified_at: string | null
  latest_checkpoint: string | null
  best_checkpoint: string | null
  has_metrics: boolean
  has_benchmarks: boolean
  has_checkpoints: boolean
  has_notes: boolean
  has_eval_replays: boolean
  best_benchmark_win_rate: number | null
  best_benchmark_opponent: string | null
  latest_benchmark_win_rate: number | null
  available_metrics: string[]
  available_benchmarks: string[]
  notes_preview: string | null
  final_avg_loss: number | null
  final_avg_reward: number | null
  final_epsilon: number | null
  total_env_steps: number | null
}

export interface RunDetail {
  run_id: string
  algorithm: string
  reward_mode: string | null
  model_arch: string | null
  total_episodes: number | null
  created_at: string | null
  modified_at: string | null
  config: Record<string, unknown>
  checkpoints: CheckpointInfo[]
  metric_files: string[]
  benchmark_files: string[]
  notes: string | null
  summary_stats: Record<string, unknown>
  has_metrics: boolean
  has_benchmarks: boolean
  has_notes: boolean
  has_eval_replays: boolean
}

export interface MetricPoint {
  episode: number | null
  step: number | null
  values: Record<string, number | null>
}

export interface MetricSeries {
  run_id: string
  points: MetricPoint[]
  columns: string[]
  warnings: string[]
}

export interface BenchmarkResult {
  run_id: string
  checkpoint: string | null
  checkpoint_episode: number | null
  checkpoint_step: number | null
  opponent: string | null
  opponent_family: string | null
  opponent_depth: number | null
  games: number | null
  wins: number | null
  losses: number | null
  draws: number | null
  win_rate: number | null
  side: string | null
  seed: number | null
  avg_game_length: number | null
  illegal_action_count: number | null
  timeout_count: number | null
  crash_count: number | null
  extra: Record<string, unknown>
}

export interface BenchmarksResponse {
  run_id: string
  benchmarks: BenchmarkResult[]
  count: number
}

export interface RunsResponse {
  runs: RunSummary[]
  count: number
  artifact_roots_scanned: string[]
  warnings: string[]
}

export interface DiagnosisItem {
  level: 'info' | 'warning' | 'positive' | 'error'
  message: string
}

export interface DiagnosisReport {
  run_id: string
  items: DiagnosisItem[]
}

export interface SummaryResponse {
  total_runs: number
  runs_with_metrics: number
  runs_with_benchmarks: number
  runs_with_checkpoints: number
  latest_run_id: string | null
  latest_modified: string | null
  available_algorithms: string[]
  available_reward_modes: string[]
  available_model_arches: string[]
}

// -------------------------------------------------------------------------
// Evaluation replays
// -------------------------------------------------------------------------

export interface EvalReplayMeta {
  episode: number | null
  checkpoint: string | null
  opponent_name: string | null
  opponent_type: string | null
  opponent_depth: number | null
  game_index: number | null
  dqn_player_id: string | null
  winner: string | null
  result_from_dqn_perspective: string | null
  game_length: number | null
  illegal_acts: number | null
  replay_path: string
  created_at: string | null
}

export interface EvalReplaysResponse {
  run_id: string
  count: number
  replays: EvalReplayMeta[]
}

/** State snapshot at one point during a replay. */
export interface EvalReplayState {
  p1: [number, number]   // [x, y] engine coords
  p2: [number, number]
  h_walls: [number, number][]   // list of [wx, wy] action_space coords
  v_walls: [number, number][]
  walls_remaining_p1: number
  walls_remaining_p2: number
}

/** A single action during a replay. */
export interface EvalReplayAction {
  turn_index: number
  player_id: 'P1' | 'P2'
  action_id: number
  action_type: 'pawn' | 'hwall' | 'vwall'
  x: number
  y: number
  is_dqn_action: boolean
  actor_name: string
  actor_type: string
}

/** Full replay detail (the replay JSON file contents). */
export interface EvalReplayDetail {
  schema_version: string
  run_id: string
  agent_id: string
  episode: number
  checkpoint: string
  eval_opponent: { name: string; type: string; depth: number | null }
  game_index: number
  dqn_player_id: string
  opponent_player_id: string
  winner: string | null
  result_from_dqn_perspective: string
  game_length: number
  illegal_acts: number
  termination_reason: string
  created_at: string
  actions: EvalReplayAction[]
  states: EvalReplayState[]
}
