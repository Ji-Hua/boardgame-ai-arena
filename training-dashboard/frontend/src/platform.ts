import type { BenchmarkResult, EvalReplayMeta } from './types'

export interface PlatformModel {
  id: string
  sha256: string
  aliases: string[]
  architecture: string | null
  output_semantics: string | null
  metadata: Record<string, unknown>
  available: boolean
  sha_verified: boolean
  analyzable: boolean
  playable: boolean
  deployed: boolean
  actual_load_verified: boolean
  requires_isolated_research_catalog: boolean
  deployment_evidence: unknown
  reasons: Record<string, string[]>
  conflicts: string[]
  checkpoints: { path: string; available: boolean; sha_verified: boolean; reason: string; actual_sha256?: string }[]
  run_refs: { run_id: string; source: string; evidence: string; checkpoint?: string; checkpoint_sha256: string }[]
  evidence: unknown[]
  registry: unknown[]
  agent_specs: unknown[]
  warnings?: string[]
}
export interface ModelsResponse { models: PlatformModel[]; count: number; warnings: string[]; sources: unknown[] }
export interface EvaluationsResponse { run_id: string; benchmarks: BenchmarkResult[]; replays: EvalReplayMeta[]; warnings?: string[] }
export interface AnalysisContext {
  position: Record<string, unknown>
  return_route: string
  position_hash?: string
  source?: {
    type?: string
    run?: { run_id: string; source?: string }
    evaluation?: { id?: string; evaluation_id?: string | null } | null
    game?: { id?: string; game_index?: number | null }
    replay_path?: string
    ply?: number
  }
  recorded_model?: { sha256?: string } | null
  analysis_model?: { sha256?: string; aliases?: string[] } | null
  warnings?: string[]
  [key: string]: unknown
}
export const RUN_TABS = ['overview', 'metrics', 'benchmarks', 'checkpoints', 'config', 'diagnosis', 'replays'] as const
export function runTab(value: string | null): typeof RUN_TABS[number] {
  return RUN_TABS.find(tab => tab === value) ?? 'overview'
}
export function plyNumber(value: string | null): number {
  return value && /^\d+$/.test(value) && Number.isSafeInteger(Number(value)) ? Number(value) : 0
}
export function runRoute(runId: string, tab?: string): string {
  return `/training/runs/${encodeURIComponent(runId)}${tab ? `?${new URLSearchParams({ tab })}` : ''}`
}
export function replayRoute(runId: string, values: Record<string, string | number | null | undefined> = {}): string {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(values)) if (value != null && value !== '') query.set(key, String(value))
  return `/replays/${encodeURIComponent(runId)}${query.size ? `?${query}` : ''}`
}
export function arenaRoute(runId: string, checkpointSha?: string | null, checkpoint?: string | null): string {
  const query = new URLSearchParams({ run_id: runId })
  if (checkpointSha) query.set('checkpoint_sha', checkpointSha)
  else if (checkpoint) query.set('checkpoint', checkpoint)
  return `/arena?${query}`
}
export function selectedReplay(params: URLSearchParams): string | null {
  return params.get('replay') || params.get('replay_path')
}
export function contextQuery(runId?: string, replayPath?: string, ply?: number, modelSha?: string): URLSearchParams {
  const query = new URLSearchParams()
  if (runId) query.set('run_id', runId)
  if (replayPath) { query.set('replay_path', replayPath); query.set('ply', String(ply ?? 0)) }
  if (modelSha) query.set('model_sha', modelSha)
  return query
}
export function encodeContext(context: AnalysisContext): string {
  const bytes = new TextEncoder().encode(JSON.stringify(context))
  return btoa(Array.from(bytes, byte => String.fromCharCode(byte)).join('')).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
export function safeReturnRoute(route: unknown): string | null {
  return typeof route === 'string' && /^\/(training|replays|models|arena)(\/|\?|$)/.test(route) && !/[\\\r\n]/.test(route) ? route : null
}
export function externalPlayUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null } catch { return null }
}