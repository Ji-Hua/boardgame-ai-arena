import type { BenchmarksResponse, DiagnosisReport, EvalReplaysResponse, EvalReplayDetail, MetricSeries, RunDetail, RunsResponse, SummaryResponse } from './types'
import type { AnalysisContext, EvaluationsResponse, ModelsResponse, PlatformModel } from './platform'

const BASE = ''  // proxied via Vite to port 8740

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  models: () => get<ModelsResponse>('/api/platform/models'),
  model: (sha: string) => get<PlatformModel>(`/api/platform/models/${encodeURIComponent(sha)}`),
  evaluations: (runId: string) => get<EvaluationsResponse>(`/api/platform/runs/${encodeURIComponent(runId)}/evaluations`),
  context: (query: URLSearchParams) => get<AnalysisContext>(`/api/platform/context?${query}`),
  health: () => get<Record<string, unknown>>('/health'),
  summary: () => get<SummaryResponse>('/api/summary'),
  listRuns: () => get<RunsResponse>('/api/runs'),
  getRun: (runId: string) => get<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`),
  getMetrics: (runId: string) => get<MetricSeries>(`/api/runs/${encodeURIComponent(runId)}/metrics`),
  getBenchmarks: (runId: string) => get<BenchmarksResponse>(`/api/runs/${encodeURIComponent(runId)}/benchmarks`),
  getDiagnosis: (runId: string) => get<DiagnosisReport>(`/api/runs/${encodeURIComponent(runId)}/diagnosis`),
  listEvalReplays: (runId: string) => get<EvalReplaysResponse>(`/api/runs/${encodeURIComponent(runId)}/eval-replays`),
  getEvalReplay: (runId: string, replayPath: string) =>
    get<EvalReplayDetail>(`/api/runs/${encodeURIComponent(runId)}/eval-replays/${replayPath.split('/').map(encodeURIComponent).join('/')}`),
}
