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
  models: () => get<ModelsResponse>('/api/v1/models'),
  model: (sha: string) => get<PlatformModel>(`/api/v1/models/${encodeURIComponent(sha)}`),
  evaluations: (runId: string) => get<EvaluationsResponse>(`/api/v1/research/runs/${encodeURIComponent(runId)}/evaluations`),
  context: (query: URLSearchParams) => get<AnalysisContext>(`/api/v1/research/context?${query}`),
  health: () => get<Record<string, unknown>>('/health'),
  summary: () => get<SummaryResponse>('/api/v1/research/summary'),
  listRuns: () => get<RunsResponse>('/api/v1/research/runs'),
  getRun: (runId: string) => get<RunDetail>(`/api/v1/research/runs/${encodeURIComponent(runId)}`),
  getMetrics: (runId: string) => get<MetricSeries>(`/api/v1/research/runs/${encodeURIComponent(runId)}/metrics`),
  getBenchmarks: (runId: string) => get<BenchmarksResponse>(`/api/v1/research/runs/${encodeURIComponent(runId)}/benchmarks`),
  getDiagnosis: (runId: string) => get<DiagnosisReport>(`/api/v1/research/runs/${encodeURIComponent(runId)}/diagnosis`),
  listEvalReplays: (runId: string) => get<EvalReplaysResponse>(`/api/v1/research/runs/${encodeURIComponent(runId)}/eval-replays`),
  getEvalReplay: (runId: string, replayPath: string) =>
    get<EvalReplayDetail>(`/api/v1/research/runs/${encodeURIComponent(runId)}/eval-replays/${replayPath.split('/').map(encodeURIComponent).join('/')}`),
}
