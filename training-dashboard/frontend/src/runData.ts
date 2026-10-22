import type { AxisKey, BenchmarkResult, EvalReplayMeta, EvalReplayState, MetricMetadata, MetricPoint, MetricSeries } from './types'

export const COLORS = ['#4f9eff', '#e05c5c', '#5ce05c', '#f0a030', '#c04fc0']
export const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
export const display = (value: unknown): string => value == null ? '—' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)
export const percent = (value: number | null | undefined) => finite(value) ? `${(value * 100).toFixed(1)}%` : '—'

export function axisValue(point: MetricPoint, axis: AxisKey): number | null {
  const value = point[axis]
  if (finite(value)) return value
  if (axis === 'timestamp' && typeof value === 'string') {
    const time = Date.parse(value)
    return finite(time) ? time : null
  }
  return null
}

export function chartData(points: MetricPoint[], axis: AxisKey, metric: string) {
  return points.flatMap(point => {
    const x = axisValue(point, axis), y = point.values[metric]
    return x !== null && finite(y) ? [{ x, y }] : []
  }).sort((a, b) => a.x - b.x)
}

export function chartDomains(series: { points: MetricPoint[]; yKey: string; xKey?: AxisKey }[], axis: AxisKey) {
  const data = series.flatMap(s => chartData(s.points, s.xKey ?? axis, s.yKey))
  if (!data.length) return null
  return data.reduce((d, p) => ({ xMin: Math.min(d.xMin, p.x), xMax: Math.max(d.xMax, p.x), yMin: Math.min(d.yMin, p.y), yMax: Math.max(d.yMax, p.y) }),
    { xMin: data[0].x, xMax: data[0].x, yMin: data[0].y, yMax: data[0].y })
}

// A generic MLflow `step` is NOT assumed to be environment steps.
export function axisType(series: MetricSeries, metric: string, axis: AxisKey): string {
  if (axis === 'update' || axis === 'gradient_updates') return 'gradient_updates'
  if (axis !== 'step') return axis
  const type = series.metadata?.[metric]?.step_type
  if (type === 'update' || type === 'gradient_updates') return 'gradient_updates'
  if (type === 'env_steps') return 'env_steps'
  return 'unknown_step'
}

export function availableAxes(series: MetricSeries): { key: AxisKey; label: string }[] {
  const declared = series.axes?.length ? series.axes : [
    { key: 'episode' as const, label: 'Episode' }, { key: 'step' as const, label: 'Recorded step' },
    { key: 'update' as const, label: 'Gradient updates' }, { key: 'gradient_updates' as const, label: 'Gradient updates' },
    { key: 'timestamp' as const, label: 'Timestamp' },
  ]
  return declared.filter(axis => series.points.some(p => axisValue(p, axis.key) !== null)).map(axis => {
    const types = new Set(series.columns.map(key => axisType(series, key, axis.key)))
    const label = axis.key === 'step' ? types.size === 1 && types.has('env_steps') ? 'Environment steps' : types.size === 1 && types.has('gradient_updates') ? 'Gradient updates (step)' : 'Recorded step (type unspecified)' : axis.label
    return { ...axis, label }
  })
}

export function semanticKey(meta: MetricMetadata | undefined, actualAxis: string): string | null {
  if (!meta?.semantic || !meta.unit || !meta.scope || meta.scope === 'unknown' || actualAxis === 'unknown_step') return null
  return JSON.stringify([meta.semantic, meta.unit, meta.scope, actualAxis])
}

export function comparisonGroups(entries: { runId: string; name: string; series: MetricSeries }[]) {
  const groups = new Map<string, { label: string; unit: string; scope: string; axisType: string; curves: { runId: string; label: string; points: MetricPoint[]; yKey: string; xKey: AxisKey }[] }>()
  for (const entry of entries) for (const metric of entry.series.columns) {
    const used = new Set<string>()
    for (const axis of availableAxes(entry.series)) {
      const actualAxis = axisType(entry.series, metric, axis.key)
      const meta = entry.series.metadata?.[metric]
      const key = semanticKey(meta, actualAxis)
      if (!key || used.has(actualAxis) || !chartData(entry.series.points, axis.key, metric).length) continue
      used.add(actualAxis)
      if (!groups.has(key)) groups.set(key, { label: meta.label ?? metric, unit: meta.unit!, scope: meta.scope!, axisType: actualAxis, curves: [] })
      groups.get(key)!.curves.push({ runId: entry.runId, label: entry.name, points: entry.series.points, yKey: metric, xKey: axis.key })
    }
  }
  return [...groups.entries()].filter(([, g]) => new Set(g.curves.map(c => c.runId)).size >= 2)
}

// Older replay indexes omit these aggregate identity fields. Missing values
// may match only a unique summary; an agent's seat is not the protocol side.
type Evidence = (BenchmarkResult | EvalReplayMeta) & {
  side?: string | null
  seed?: number | null
  source_manifest?: Record<string, unknown>
}
const isBenchmark = (row: Evidence): row is BenchmarkResult => 'checkpoint_episode' in row
const identityValue = (value: unknown): string => JSON.stringify(value, (_key, item) =>
  item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item)
const protocolIdentity = (row: Evidence) => [row.evaluation_id ?? null, row.protocol ?? null,
  isBenchmark(row) ? row.opponent : row.opponent_name,
  isBenchmark(row) ? row.opponent_family : row.opponent_type,
  row.opponent_depth ?? null, row.game_type ?? 'evaluation', row.side ?? null, row.seed ?? null,
  row.source_manifest?.sha256 ?? null, row.source_manifest?.protocol ?? null]
export function checkpointKey(row: Evidence): string {
  return JSON.stringify([row.checkpoint_sha ?? null, row.checkpoint ?? null, isBenchmark(row) ? row.checkpoint_episode : row.episode,
    isBenchmark(row) ? row.checkpoint_step : row.step, row.progress_type ?? null])
}
export function evaluationKey(row: Evidence): string {
  return identityValue(protocolIdentity(row))
}
export function evaluationGroups(replays: EvalReplayMeta[], benchmarks: BenchmarkResult[]) {
  const checkpoints = new Map<string, { label: string; groups: Map<string, { label: string; games: EvalReplayMeta[]; summaries: BenchmarkResult[] }> }>()
  for (const row of [...benchmarks, ...replays]) {
    // Legacy indexes can omit the checkpoint step stored on the aggregate. Join
    // only when all recorded identity fields agree and the summary group is unique.
    const matches = isBenchmark(row) ? [] : benchmarks.filter(b => {
      const agrees = (a: unknown, c: unknown) => a == null || c == null || identityValue(a) === identityValue(c)
      const identity = row.checkpoint_sha && b.checkpoint_sha ? row.checkpoint_sha === b.checkpoint_sha :
        Boolean(row.checkpoint && b.checkpoint && row.checkpoint === b.checkpoint)
      return identity && agrees(row.checkpoint_sha, b.checkpoint_sha) && agrees(row.checkpoint, b.checkpoint)
        && agrees(row.episode, b.checkpoint_episode) && agrees(row.step, b.checkpoint_step)
        && agrees(row.progress_type, b.progress_type)
        && protocolIdentity(row).every((value, i) => agrees(value, protocolIdentity(b)[i]))
    })
    const canonical: Evidence = matches.length === 1 ? matches[0] : row
    // Never let an unmatched replay fall into a summary through equal partial
    // keys (including identity-less records and ambiguous unknown side/seed).
    const ck = checkpointKey(canonical), ev = `${!isBenchmark(row) && canonical === row ? 'unmatched-replay:' : ''}${evaluationKey(canonical)}`
    if (!checkpoints.has(ck)) {
      const episode = isBenchmark(row) ? row.checkpoint_episode : row.episode
      const step = isBenchmark(row) ? row.checkpoint_step : row.step
      checkpoints.set(ck, { label: `${row.checkpoint ?? 'Checkpoint unspecified'} · ${row.checkpoint_sha ?? 'SHA unavailable'} · episode ${episode ?? '—'} · ${row.progress_type ?? 'recorded step'} ${step ?? '—'}`, groups: new Map() })
    }
    const groups = checkpoints.get(ck)!.groups
    if (!groups.has(ev)) groups.set(ev, { label: `${canonical.evaluation_id ?? 'Legacy evaluation (ID unavailable)'} · ${canonical.protocol ?? 'Protocol unspecified'} · ${(isBenchmark(canonical) ? canonical.opponent : canonical.opponent_name) ?? 'Opponent unspecified'}${canonical.opponent_depth != null ? ` (d${canonical.opponent_depth})` : ''} · ${canonical.game_type ?? 'evaluation'} · side ${canonical.side ?? 'unknown'} · seed ${canonical.seed ?? 'unknown'} · manifest ${canonical.source_manifest?.sha256 ?? 'unknown'}`, games: [], summaries: [] })
    if (isBenchmark(row)) groups.get(ev)!.summaries.push(row)
    else groups.get(ev)!.games.push(row)
  }
  return [...checkpoints.entries()].map(([key, ck]) => ({ key, label: ck.label, groups: [...ck.groups.entries()].map(([key, group]) => ({ key, ...group })) }))
}

// Use recorded turn or next recorded actor, never opening/step parity.
export function replayTurn(state: EvalReplayState, nextActor?: string): 1 | 2 | null {
  if (state.current_player === 'P1' || state.current_player === 1) return 1
  if (state.current_player === 'P2' || state.current_player === 2) return 2
  return nextActor === 'P1' ? 1 : nextActor === 'P2' ? 2 : null
}