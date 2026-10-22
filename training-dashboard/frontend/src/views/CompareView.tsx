import React, { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { BenchmarkResult, MetricSeries, RunDetail, RunSummary, RunsResponse } from '../types'
import { LineChart } from '../components/LineChart'
import { comparisonGroups, display } from '../runData'
import { EvaluationTable } from './BenchmarkView'

interface CompareViewProps {
  onBack: () => void
}

function pct(v: number | null | undefined): string {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

const COLORS = ['#4f9eff', '#e05c5c', '#5ce05c', '#f0a030']

export function CompareView({ onBack }: CompareViewProps) {
  const [allRuns, setAllRuns] = useState<RunSummary[]>([])
  const [params, setParams] = useSearchParams()
  const selectionKey = params.getAll('run').join('\u0000')
  const selected = React.useMemo(() => Array.from(new Set(params.getAll('run').filter(Boolean))).slice(0, 5), [selectionKey])
  const [details, setDetails] = useState<Record<string, RunDetail>>({})
  const [metricMap, setMetricMap] = useState<Record<string, MetricSeries>>({})
  const [benchMap, setBenchMap] = useState<Record<string, BenchmarkResult[]>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadErrors, setLoadErrors] = useState<Record<string, string>>({})
  const [warnings, setWarnings] = useState<string[]>([])
  const [benchmarkWarnings, setBenchmarkWarnings] = useState<Record<string, string[]>>({})
  const [refreshVersion, setRefreshVersion] = useState(0)
  const [refreshing, setRefreshing] = useState(false)
  const [xKey] = useState<'episode' | 'step'>('episode')

  useEffect(() => {
    api.listRuns()
      .then((r: RunsResponse) => {
        setAllRuns(r.runs)
        setWarnings(r.warnings)
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  function toggleRun(runId: string) {
    const nextSelected = selected.includes(runId) ? selected.filter(r => r !== runId) : selected.length < 5 ? [...selected, runId] : selected
    const next = new URLSearchParams(params)
    next.delete('run')
    nextSelected.forEach(id => next.append('run', id))
    setParams(next)
  }

  useEffect(() => {
    let active = true
    // One request generation for every selected run and endpoint. Invalidate
    // snapshots and warnings together, never overlay old and refreshed evidence.
    setDetails({}); setMetricMap({}); setBenchMap({}); setBenchmarkWarnings({})
    setRefreshing(selected.length > 0)
    async function load<T>(key: string, request: () => Promise<T>, apply: (data: T) => void) {
      try {
        const data = await request()
        if (!active) return
        apply(data)
        setLoadErrors(prev => {
          const next = { ...prev }
          delete next[key]
          return next
        })
      } catch (e) {
        if (active) setLoadErrors(prev => ({ ...prev, [key]: String(e) }))
      }
    }
    void Promise.all(selected.flatMap(runId => [
      load(`${runId}:detail`, () => api.getRun(runId), d => setDetails(prev => ({ ...prev, [runId]: d }))),
      load(`${runId}:metrics`, () => api.getMetrics(runId), m => setMetricMap(prev => ({ ...prev, [runId]: m }))),
      load(`${runId}:benchmarks`, () => api.getBenchmarks(runId), b => {
        setBenchMap(prev => ({ ...prev, [runId]: b.benchmarks }))
        setBenchmarkWarnings(prev => ({ ...prev, [runId]: b.warnings ?? [] }))
      }),
    ])).then(() => { if (active) setRefreshing(false) })
    return () => { active = false }
  }, [selected, refreshVersion])

  if (loading) return <div className="loading">Loading runs…</div>
  if (error) return <div className="error">Error: {error}</div>

  const selectedDetails = selected.map(id => details[id]).filter(Boolean)
  const name = (id: string) => allRuns.find(r => r.run_id === id)?.run_name ?? id
  const overlays = comparisonGroups(selected.flatMap(id => metricMap[id] ? [{ runId: id, name: name(id), series: metricMap[id] }] : []))

  return (
    <div>
      <button onClick={onBack} className="back-btn">← Back</button>
      <h2>Compare Runs</h2>
      <button onClick={() => setRefreshVersion(version => version + 1)} disabled={refreshing || !selected.length}>{refreshing ? 'Refreshing selected runs…' : 'Refresh selected runs'}</button>
      <p className="muted">Select 2–5 Legacy or MLflow runs. Selection is stored in the URL. Overlay only exact semantic/unit/scope and actual axis types; unknown types are excluded. All overlaid curves share X/Y domains.</p>
      {warnings.map((w, i) => <div key={i} className="warning">{w}</div>)}
      {selected.flatMap(id => (benchmarkWarnings[id] ?? []).map((w, i) => <div key={`${id}-${i}`} className="warning">{name(id)}: {w}</div>))}
      {Object.entries(loadErrors).filter(([key]) => selected.some(id => key.startsWith(`${id}:`))).map(([key, message]) => <div className="error" key={key}>{key}: {message}</div>)}

      <div className="compare-selector">
        {allRuns.map(r => (
          <label key={r.run_id} className={`compare-chip ${selected.includes(r.run_id) ? 'selected' : ''}`}>
            <input
              type="checkbox"
              checked={selected.includes(r.run_id)}
              onChange={() => toggleRun(r.run_id)}
              disabled={!selected.includes(r.run_id) && selected.length >= 5}
            />
            {r.run_name ?? r.run_id} · {r.source ?? 'legacy'} · {r.algorithm}
          </label>
        ))}
      </div>

      {selected.length < 2 ? (
        <div className="empty">Select at least 2 runs above.</div>
      ) : (
        <>
          {/* Metadata comparison with diff highlighting */}
          <section>
            <h3>Config Comparison</h3>
            <div className="table-scroll">
              <table className="diff-table">
                <thead>
                  <tr>
                    <th>Field</th>
                    {selected.map(id => <th key={id}>{name(id)}<div className="muted mono">{id}</div></th>)}
                  </tr>
                </thead>
                <tbody>
                  {(['source', 'status', 'experiment_name', 'algorithm', 'reward_mode', 'model_arch', 'total_episodes', 'provenance'] as const).map(field => {
                    const vals = selected.map(id => display(details[id]?.[field] ?? (loadErrors[`${id}:detail`] ? 'Unavailable' : '…')))
                    const allSame = vals.every(v => v === vals[0])
                    return (
                      <tr key={field}>
                        <td className="kv-key">{field}</td>
                        {selected.map((id, i) => (
                          <td key={id} className={!allSame ? 'diff-highlight' : ''}><pre>{vals[i]}</pre></td>
                        ))}
                      </tr>
                    )
                  })}
                  {/* Config keys — highlight differing values */}
                  {Array.from(new Set(selectedDetails.flatMap(d => Object.keys(d?.config ?? {})))).map(k => {
                    const vals = selected.map(id => display(details[id]?.config?.[k]))
                    const allSame = vals.every(v => v === vals[0])
                    return (
                      <tr key={k}>
                        <td className="kv-key">{k}</td>
                        {selected.map((id, i) => (
                          <td key={id} className={!allSame ? 'diff-highlight' : ''}><pre>{vals[i]}</pre></td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Benchmark win rate comparison */}
          <section>
            <h3>Checkpoint evaluation summaries</h3>
            <p className="muted">Recorded aggregate W/D/L, win rate and score rate remain separate for each protocol/opponent/checkpoint. No pooled score or cross-protocol ranking.</p>
            {selected.map(id => <div key={id}><h4>{name(id)}</h4>{benchMap[id] ? <EvaluationTable rows={benchMap[id]} /> : <div className="muted">{loadErrors[`${id}:benchmarks`] ? 'Unavailable' : 'Loading summaries…'}</div>}</div>)}
          </section>

          {/* Metric overlay charts */}
          <section>
            <h3>Compatible metric curves</h3>
            {selected.some(id => !metricMap[id] && !loadErrors[`${id}:metrics`]) && <p className="muted">Loading metric evidence…</p>}
            {!overlays.length && <div className="empty">No compatible curves with exact semantic/unit/scope/axis metadata among loaded runs.</div>}
            <div className="charts-grid">
              {overlays.map(([key, group]) => <div key={key} className="chart-card">
                <div className="chart-title">{group.label} · {group.unit}</div><p className="muted">{group.scope} · {group.axisType} · shared X/Y domains</p>
                <LineChart series={group.curves.map(c => ({ ...c, color: COLORS[selected.indexOf(c.runId) % COLORS.length] }))} xKey={group.curves[0].xKey} xLabel={group.axisType} yLabel={`${group.label} (${group.unit})`} width={550} height={220} />
                <div className="legend-row">{group.curves.map((c, i) => <span key={`${c.runId}-${i}`} className="legend-item" style={{ color: COLORS[selected.indexOf(c.runId) % COLORS.length] }}>■ {c.label} · {c.yKey}</span>)}</div>
              </div>)}
            </div>
            {selected.flatMap(id => (metricMap[id]?.warnings ?? []).map((w, i) => <div className="warning" key={`${id}-${i}`}>{name(id)}: {w}</div>))}
          </section>
          <section><h3>Recorded snapshots — not training curves</h3>
            {selected.map(id => <details key={id}><summary>{name(id)} · {metricMap[id]?.snapshots?.length ?? 0} snapshots</summary>
              {(metricMap[id]?.snapshots ?? []).map((snapshot, i) => <div key={i}><h4>{snapshot.scope ?? 'Scope unspecified'} · {snapshot.source ?? snapshot.run_id}</h4><pre>{display(snapshot)}</pre></div>)}
            </details>)}
          </section>
        </>
      )}
    </div>
  )
}
