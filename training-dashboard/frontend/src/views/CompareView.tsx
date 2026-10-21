import React, { useEffect, useState } from 'react'
import { api } from '../api'
import type { BenchmarkResult, MetricSeries, RunDetail, RunSummary, RunsResponse } from '../types'
import { LineChart } from '../components/LineChart'

interface CompareViewProps {
  onBack: () => void
}

function pct(v: number | null | undefined): string {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

const COMPARE_METRICS = [
  { key: 'loss', label: 'Loss', color: '#e05c5c' },
  { key: 'avg_reward', label: 'Avg Reward', color: '#4f9eff' },
  { key: 'win_rate', label: 'Win Rate', color: '#5ce05c' },
]

const COLORS = ['#4f9eff', '#e05c5c', '#5ce05c', '#f0a030']

export function CompareView({ onBack }: CompareViewProps) {
  const [allRuns, setAllRuns] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [details, setDetails] = useState<Record<string, RunDetail>>({})
  const [metricMap, setMetricMap] = useState<Record<string, MetricSeries>>({})
  const [benchMap, setBenchMap] = useState<Record<string, BenchmarkResult[]>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [xKey] = useState<'episode' | 'step'>('episode')

  useEffect(() => {
    api.listRuns()
      .then((r: RunsResponse) => {
        setAllRuns(r.runs)
        // Preselect latest 3 runs by default
        setSelected(r.runs.slice(0, 3).map(x => x.run_id))
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  function toggleRun(runId: string) {
    setSelected(prev => {
      if (prev.includes(runId)) return prev.filter(r => r !== runId)
      if (prev.length >= 5) return prev
      return [...prev, runId]
    })
  }

  useEffect(() => {
    for (const runId of selected) {
      if (!details[runId]) {
        api.getRun(runId).then(d => setDetails(prev => ({ ...prev, [runId]: d }))).catch(() => {})
      }
      if (!metricMap[runId]) {
        api.getMetrics(runId).then(m => setMetricMap(prev => ({ ...prev, [runId]: m }))).catch(() => {})
      }
      if (!benchMap[runId]) {
        api.getBenchmarks(runId).then(b => setBenchMap(prev => ({ ...prev, [runId]: b.benchmarks }))).catch(() => {})
      }
    }
  }, [selected]) // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) return <div className="loading">Loading runs…</div>
  if (error) return <div className="error">Error: {error}</div>

  const selectedDetails = selected.map(id => details[id]).filter(Boolean)

  return (
    <div>
      <button onClick={onBack} className="back-btn">← Back</button>
      <h2>Compare Runs</h2>
      <p className="muted">Select 2–5 runs to compare. Latest 3 preselected.</p>

      <div className="compare-selector">
        {allRuns.map(r => (
          <label key={r.run_id} className={`compare-chip ${selected.includes(r.run_id) ? 'selected' : ''}`}>
            <input
              type="checkbox"
              checked={selected.includes(r.run_id)}
              onChange={() => toggleRun(r.run_id)}
              disabled={!selected.includes(r.run_id) && selected.length >= 4}
            />
            {r.run_id}
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
                    {selected.map(id => <th key={id}>{id}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {(['algorithm', 'reward_mode', 'model_arch', 'total_episodes'] as const).map(field => {
                    const vals = selected.map(id => String(selectedDetails.find(d => d?.run_id === id)?.[field] ?? '…'))
                    const allSame = vals.every(v => v === vals[0])
                    return (
                      <tr key={field}>
                        <td className="kv-key">{field}</td>
                        {selected.map((id, i) => (
                          <td key={id} className={!allSame ? 'diff-highlight' : ''}>{vals[i]}</td>
                        ))}
                      </tr>
                    )
                  })}
                  {/* Config keys — highlight differing values */}
                  {Array.from(new Set(selectedDetails.flatMap(d => Object.keys(d?.config ?? {})))).map(k => {
                    const vals = selected.map(id => String(details[id]?.config?.[k] ?? '—'))
                    const allSame = vals.every(v => v === vals[0])
                    return (
                      <tr key={k}>
                        <td className="kv-key">{k}</td>
                        {selected.map((id, i) => (
                          <td key={id} className={!allSame ? 'diff-highlight' : ''}>{vals[i]}</td>
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
            <h3>Latest Benchmark Win Rates</h3>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Opponent</th>
                    <th>Episode</th>
                    <th>Win Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.flatMap(id => {
                    const bs = benchMap[id] ?? []
                    if (bs.length === 0) return [
                      <tr key={id}><td>{id}</td><td colSpan={3} className="muted">loading…</td></tr>
                    ]
                    // Show last benchmark per opponent
                    const byOpp: Record<string, BenchmarkResult> = {}
                    for (const b of bs) {
                      const k = b.opponent ?? 'unknown'
                      byOpp[k] = b
                    }
                    return Object.values(byOpp).map((b, i) => (
                      <tr key={`${id}-${i}`}>
                        <td>{id}</td>
                        <td>{b.opponent ?? '—'}</td>
                        <td>{b.checkpoint_episode ?? '—'}</td>
                        <td>{pct(b.win_rate)}</td>
                      </tr>
                    ))
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Metric overlay charts */}
          <section>
            <h3>Metric Curves</h3>
            <div className="charts-grid">
              {COMPARE_METRICS.map(m => {
                const hasData = selected.some(id => metricMap[id]?.columns.includes(m.key))
                if (!hasData) return null
                return (
                  <div key={m.key} className="chart-card">
                    <div className="chart-title">{m.label}</div>
                    <div style={{ position: 'relative' }}>
                      {selected.map((id, idx) => {
                        const pts = metricMap[id]?.points ?? []
                        if (!pts.length) return null
                        return (
                          <div key={id} style={{ position: idx === 0 ? 'relative' : 'absolute', top: 0, left: 0 }}>
                            <LineChart
                              points={pts}
                              xKey={xKey}
                              yKey={m.key}
                              color={COLORS[idx % COLORS.length]}
                              label={id}
                              width={550}
                              height={200}
                            />
                          </div>
                        )
                      })}
                    </div>
                    <div className="legend-row">
                      {selected.map((id, idx) => (
                        <span key={id} className="legend-item" style={{ color: COLORS[idx % COLORS.length] }}>■ {id}</span>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </section>
        </>
      )}
    </div>
  )
}
