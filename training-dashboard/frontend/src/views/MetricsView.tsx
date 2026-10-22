import React, { useEffect, useState } from 'react'
import { api } from '../api'
import type { AxisKey, MetricSeries } from '../types'
import { LineChart } from '../components/LineChart'
import { availableAxes, COLORS, display } from '../runData'

interface MetricsViewProps {
  runId: string
  onBack?: () => void
  embedded?: boolean
}

export function MetricsView({ runId, onBack, embedded = false }: MetricsViewProps) {
  const [series, setSeries] = useState<MetricSeries | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [xKey, setXKey] = useState<AxisKey>('episode')

  useEffect(() => {
    setLoading(true)
    setError(null)
    setSeries(null)
    let active = true
    api.getMetrics(runId)
      .then(r => { if (active) setSeries(r) })
      .catch(e => { if (active) setError(String(e)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [runId])

  if (loading) return <div className="loading">Loading metrics…</div>
  if (error) return <div className="error">Error: {error}</div>
  if (!series) return null

  const axes = availableAxes(series)
  const axis = axes.find(a => a.key === xKey) ?? axes[0]

  return (
    <div>
      {!embedded && <button onClick={onBack} className="back-btn">← Back to Run</button>}
      {!embedded && <h2>Metrics — {runId}</h2>}

      {series.warnings.map((w, i) => <div key={i} className="warning">{w}</div>)}

      {axis && <div className="action-bar">
        <label>X axis: </label>
        <select aria-label="Metric X axis" value={axis.key} onChange={e => setXKey(e.target.value as AxisKey)}>
          {axes.map(a => <option key={a.key} value={a.key}>{a.label}</option>)}
        </select>
        <span className="muted" style={{ marginLeft: 12 }}>{series.points.length} data points</span>
      </div>}

      {series.points.length === 0 || !axis ? (
        <div className="empty">No recorded metric curve with an actual progress axis available.</div>
      ) : (
        <div className="charts-grid">
          {series.columns.map((key, index) => {
            const meta = series.metadata?.[key]
            const label = meta?.label ?? key
            return <div key={key} className="chart-card">
              <div className="chart-title">{label} · {meta?.unit ?? 'unit unspecified'}</div>
              <p className="muted">{meta?.semantic ?? 'Semantic unspecified'} · {meta?.scope ?? 'scope unspecified'}</p>
              <LineChart
                points={series.points}
                xKey={axis.key}
                xLabel={axis.label}
                yKey={key}
                yLabel={`${label} (${meta?.unit ?? 'unspecified'})`}
                color={COLORS[index % COLORS.length]}
                width={550}
                height={200}
              />
            </div>
          })}
        </div>
      )}
      <section>
        <h3>Recorded snapshots (not training curves)</h3>
        {!series.snapshots?.length && <div className="empty">No held-out or checkpoint snapshots recorded.</div>}
        {(series.snapshots ?? []).map((snapshot, i) => <div className="chart-card" key={i}>
          <h4>{snapshot.scope ?? 'Unspecified scope'} · {snapshot.source ?? snapshot.run_id ?? 'Recorded snapshot'}</h4>
          <table className="kv-table"><tbody>{Object.entries(snapshot.values).map(([key, value]) => <tr key={key}>
            <td>{snapshot.metadata?.[key]?.label ?? key}</td><td><pre>{display(value)}</pre></td>
            <td>{snapshot.metadata?.[key]?.unit ?? 'unit unspecified'} · {snapshot.metadata?.[key]?.semantic ?? 'semantic unspecified'}</td>
          </tr>)}</tbody></table>
        </div>)}
      </section>
      {Object.keys(series.provenance ?? {}).length > 0 && <details><summary>Metric provenance</summary><pre>{display(series.provenance)}</pre></details>}
    </div>
  )
}
