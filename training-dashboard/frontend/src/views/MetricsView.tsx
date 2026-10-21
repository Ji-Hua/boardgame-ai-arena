import React, { useEffect, useState } from 'react'
import { api } from '../api'
import type { MetricSeries } from '../types'
import { LineChart } from '../components/LineChart'

interface MetricsViewProps {
  runId: string
  onBack?: () => void
  embedded?: boolean
}

const METRIC_DISPLAY: Array<{ key: string; label: string; color: string }> = [
  { key: 'loss', label: 'Loss', color: '#e05c5c' },
  { key: 'avg_reward', label: 'Avg Reward', color: '#4f9eff' },
  { key: 'win_rate', label: 'Win Rate', color: '#5ce05c' },
  { key: 'epsilon', label: 'Epsilon', color: '#f0a030' },
  { key: 'avg_q_max', label: 'Avg Q Max', color: '#c04fc0' },
  { key: 'avg_episode_length', label: 'Avg Episode Length', color: '#808080' },
]

export function MetricsView({ runId, onBack, embedded = false }: MetricsViewProps) {
  const [series, setSeries] = useState<MetricSeries | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [xKey, setXKey] = useState<'episode' | 'step'>('episode')

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.getMetrics(runId)
      .then(setSeries)
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [runId])

  if (loading) return <div className="loading">Loading metrics…</div>
  if (error) return <div className="error">Error: {error}</div>
  if (!series) return null

  const activeColumns = new Set(series.columns)

  // Additional unknown numeric columns not in the standard display list
  const knownKeys = new Set(METRIC_DISPLAY.map(m => m.key))
  const extraCols = series.columns.filter(c => !knownKeys.has(c))

  return (
    <div>
      {!embedded && <button onClick={onBack} className="back-btn">← Back to Run</button>}
      {!embedded && <h2>Metrics — {runId}</h2>}

      {series.warnings.map((w, i) => <div key={i} className="warning">{w}</div>)}

      <div className="action-bar">
        <label>X axis: </label>
        <select value={xKey} onChange={e => setXKey(e.target.value as 'episode' | 'step')}>
          <option value="episode">Episode</option>
          <option value="step">Step</option>
        </select>
        <span className="muted" style={{ marginLeft: 12 }}>{series.points.length} data points</span>
      </div>

      {series.points.length === 0 ? (
        <div className="empty">No metric data points found.</div>
      ) : (
        <div className="charts-grid">
          {METRIC_DISPLAY.filter(m => activeColumns.has(m.key)).map(m => (
            <div key={m.key} className="chart-card">
              <div className="chart-title">{m.label}</div>
              <LineChart
                points={series.points}
                xKey={xKey}
                yKey={m.key}
                color={m.color}
                width={550}
                height={200}
              />
            </div>
          ))}
          {extraCols.map(col => (
            <div key={col} className="chart-card">
              <div className="chart-title">{col}</div>
              <LineChart
                points={series.points}
                xKey={xKey}
                yKey={col}
                color="#aaa"
                width={550}
                height={200}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
