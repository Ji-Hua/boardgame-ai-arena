import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { BenchmarkResult, BenchmarksResponse } from '../types'

interface BenchmarkViewProps {
  runId: string
  onBack?: () => void
  embedded?: boolean
}

function pct(v: number | null): string {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

function num(v: number | null | undefined): string {
  if (v == null) return '—'
  return String(v)
}

function fnum(v: number | null | undefined, dec = 1): string {
  if (v == null) return '—'
  return v.toFixed(dec)
}

const COLORS = ['#4f9eff', '#5ce05c', '#f0a030', '#e05c5c', '#c04fc0', '#40c0c0']

/** Mini SVG win-rate progression chart (single series, all benchmarks mixed) */
function WrProgressChart({ benchmarks }: { benchmarks: BenchmarkResult[] }) {
  const pts = benchmarks
    .filter(b => b.checkpoint_episode != null && b.win_rate != null)
    .sort((a, b) => (a.checkpoint_episode ?? 0) - (b.checkpoint_episode ?? 0))

  if (pts.length < 2) return null

  const W = 460, H = 140
  const pad = { top: 10, right: 10, bottom: 30, left: 45 }
  const cW = W - pad.left - pad.right
  const cH = H - pad.top - pad.bottom

  const xs = pts.map(p => p.checkpoint_episode!)
  const ys = pts.map(p => p.win_rate!)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = 0, yMax = 1

  const sx = (x: number) => ((x - xMin) / ((xMax - xMin) || 1)) * cW
  const sy = (y: number) => cH - ((y - yMin) / ((yMax - yMin) || 1)) * cH

  const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.checkpoint_episode!).toFixed(1)} ${sy(p.win_rate!).toFixed(1)}`).join(' ')

  return (
    <div className="chart-card" style={{ marginBottom: 16 }}>
      <div className="chart-title">Win Rate Progression</div>
      <svg width={W} height={H} style={{ display: 'block' }}>
        <g transform={`translate(${pad.left},${pad.top})`}>
          {[0, 0.25, 0.5, 0.75, 1].map(v => (
            <g key={v}>
              <line x1={0} y1={sy(v)} x2={cW} y2={sy(v)} stroke={v === 0.5 ? '#3a5a3a' : '#333'} strokeWidth={v === 0.5 ? 1 : 0.5} />
              <text x={-4} y={sy(v) + 4} textAnchor="end" fontSize={10} fill="#aaa">{(v * 100).toFixed(0)}%</text>
            </g>
          ))}
          <line x1={0} y1={sy(0.5)} x2={cW} y2={sy(0.5)} stroke="#3a5a3a" strokeWidth={1} strokeDasharray="4,4" />
          <path d={pathD} fill="none" stroke="#5ce05c" strokeWidth={1.5} />
          {pts.map((p, i) => (
            <circle key={i} cx={sx(p.checkpoint_episode!)} cy={sy(p.win_rate!)} r={3} fill="#5ce05c" />
          ))}
          <line x1={0} y1={0} x2={0} y2={cH} stroke="#555" />
          <line x1={0} y1={cH} x2={cW} y2={cH} stroke="#555" />
          <text x={cW / 2} y={cH + 26} textAnchor="middle" fontSize={10} fill="#888">Episode</text>
        </g>
      </svg>
    </div>
  )
}

/** Per-opponent multi-line win-rate chart (shown when multiple named opponents exist) */
function PerOpponentWrChart({ benchmarks }: { benchmarks: BenchmarkResult[] }) {
  // Group by opponent name
  const byOpponent: Record<string, BenchmarkResult[]> = {}
  for (const b of benchmarks) {
    const key = b.opponent ?? 'unknown'
    if (!byOpponent[key]) byOpponent[key] = []
    byOpponent[key].push(b)
  }
  const opponentNames = Object.keys(byOpponent)
  if (opponentNames.length < 2) return null

  // Only render if at least one opponent has >= 2 data points
  const seriesData = opponentNames.map(name => ({
    name,
    pts: byOpponent[name]
      .filter(b => b.checkpoint_episode != null && b.win_rate != null)
      .sort((a, b) => (a.checkpoint_episode ?? 0) - (b.checkpoint_episode ?? 0)),
  })).filter(s => s.pts.length >= 1)

  if (seriesData.length < 2) return null

  const W = 500, H = 180
  const pad = { top: 16, right: 120, bottom: 36, left: 45 }
  const cW = W - pad.left - pad.right
  const cH = H - pad.top - pad.bottom

  const allEps = seriesData.flatMap(s => s.pts.map(p => p.checkpoint_episode!))
  const xMin = Math.min(...allEps), xMax = Math.max(...allEps)
  const sx = (x: number) => ((x - xMin) / ((xMax - xMin) || 1)) * cW
  const sy = (y: number) => cH - y * cH

  return (
    <div className="chart-card" style={{ marginBottom: 16 }}>
      <div className="chart-title">Win Rate by Opponent</div>
      <svg width={W} height={H} style={{ display: 'block' }}>
        <g transform={`translate(${pad.left},${pad.top})`}>
          {[0, 0.25, 0.5, 0.75, 1].map(v => (
            <g key={v}>
              <line x1={0} y1={sy(v)} x2={cW} y2={sy(v)} stroke={v === 0.5 ? '#3a5a3a' : '#2a2a2a'} strokeWidth={v === 0.5 ? 1 : 0.5} />
              <text x={-4} y={sy(v) + 4} textAnchor="end" fontSize={9} fill="#aaa">{(v * 100).toFixed(0)}%</text>
            </g>
          ))}
          <line x1={0} y1={sy(0.5)} x2={cW} y2={sy(0.5)} stroke="#3a5a3a" strokeWidth={1} strokeDasharray="4,4" />
          {seriesData.map((series, si) => {
            const color = COLORS[si % COLORS.length]
            const pathD = series.pts.map((p, i) =>
              `${i === 0 ? 'M' : 'L'} ${sx(p.checkpoint_episode!).toFixed(1)} ${sy(p.win_rate!).toFixed(1)}`
            ).join(' ')
            return (
              <g key={series.name}>
                {series.pts.length >= 2 && (
                  <path d={pathD} fill="none" stroke={color} strokeWidth={1.5} />
                )}
                {series.pts.map((p, i) => (
                  <circle key={i} cx={sx(p.checkpoint_episode!)} cy={sy(p.win_rate!)} r={3} fill={color} />
                ))}
              </g>
            )
          })}
          <line x1={0} y1={0} x2={0} y2={cH} stroke="#555" />
          <line x1={0} y1={cH} x2={cW} y2={cH} stroke="#555" />
          <text x={cW / 2} y={cH + 28} textAnchor="middle" fontSize={10} fill="#888">Episode</text>
          {/* Legend */}
          {seriesData.map((series, si) => (
            <g key={series.name} transform={`translate(${cW + 8}, ${si * 18})`}>
              <rect x={0} y={-8} width={12} height={3} fill={COLORS[si % COLORS.length]} />
              <text x={16} y={0} fontSize={10} fill="#ccc">{series.name}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  )
}

export function BenchmarkView({ runId, onBack, embedded = false }: BenchmarkViewProps) {
  const [data, setData] = useState<BenchmarksResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.getBenchmarks(runId)
      .then(setData)
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [runId])

  const groups = useMemo(() => {
    if (!data) return {}
    const out: Record<string, BenchmarkResult[]> = {}
    for (const b of data.benchmarks) {
      const fam = b.opponent_family ?? 'other'
      if (!out[fam]) out[fam] = []
      out[fam].push(b)
    }
    return out
  }, [data])

  const latestVsBest = useMemo(() => {
    if (!data || data.benchmarks.length === 0) return null
    const wrs = data.benchmarks.filter(b => b.win_rate != null)
    if (wrs.length === 0) return null
    const latest = wrs[wrs.length - 1]
    const best = wrs.reduce((a, b) => (b.win_rate! > a.win_rate! ? b : a))
    return { latest, best }
  }, [data])

  const failureStats = useMemo(() => {
    if (!data) return null
    const illegal = data.benchmarks.reduce((s, b) => s + (b.illegal_action_count ?? 0), 0)
    const timeouts = data.benchmarks.reduce((s, b) => s + (b.timeout_count ?? 0), 0)
    const crashes = data.benchmarks.reduce((s, b) => s + (b.crash_count ?? 0), 0)
    if (illegal + timeouts + crashes === 0) return null
    return { illegal, timeouts, crashes }
  }, [data])

  if (loading) return <div className="loading">Loading benchmarks…</div>
  if (error) return <div className="error">Error: {error}</div>
  if (!data) return null

  const benchmarks = data.benchmarks

  return (
    <div>
      {!embedded && <button onClick={onBack} className="back-btn">← Back to Run</button>}
      {!embedded && <h2>Benchmarks — {runId}</h2>}
      <p className="muted" style={{ marginBottom: 12 }}>{benchmarks.length} result{benchmarks.length !== 1 ? 's' : ''}</p>

      {benchmarks.length === 0 ? (
        <div className="empty">No benchmark results found for this run.</div>
      ) : (
        <>
          {/* Latest vs Best */}
          {latestVsBest && (
            <div className="cards-row" style={{ marginBottom: 16 }}>
              <div className="card">
                <div className="card-label">Latest WR</div>
                <div className={`card-value ${(latestVsBest.latest.win_rate ?? 0) >= 0.5 ? 'good' : 'warn'}`}>
                  {pct(latestVsBest.latest.win_rate)}
                </div>
                <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>{latestVsBest.latest.opponent}</div>
              </div>
              <div className="card">
                <div className="card-label">Best WR</div>
                <div className={`card-value ${(latestVsBest.best.win_rate ?? 0) >= 0.5 ? 'good' : 'warn'}`}>
                  {pct(latestVsBest.best.win_rate)}
                </div>
                <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>ep {latestVsBest.best.checkpoint_episode ?? '?'} vs {latestVsBest.best.opponent}</div>
              </div>
            </div>
          )}

          {/* Win rate progression */}
          <WrProgressChart benchmarks={benchmarks} />
          <PerOpponentWrChart benchmarks={benchmarks} />

          {/* Failure stats */}
          {failureStats && (
            <div className="warning" style={{ marginBottom: 16 }}>
              ⚠ Failure stats — illegal actions: {failureStats.illegal} | timeouts: {failureStats.timeouts} | crashes: {failureStats.crashes}
            </div>
          )}

          {/* Grouped tables */}
          {Object.entries(groups).map(([fam, rows], gi) => (
            <div key={fam} className="bench-section">
              <div className="bench-section-title">
                {fam.charAt(0).toUpperCase() + fam.slice(1)} ({rows.length})
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Episode</th>
                      <th>Opponent</th>
                      {rows.some(r => r.opponent_depth != null) && <th>Depth</th>}
                      <th>Games</th>
                      <th>Wins</th>
                      <th>Losses</th>
                      <th>Draws</th>
                      <th>Win Rate</th>
                      <th>Avg Len</th>
                      {rows.some(r => r.illegal_action_count != null) && <th>Illegal</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((b: BenchmarkResult, i: number) => (
                      <tr key={i}>
                        <td>{num(b.checkpoint_episode)}</td>
                        <td>{b.opponent ?? '—'}</td>
                        {rows.some(r => r.opponent_depth != null) && <td>{num(b.opponent_depth)}</td>}
                        <td>{num(b.games)}</td>
                        <td>{num(b.wins)}</td>
                        <td>{num(b.losses)}</td>
                        <td>{num(b.draws)}</td>
                        <td style={{ color: b.win_rate != null && b.win_rate > 0.5 ? 'var(--good)' : undefined }}>
                          {pct(b.win_rate)}
                        </td>
                        <td>{fnum(b.avg_game_length)}</td>
                        {rows.some(r => r.illegal_action_count != null) && <td>{num(b.illegal_action_count)}</td>}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
