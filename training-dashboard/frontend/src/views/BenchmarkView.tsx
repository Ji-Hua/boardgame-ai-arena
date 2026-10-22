import React, { useEffect, useState } from 'react'
import { api } from '../api'
import type { AxisKey, BenchmarkResult, BenchmarksResponse, MetricPoint } from '../types'
import { LineChart } from '../components/LineChart'
import { COLORS, display, percent } from '../runData'

interface BenchmarkViewProps {
  runId: string
  onBack?: () => void
  embedded?: boolean
}

/** Recorded aggregate results only; replay samples are never used to infer W/D/L. */
export function EvaluationTable({ rows }: { rows: BenchmarkResult[] }) {
  if (!rows.length) return <div className="empty">No aggregate evaluation summaries available.</div>
  return <div className="table-scroll"><table><thead><tr>
    <th>Evaluation / protocol</th><th>Checkpoint / SHA</th><th>Progress</th><th>Opponent / side / seed</th>
    <th>Games</th><th>W / D / L</th><th>Win rate</th><th>Score rate</th><th>Replay / evidence</th>
  </tr></thead><tbody>{rows.map((b, i) => <tr key={i}>
    <td>{b.evaluation_id ?? 'ID not recorded'}<div>{b.protocol ?? 'Protocol unspecified'} · {b.game_type ?? 'evaluation'}</div>
      <details><summary>Protocol / manifest</summary><pre>{display(b.source_manifest ?? {})}</pre><pre>{display(b.extra)}</pre></details></td>
    <td>{b.checkpoint ?? 'Not recorded'}<div className="mono">{b.checkpoint_sha ?? 'SHA unavailable'}</div></td>
    <td>Episode {b.checkpoint_episode ?? '—'}<div>{b.progress_type ?? 'Recorded step'}: {b.checkpoint_step ?? '—'}</div></td>
    <td>{b.opponent ?? 'Unspecified'}{b.opponent_depth != null && ` (d${b.opponent_depth})`}<div>Side {b.side ?? '—'} · seed {b.seed ?? '—'}</div></td>
    <td>{b.games ?? '—'}</td><td>{b.wins ?? '—'} / {b.draws ?? '—'} / {b.losses ?? '—'}</td>
    <td>{percent(b.win_rate)}</td><td>{percent(b.score_rate)}</td>
    <td>{b.replay_available ? 'Recorded replay available' : 'No full replay available'}<div>Illegal: {b.illegal_action_count ?? '—'} · timeouts: {b.timeout_count ?? '—'} · crashes: {b.crash_count ?? '—'}</div></td>
  </tr>)}</tbody></table></div>
}

export function BenchmarkView({ runId, onBack, embedded = false }: BenchmarkViewProps) {
  const [data, setData] = useState<BenchmarksResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    setData(null); setError(null)
    api.getBenchmarks(runId).then(r => { if (active) setData(r) }).catch(e => { if (active) setError(String(e)) })
    return () => { active = false }
  }, [runId])
  if (error) return <div className="error">{error}</div>
  if (!data) return <div className="loading">Loading benchmarks…</div>

  // Keep exact protocol evidence, opponent and actual progress type separate.
  // Unknown legacy protocols remain visible as individual recorded points, not an inferred curve.
  const groups = new Map<string, BenchmarkResult[]>()
  for (const b of data.benchmarks) {
    const key = JSON.stringify([b.evaluation_id, b.protocol, b.source_manifest?.protocol ?? b.extra?.manifest_protocol, b.opponent, b.opponent_depth, b.side, b.seed, b.game_type, b.progress_type])
    groups.set(key, [...(groups.get(key) ?? []), b])
  }
  return <div>
    {!embedded && <button onClick={onBack} className="back-btn">← Back to Run</button>}
    <h3>Checkpoint evaluations</h3>
    <p className="muted">Win rate = wins / games. Score rate is separately recorded (often wins + half draws). Protocols are not pooled; sampled replays do not represent full evaluation counts.</p>
    {(data.warnings ?? []).map((w, i) => <div className="warning" key={i}>{w}</div>)}
    {!data.benchmarks.length && <div className="empty">No benchmark results found for this run.</div>}
    {[...groups.entries()].map(([key, rows], index) => {
      const first = rows[0]
      const hasEpisodes = rows.some(b => b.checkpoint_episode != null)
      const axis: AxisKey = hasEpisodes ? 'episode' : 'step'
      const label = hasEpisodes ? 'Episode' : first.progress_type === 'env_steps' ? 'Environment steps' : first.progress_type === 'gradient_updates' ? 'Gradient updates' : 'Recorded step (type unspecified)'
      const points: MetricPoint[] = rows.map(b => ({ episode: b.checkpoint_episode, step: b.checkpoint_step, values: { win_rate: b.win_rate, score_rate: b.score_rate } }))
      return <section key={key}>
        <h4>{first.protocol ?? 'Protocol unspecified'} · {first.opponent ?? 'Opponent unspecified'} · {first.game_type ?? 'evaluation'}</h4>
        <div className="charts-grid">{(['win_rate', 'score_rate'] as const).filter(metric => points.some(p => p.values[metric] != null)).map(metric => <div className="chart-card" key={metric}>
          <div className="chart-title">{metric === 'win_rate' ? 'Win rate' : 'Score rate'} · fraction</div>
          {!first.protocol && <p className="muted">Protocol not recorded: points only, no inferred progression.</p>}
          <LineChart points={points} xKey={axis} xLabel={label} yKey={metric} connect={Boolean(first.protocol)} color={COLORS[index % COLORS.length]} width={550} />
        </div>)}</div>
        <EvaluationTable rows={rows} />
      </section>
    })}
  </div>
}
