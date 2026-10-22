import React, { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { DiagnosisReport, RunDetail } from '../types'
import { BenchmarkView } from './BenchmarkView'
import { MetricsView } from './MetricsView'
import EvalReplaysView from './EvalReplaysView'
import { display } from '../runData'
import { arenaRoute, replayRoute, runTab } from '../platform'

interface Props {
  runId: string
  onBack: () => void
}

type Tab = 'overview' | 'metrics' | 'benchmarks' | 'checkpoints' | 'config' | 'diagnosis' | 'replays'

function Card({ label, value, cls }: { label: string; value: React.ReactNode; cls?: string }) {
  return (
    <div className="card">
      <div className="card-label">{label}</div>
      <div className={`card-value${cls ? ' ' + cls : ''}`}>{value ?? '—'}</div>
    </div>
  )
}

function pct(v: number | null | undefined) {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

function fmt3(v: number | null | undefined) {
  if (v == null) return '—'
  return v.toFixed(4)
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  return iso.slice(0, 16).replace('T', ' ')
}

function DiagnosisPanel({ runId }: { runId: string }) {
  const [report, setReport] = useState<DiagnosisReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setErr(null)
    let active = true
    api.getDiagnosis(runId)
      .then(r => { if (active) setReport(r) })
      .catch(e => { if (active) setErr(String(e)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [runId])

  if (loading) return <div className="loading">Analyzing run…</div>
  if (err) return <div className="error">{err}</div>
  if (!report) return null

  const iconMap: Record<string, string> = {
    info: 'ℹ',
    warning: '⚠',
    positive: '✓',
    error: '✗',
  }

  return (
    <div>
      <h3>Automated Diagnosis</h3>
      {report.items.length === 0 ? (
        <div className="muted">No diagnosis observations recorded.</div>
      ) : (
        <div className="diag-list">
          {report.items.map((item, i) => (
            <div key={i} className={`diag-item diag-${item.level}`}>
              <span className="diag-icon">{iconMap[item.level] ?? '•'}</span>
              <span>{item.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function RunDetailView({ runId, onBack }: Props) {
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [params, setParams] = useSearchParams()
  const tab = runTab(params.get('tab'))
  const setTab = (value: Tab) => { const next = new URLSearchParams(params); next.set('tab', value); setParams(next) }

  useEffect(() => {
    setLoading(true)
    setError(null)
    setDetail(null)
    let active = true
    api.getRun(runId)
      .then(r => { if (active) setDetail(r) })
      .catch(e => { if (active) setError(String(e)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [runId])

  if (loading) return <div className="loading">Loading run {runId}…</div>
  if (error) return <div><div className="error">{error}</div><button className="back-btn" onClick={onBack}>← Back</button></div>
  if (!detail) return null

  const stats = detail.summary_stats as Record<string, number | null>

  const tabs: { id: Tab; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'metrics', label: 'Metrics' },
    { id: 'benchmarks', label: 'Benchmarks' },
    { id: 'checkpoints', label: `Checkpoints (${detail.checkpoints.length})` },
    { id: 'config', label: 'Config' },
    { id: 'diagnosis', label: 'Diagnosis' },
    { id: 'replays', label: 'Eval Replays' },
  ]

  return (
    <div>
      <button className="back-btn" onClick={onBack}>← All Runs</button>

      {/* Header */}
      <div className="run-header-bar">
        <span className="run-header-id">{detail.run_name ?? runId}</span>
        <div className="run-header-meta">
          <span>{detail.source ?? 'legacy'}</span>
          <span>{detail.status ?? 'Status not recorded'}</span>
          <span>{detail.experiment_name ?? 'Experiment not recorded'}</span>
          <span>{detail.algorithm}</span>
          {detail.reward_mode && <span>{detail.reward_mode}</span>}
          {detail.model_arch && <span>{detail.model_arch}</span>}
          <span style={{ color: 'var(--text-muted)' }}>Modified: {fmtDate(detail.modified_at)}</span>
        </div>
      </div>
      <div className="muted mono">{runId}</div>
      <div className="action-bar"><Link to={arenaRoute(runId)}>Run Arena evaluations →</Link><Link to={replayRoute(runId)}>Recorded evaluations & replays →</Link><Link to={`/training/compare?${new URLSearchParams({ run: runId })}`}>Compare this run</Link></div>
      {(detail.warnings ?? []).map((w, i) => <div className="warning" key={i}>{w}</div>)}

      {/* Tab bar */}
      <div className="tab-bar">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab${tab === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {tab === 'overview' && (
        <div>
          <div className="cards-row">
            <Card label="Total Episodes" value={stats.total_episodes ?? detail.total_episodes ?? '—'} />
            <Card label="Env Steps" value={stats.total_env_steps != null ? stats.total_env_steps.toLocaleString() : '—'} />
            <Card label="Avg Reward" value={fmt3(stats.avg_episode_reward)} />
            <Card label="Final Loss" value={fmt3(stats.final_avg_loss)} />
            {stats.final_avg_q_max != null && <Card label="Final Q-Max (Q-value)" value={fmt3(stats.final_avg_q_max)} />}
            <Card label="Checkpoints" value={detail.checkpoints.length || '—'} />
          </div>

          <section>
            <h3>Summary</h3>
            <table className="kv-table">
              <tbody>
                {Object.entries(stats).map(([k, v]) => (
                  <tr key={k}>
                    <td className="kv-key">{k}</td>
                    <td className="kv-val"><pre>{display(v)}</pre></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {detail.has_notes && detail.notes && (
            <section>
              <h3>Notes</h3>
              <div className="notes">{detail.notes}</div>
            </section>
          )}

          <section>
            <h3>Available Files</h3>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              {detail.metric_files.length > 0 && (
                <div>
                  <div className="muted" style={{ marginBottom: 4 }}>Metric files</div>
                  {detail.metric_files.map(f => <div key={f} className="mono">{f}</div>)}
                </div>
              )}
              {detail.benchmark_files.length > 0 && (
                <div>
                  <div className="muted" style={{ marginBottom: 4 }}>Benchmark files</div>
                  {detail.benchmark_files.map(f => <div key={f} className="mono">{f}</div>)}
                </div>
              )}
            </div>
          </section>
        </div>
      )}

      {/* Metrics */}
      {tab === 'metrics' && <MetricsView runId={runId} embedded />}

      {/* Benchmarks */}
      {tab === 'benchmarks' && <BenchmarkView runId={runId} embedded />}

      {/* Checkpoints */}
      {tab === 'checkpoints' && (
        <section>
          <h3>Checkpoints</h3>
          {detail.checkpoints.length === 0 ? (
            <div className="empty">No checkpoints found</div>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Episode</th>
                    <th>Step</th>
                    <th>Progress type</th>
                    <th>SHA-256 / model type / artifact reference</th>
                    <th>Has Benchmark</th>
                    <th>Best WR</th>
                    <th>Modified</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.checkpoints.map((ck, i) => (
                    <tr key={`${ck.sha256 ?? ck.path}-${i}`}>
                      <td className="mono">{ck.filename}</td>
                      <td>{ck.episode ?? '—'}</td>
                      <td>{ck.step ?? '—'}</td>
                      <td>{ck.progress_type ?? 'Unspecified'}</td>
                      <td><div className="mono sha">{ck.sha256 ? <Link to={`/models/${ck.sha256}`}>{ck.sha256}</Link> : 'SHA unavailable'}</div><div>{ck.model_type ?? 'Model type unavailable'}</div><div className="mono">{ck.artifact_ref ?? ck.path ?? 'Artifact reference unavailable'}</div><div className="action-bar"><Link to={arenaRoute(runId, ck.sha256, ck.filename)}>Arena evaluations</Link><Link to={replayRoute(runId, { checkpoint_sha: ck.sha256, checkpoint: ck.sha256 ? undefined : ck.filename })}>Checkpoint evaluations & replays</Link></div></td>
                      <td>
                        {ck.has_benchmark
                          ? <span style={{ color: 'var(--good)' }}>✓</span>
                          : <span className="muted">—</span>}
                      </td>
                      <td>
                        {ck.best_win_rate != null
                          ? <span style={{ color: 'var(--good)' }}>{(ck.best_win_rate * 100).toFixed(1)}%</span>
                          : <span className="muted">—</span>}
                      </td>
                      <td className="muted">{ck.modified_at ? ck.modified_at.slice(0, 16).replace('T', ' ') : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* Config */}
      {tab === 'config' && (
        <section>
          <h3>Training Config</h3>
          {Object.keys(detail.config).length === 0 ? (
            <div className="muted">No configuration recorded or accessible.</div>
          ) : (
            <table className="kv-table">
              <tbody>
                {Object.entries(detail.config).map(([k, v]) => (
                  <tr key={k}>
                    <td className="kv-key">{k}</td>
                    <td className="kv-val"><pre>{display(v)}</pre></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      {(tab === 'overview' || tab === 'config') && <section>
        <h3>Architecture & provenance</h3>
        <p>Architecture: {detail.model_arch ?? 'Not recorded'} · Algorithm: {detail.algorithm} · Output semantics: {display(detail.provenance?.output_semantics)}</p>
        <pre className="notes">{display(detail.provenance ?? {})}</pre>
      </section>}

      {/* Diagnosis */}
      {tab === 'diagnosis' && <DiagnosisPanel runId={runId} />}

      {/* Eval Replays */}
      {tab === 'replays' && <EvalReplaysView runId={runId} />}
    </div>
  )
}
