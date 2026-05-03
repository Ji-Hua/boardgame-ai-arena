import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { RunSummary, SummaryResponse } from '../types'

interface Props {
  onSelectRun: (runId: string) => void
}

type SortKey = 'run_id' | 'algorithm' | 'reward_mode' | 'model_arch' | 'total_episodes' | 'modified_at' | 'best_benchmark_win_rate'
type SortDir = 'asc' | 'desc'

function fmt(v: number | null | undefined, digits = 1): string {
  if (v == null) return '—'
  return (v * 100).toFixed(digits) + '%'
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  return iso.slice(0, 16).replace('T', ' ')
}

function Badges({ run }: { run: RunSummary }) {
  return (
    <span>
      <span className={run.has_metrics ? 'badge badge-ok' : 'badge badge-missing'}>M</span>
      <span className={run.has_benchmarks ? 'badge badge-ok' : 'badge badge-missing'}>B</span>
      <span className={run.has_checkpoints ? 'badge badge-ok' : 'badge badge-missing'}>C</span>
      {run.has_notes && <span className="badge badge-ok">N</span>}
    </span>
  )
}

function WrBar({ wr }: { wr: number | null }) {
  if (wr == null) return <span className="muted">—</span>
  const pct = Math.round(wr * 100)
  const color = wr >= 0.5 ? 'var(--good)' : wr >= 0.2 ? 'var(--warn)' : 'var(--error)'
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span className="wr-bar-outer"><span className="wr-bar-inner" style={{ width: `${pct}%`, background: color }} /></span>
      <span style={{ color }}>{pct}%</span>
    </span>
  )
}

export function RunListView({ onSelectRun }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [filterAlg, setFilterAlg] = useState('')
  const [filterReward, setFilterReward] = useState('')
  const [filterArch, setFilterArch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('modified_at')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  useEffect(() => {
    setLoading(true)
    Promise.all([api.listRuns(), api.summary()])
      .then(([r, s]) => { setRuns(r.runs); setSummary(s) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  const algorithms = useMemo(() => [...new Set(runs.map(r => r.algorithm).filter(Boolean))].sort(), [runs])
  const rewardModes = useMemo(() => [...new Set(runs.map(r => r.reward_mode).filter((x): x is string => x != null))].sort(), [runs])
  const modelArches = useMemo(() => [...new Set(runs.map(r => r.model_arch).filter((x): x is string => x != null))].sort(), [runs])

  const filtered = useMemo(() => {
    let list = runs
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(r => r.run_id.toLowerCase().includes(q))
    }
    if (filterAlg) list = list.filter(r => r.algorithm === filterAlg)
    if (filterReward) list = list.filter(r => r.reward_mode === filterReward)
    if (filterArch) list = list.filter(r => r.model_arch === filterArch)
    return list
  }, [runs, search, filterAlg, filterReward, filterArch])

  const sorted = useMemo(() => {
    const asc = sortDir === 'asc'
    return [...filtered].sort((a, b) => {
      const av = a[sortKey] as string | number | null | undefined
      const bv = b[sortKey] as string | number | null | undefined
      if (av == null && bv == null) return 0
      if (av == null) return asc ? 1 : -1
      if (bv == null) return asc ? -1 : 1
      if (typeof av === 'string' && typeof bv === 'string') {
        return asc ? av.localeCompare(bv) : bv.localeCompare(av)
      }
      const an = Number(av)
      const bn = Number(bv)
      return asc ? an - bn : bn - an
    })
  }, [filtered, sortKey, sortDir])

  const handleSort = useCallback((key: SortKey) => {
    if (key === sortKey) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }, [sortKey])

  function thProps(key: SortKey) {
    const active = sortKey === key
    return {
      className: `sortable${active ? (' sort-' + sortDir) : ''}`,
      onClick: () => handleSort(key),
      style: { cursor: 'pointer' },
    }
  }

  if (loading) return <div className="loading">Loading runs…</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div>
      <h2>Training Runs</h2>

      {summary && (
        <div className="summary-bar">
          <span className="summary-bar-item"><strong>{summary.total_runs}</strong> runs</span>
          <span className="summary-bar-item"><strong>{summary.runs_with_metrics}</strong> with metrics</span>
          <span className="summary-bar-item"><strong>{summary.runs_with_benchmarks}</strong> with benchmarks</span>
          <span className="summary-bar-item"><strong>{summary.runs_with_checkpoints}</strong> with checkpoints</span>
          {summary.latest_run_id && (
            <span className="summary-bar-item">Latest: <strong style={{ color: 'var(--accent)' }}>{summary.latest_run_id}</strong></span>
          )}
        </div>
      )}

      <div className="list-controls">
        <input
          type="text"
          placeholder="Search by run ID…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <select value={filterAlg} onChange={e => setFilterAlg(e.target.value)}>
          <option value="">All algorithms</option>
          {algorithms.map(a => <option key={a} value={a}>{a}</option>)}
        </select>
        <select value={filterReward} onChange={e => setFilterReward(e.target.value)}>
          <option value="">All reward modes</option>
          {rewardModes.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
        <select value={filterArch} onChange={e => setFilterArch(e.target.value)}>
          <option value="">All architectures</option>
          {modelArches.map(a => <option key={a} value={a}>{a}</option>)}
        </select>
        <span className="muted" style={{ marginLeft: 'auto' }}>
          {sorted.length} / {runs.length} runs
        </span>
      </div>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th {...thProps('run_id')}>Run ID</th>
              <th {...thProps('algorithm')}>Algorithm</th>
              <th {...thProps('reward_mode')}>Reward</th>
              <th {...thProps('model_arch')}>Arch</th>
              <th {...thProps('total_episodes')}>Episodes</th>
              <th title="M=metrics B=benchmarks C=checkpoints N=notes">Data</th>
              <th {...thProps('best_benchmark_win_rate')}>Best WR</th>
              <th {...thProps('modified_at')}>Modified</th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 && (
              <tr><td colSpan={8} className="empty">No runs match filters</td></tr>
            )}
            {sorted.map(run => (
              <tr
                key={run.run_id}
                className="clickable-row"
                onClick={() => onSelectRun(run.run_id)}
              >
                <td className="run-id">{run.run_id}</td>
                <td>{run.algorithm || '—'}</td>
                <td className="muted">{run.reward_mode || '—'}</td>
                <td className="muted">{run.model_arch || '—'}</td>
                <td>{run.total_episodes ?? '—'}</td>
                <td><Badges run={run} /></td>
                <td><WrBar wr={run.best_benchmark_win_rate} /></td>
                <td className="muted">{fmtDate(run.modified_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p style={{ marginTop: 10, fontSize: 12, color: 'var(--text-muted)' }}>
        Badges: M=metrics, B=benchmarks, C=checkpoints, N=notes. Click a row to view details.
      </p>
    </div>
  )
}
