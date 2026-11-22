import React, { useEffect, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { RunSummary } from '../types'
import { replayRoute, runRoute, selectedReplay } from '../platform'
import EvalReplaysView from './EvalReplaysView'
import { ResearchContext } from '../components/ResearchContext'

export function ReplaysView() {
  const { runId } = useParams()
  const [params, setParams] = useSearchParams()
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    if (runId) return
    let active = true
    setError(''); setRuns(null)
    api.listRuns().then(data => { if (active) { setRuns(data.runs); setWarnings(data.warnings ?? []) } }).catch(e => { if (active) setError(String(e)) })
    return () => { active = false }
  }, [runId, retry])
  const search = params.get('q') ?? ''
  return <div>{runId && !selectedReplay(params) && <ResearchContext />}<div className="page-heading"><div><p className="eyebrow">Research / Replay</p><h1>{runId ? 'Run replay archive' : 'Global replay index'}</h1><p className="muted">Run → checkpoint → evaluation / opponent → recorded game. Replay samples are not aggregate evaluation scores.</p></div><Link to={runId ? '/replays' : '/training'}>{runId ? 'All replay runs' : 'Training runs'}</Link></div>
    {runId ? <><div className="action-bar"><Link to={runRoute(runId)}>Run details</Link><span className="mono sha">{runId}</span></div><EvalReplaysView key={runId} runId={runId} /></> : <>
      <div className="list-controls"><label>Find run <input type="search" value={search} onChange={e => { const next = new URLSearchParams(params); if (e.target.value) next.set('q', e.target.value); else next.delete('q'); setParams(next, { replace: true }) }} /></label></div>
      {warnings.map((warning, i) => <div className="warning" key={i}>{warning}</div>)}
      {error && <div className="error" role="alert">{error}<button onClick={() => setRetry(v => v + 1)}>Retry</button></div>}
      {!runs && !error && <div className="loading">Loading run index…</div>}
      {runs?.filter(run => [run.run_id, run.run_name ?? '', run.source].join(' ').toLowerCase().includes(search.toLowerCase())).map(run => <details className="replay-group" key={run.run_id}>
        <summary>{run.run_name ?? run.run_id} · {run.source} · {run.has_eval_replays ? 'Recorded games indexed' : run.has_benchmarks ? 'Aggregate summaries; replay availability may vary' : 'No replay evidence advertised'}</summary>
        <div className="action-bar"><Link to={replayRoute(run.run_id)}>Open stable run replay list →</Link><Link to={runRoute(run.run_id)}>Run details</Link></div>
        <LazyRunReplays runId={run.run_id} />
      </details>)}
      {runs?.length === 0 && <div className="empty">No runs available.</div>}
    </>}
  </div>
}

// Fetch a run's evidence only when its global index group is expanded.
function LazyRunReplays({ runId }: { runId: string }) {
  const [expanded, setExpanded] = useState(false)
  const ref = React.useRef<HTMLDivElement>(null)
  useEffect(() => {
    const parent = ref.current?.closest('details')
    const toggle = () => { if (parent?.open) setExpanded(true) }
    parent?.addEventListener('toggle', toggle); toggle()
    return () => parent?.removeEventListener('toggle', toggle)
  }, [])
  return <div ref={ref}>{expanded && <EvalReplaysView runId={runId} indexOnly />}</div>
}