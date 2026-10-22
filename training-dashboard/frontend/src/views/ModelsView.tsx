import React, { useEffect, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { arenaRoute, contextQuery, replayRoute, runRoute } from '../platform'
import type { ModelsResponse, PlatformModel } from '../platform'

const capabilities = ['available', 'sha_verified', 'analyzable', 'playable', 'deployed'] as const
function Capabilities({ model }: { model: PlatformModel }) {
  return <div className="capabilities">{capabilities.map(key => <div key={key} className={`capability ${model[key] ? 'eligible' : 'ineligible'}`}><strong>{key.replace(/_/g, ' ')}: {model[key] ? 'Yes' : 'No'}</strong><p>{(model.reasons[key] ?? []).join(' · ') || 'No blockers recorded'}</p></div>)}</div>
}
export function ModelsView() {
  const { sha } = useParams()
  const [params, setParams] = useSearchParams()
  const [listing, setListing] = useState<ModelsResponse | null>(null)
  const [model, setModel] = useState<PlatformModel | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    let active = true
    setError(''); setModel(null); setListing(null)
    if (sha && !/^[a-f0-9]{64}$/.test(sha)) { setError('A model route requires its full lowercase 64-character SHA-256, not an alias or abbreviated hash.'); return }
    const request = sha ? api.model(sha).then(data => { if (active) setModel(data) }) : api.models().then(data => { if (active) setListing(data) })
    request.catch(e => { if (active) setError(String(e)) })
    return () => { active = false }
  }, [sha, retry])
  const query = params.get('q') ?? ''
  const capability = params.get('capability') ?? ''
  const change = (key: string, value: string) => { const next = new URLSearchParams(params); if (value) next.set(key, value); else next.delete(key); setParams(next, { replace: true }) }
  return <div><div className="page-heading"><div><p className="eyebrow">Research / Models</p><h1>{sha ? 'Model identity' : 'Model library'}</h1><p className="muted">SHA-addressed provenance and explicit capabilities. Registration is not deployment.</p></div>{sha && <Link to="/models">All models</Link>}</div>
    {error && <div className="error" role="alert">{error}<div><button onClick={() => setRetry(v => v + 1)}>Retry</button></div></div>}
    {!error && !listing && !model && <div className="loading">Loading model evidence…</div>}
    {(model?.warnings ?? listing?.warnings ?? []).map((warning, i) => <div className="warning" key={i}>{warning}</div>)}
    {listing && <>
      <div className="list-controls"><label>Search <input aria-label="Search models" type="search" value={query} onChange={e => change('q', e.target.value)} placeholder="Alias, architecture or full SHA" /></label><label>Capability <select value={capability} onChange={e => change('capability', e.target.value)}><option value="">All models</option>{capabilities.map(key => <option key={key} value={key}>{key.replace(/_/g, ' ')}</option>)}</select></label><span className="muted">{listing.count} indexed models</span></div>
      <div className="model-grid">{listing.models.filter(m => [m.sha256, ...m.aliases, m.architecture ?? ''].join(' ').toLowerCase().includes(query.toLowerCase()) && (!capability || !capabilities.includes(capability as typeof capabilities[number]) || m[capability as typeof capabilities[number]])).map(m => <article className="panel model-card" key={m.sha256}>
        <h2><Link to={`/models/${m.sha256}`}>{m.aliases.join(' / ') || m.architecture || 'Unnamed model'}</Link></h2><Link className="mono sha" to={`/models/${m.sha256}`}>{m.sha256}</Link><p>{m.architecture ?? 'Architecture unknown'} · {m.output_semantics ?? 'Output semantics unknown'}</p><Capabilities model={m} />
        <Link to={`/models/${m.sha256}`}>Inspect provenance →</Link>
      </article>)}</div>{listing.models.length === 0 && <div className="empty">No models indexed. Check source warnings; no identities are inferred from names.</div>}
      <details className="panel"><summary>Catalog source availability</summary><pre>{JSON.stringify(listing.sources, null, 2)}</pre></details>
    </>}
    {model && <>
      <section className="panel"><h2>{model.aliases.join(' / ') || 'SHA-addressed checkpoint'}</h2><p className="mono sha">{model.sha256}</p><p>{model.architecture ?? 'Architecture unknown'} · {model.output_semantics ?? 'Output semantics unknown'}</p><Capabilities model={model} />
        <p className="warning">Analysis/play capabilities are static eligibility, not successful load claims. SHA verification and observed deployment are separate evidence. No deploy action is available.</p>
        <div className="action-bar">{model.analyzable ? <Link className="button-link" to={`/lab?${contextQuery(undefined, undefined, undefined, model.sha256)}`}>Prepare analysis context</Link> : <button disabled title={(model.reasons.analyzable ?? []).join(', ')}>Analysis unavailable</button>}<Link to="/play">External Play information ↗</Link></div>
        {model.conflicts.length > 0 && <div className="error">Conflicting evidence: {model.conflicts.join(' · ')}</div>}
      </section>
      <section className="panel"><h2>Run & checkpoint lineage</h2>{model.run_refs.length ? model.run_refs.map((ref, i) => <div className="lineage-row" key={i}><Link to={runRoute(ref.run_id, 'checkpoints')}>{ref.run_id}</Link><span>{ref.source} · {ref.evidence}</span><span className="mono sha">{ref.checkpoint_sha256}</span><Link to={arenaRoute(ref.run_id, model.sha256)}>Arena evaluations →</Link><Link to={replayRoute(ref.run_id, { checkpoint_sha: model.sha256 })}>Recorded evaluations & replays →</Link></div>) : <p className="muted">No exact SHA-backed run reference is recorded. No name-based run join is attempted.</p>}</section>
      <section className="panel"><h2>Checkpoint availability & verification</h2>{model.checkpoints.map((checkpoint, i) => <div key={i}><p className="mono sha">{checkpoint.path}</p><p>{checkpoint.available ? 'Available' : 'Unavailable'} · {checkpoint.sha_verified ? 'SHA verified' : 'SHA not verified'} · {checkpoint.reason}</p>{checkpoint.actual_sha256 && <p className="mono sha">Actual SHA: {checkpoint.actual_sha256}</p>}</div>)}</section>
      <details className="panel"><summary>Full model provenance / registry / AgentSpec evidence</summary><pre>{JSON.stringify(model, null, 2)}</pre></details>
    </>}
  </div>
}