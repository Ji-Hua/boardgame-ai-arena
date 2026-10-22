import React, { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { encodeContext, externalPlayUrl, safeReturnRoute } from '../platform'
import type { AnalysisContext } from '../platform'
import { ModelPicker } from '../components/ModelPicker'
import { ResearchContext } from '../components/ResearchContext'

export function LabView() {
  const [params, setParams] = useSearchParams()
  const query = params.toString()
  const [context, setContext] = useState<AnalysisContext | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    let active = true
    setContext(null); setError('')
    // GET resolves evidence only. No scoring/search/analysis request is sent.
    api.context(new URLSearchParams(query)).then(data => {
      if (!data.position || typeof data.position !== 'object' || !safeReturnRoute(data.return_route)) throw new Error('Incomplete or unsafe AnalysisContext returned by server')
      if (active) setContext(data)
    }).catch(e => { if (active) setError(String(e)) })
    return () => { active = false }
  }, [query, retry])
  return <div className="transition-page">
    <p className="eyebrow">Research / Decision Lab</p><h1>Inspect an exact decision</h1>
    <p className="muted">Import a server-validated position and its source identities into the isolated, same-origin Lab. Opening this page does not run analysis.</p>
    <ModelPicker recordedSha={context?.recorded_model?.sha256} value={params.get('model_sha') ?? ''} onChange={sha => { const next = new URLSearchParams(params); if (sha) next.set('model_sha', sha); else next.delete('model_sha'); setParams(next) }} />
    {error ? <div className="error" role="alert">Context unavailable: {error}<div><button onClick={() => setRetry(v => v + 1)}>Retry context</button></div></div> : !context ? <div className="loading" role="status">Resolving validated context…</div> : <section className="panel">
      <h2>Ready to import</h2><p>Recorded actor and selected analysis model remain separate. No production roster or alias is changed.</p>
      <ResearchContext context={context} />
      {(context.warnings ?? []).map((warning, i) => <p className="warning" key={i}>{warning}</p>)}
      <details className="context-provenance"><summary>Full context provenance (JSON)</summary><pre className="context-preview">{JSON.stringify(context, null, 2)}</pre></details>
      <div className="action-bar"><a className="button-link primary" href={`/research-lab/?context=${encodeContext(context)}`}>Open isolated Lab ↗</a><Link to={safeReturnRoute(context.return_route)!}>Return to source</Link></div>
    </section>}
  </div>
}

export function PlayView() {
  const url = externalPlayUrl(import.meta.env.VITE_PLAY_URL)
  return <div className="transition-page"><p className="eyebrow">Application / External Dev</p><h1>Play Quoridor</h1>
    <section className="panel"><h2>Continue to the existing Dev application ↗</h2><p>Live gameplay runs separately from this read-only research platform. It is not embedded here, and research model selection does not alter its deployed agents.</p>
      {url ? <><p className="mono">{url}</p><a className="button-link primary" href={url} target="_blank" rel="noopener noreferrer">Open external Play ↗</a></> : <p className="warning">External Play is not configured. Set VITE_PLAY_URL to the existing Dev application’s HTTP(S) URL at build time.</p>}
      <Link to="/arena">Inspect Arena evidence instead</Link>
    </section></div>
}