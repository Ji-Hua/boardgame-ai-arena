import React from 'react'
import { Link, matchPath, useLocation } from 'react-router-dom'
import { arenaRoute, replayRoute, runRoute, selectedReplay } from '../platform'
import type { AnalysisContext } from '../platform'

/** Compact identities only. Full evidence remains in collapsible provenance. */
export function ResearchContext({ context }: { context?: AnalysisContext }) {
  const { pathname, search } = useLocation()
  const params = new URLSearchParams(search)
  const encodedRun = matchPath('/training/runs/:runId', pathname)?.params.runId ?? matchPath('/replays/:runId', pathname)?.params.runId
  let routeRun = encodedRun
  try { if (encodedRun) routeRun = decodeURIComponent(encodedRun) } catch { /* Keep malformed route visible without crashing the shell. */ }
  const run = context?.source?.run?.run_id ?? routeRun ?? params.get('run_id')
  const checkpoint = context?.recorded_model?.sha256 ?? params.get('checkpoint_sha') ?? matchPath('/models/:sha', pathname)?.params.sha
  const replay = context?.source?.replay_path ?? selectedReplay(params)
  const ply = context?.source?.ply ?? params.get('ply') ?? 0
  const analysis = context?.analysis_model?.sha256 ?? params.get('model_sha')
  const evaluationFilter = context ? context.source?.evaluation?.evaluation_id : params.get('evaluation')
  const evaluation = context ? context.source?.evaluation?.evaluation_id ?? context.source?.evaluation?.id : evaluationFilter
  const game = context?.source?.game?.id
  if (!run && !checkpoint && !replay && !analysis && !context) return null
  return <aside className="research-context" aria-label="Research context">
    <strong>Research context</strong>
    <dl className="context-identities">
      {run && <div><dt>Run {context?.source?.run?.source && `· ${context.source.run.source}`}</dt><dd><Link to={runRoute(run)}>{run}</Link></dd></div>}
      {checkpoint && <div><dt>Recorded checkpoint</dt><dd><Link to={`/models/${checkpoint}`}>{checkpoint}</Link></dd></div>}
      {evaluation && <div><dt>Evaluation</dt><dd>{evaluation}</dd></div>}
      {game && <div><dt>Recorded game</dt><dd>{game}</dd></div>}
      {replay && <div><dt>Replay · ply {ply}</dt><dd>{replay}</dd></div>}
      {analysis && <div><dt>Analysis model {checkpoint === analysis ? '· recorded model' : '· hypothetical selection'}</dt><dd><Link to={`/models/${analysis}`}>{analysis}</Link></dd></div>}
      {context?.position_hash && <div><dt>Position hash</dt><dd>{context.position_hash}</dd></div>}
    </dl>
    {run && <div className="action-bar"><Link to={arenaRoute(run, checkpoint, params.get('checkpoint'))}>Arena evaluations</Link><Link to={replayRoute(run, { checkpoint_sha: checkpoint, replay, ply: replay ? ply : undefined, model_sha: analysis, evaluation: evaluationFilter })}>Replay archive</Link></div>}
  </aside>
}