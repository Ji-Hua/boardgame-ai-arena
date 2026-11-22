/**
 * EvalReplaysView — shows a list of saved evaluation game replays for a run
 * and provides an inline step-by-step replay viewer with a Quoridor board.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { BenchmarkResult, EvalReplayMeta, EvalReplayDetail, EvalReplayState, EvalReplaysResponse } from '../types'
import { EvalReplayBoard } from '../components/EvalReplayBoard'
import { evaluationGroups, replayTurn } from '../runData'
import { EvaluationTable } from './BenchmarkView'
import { contextQuery, plyNumber, replayRoute, selectedReplay } from '../platform'
import { ModelPicker } from '../components/ModelPicker'
import { ResearchContext } from '../components/ResearchContext'

interface Props {
  runId: string
  indexOnly?: boolean
  arena?: boolean
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

function resultBadge(result: string | null) {
  if (result === 'win') return <span style={badge('green')}>WIN</span>
  if (result === 'loss') return <span style={badge('red')}>LOSS</span>
  if (result === 'draw') return <span style={badge('gray')}>DRAW</span>
  return <span style={badge('gray')}>UNKNOWN</span>
}

function badge(color: string): React.CSSProperties {
  const map: Record<string, string> = { green: '#28a745', red: '#dc3545', gray: '#6c757d' }
  return {
    background: map[color] ?? '#6c757d',
    color: '#fff',
    padding: '2px 7px',
    borderRadius: 999,
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.03em',
  }
}

function formatOpponent(meta: EvalReplayMeta): string {
  if (meta.opponent_depth != null) {
    return `${meta.opponent_name ?? meta.opponent_type} (d${meta.opponent_depth})`
  }
  return meta.opponent_name ?? meta.opponent_type ?? '—'
}

// -------------------------------------------------------------------------
// ReplayViewer sub-component
// -------------------------------------------------------------------------

interface ReplayViewerProps {
  runId: string
  meta: Pick<EvalReplayMeta, 'replay_path'> & Partial<EvalReplayMeta>
  onClose: () => void
}

export const ReplayViewer: React.FC<ReplayViewerProps> = ({ runId, meta, onClose }) => {
  const [detail, setDetail] = useState<EvalReplayDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [params, setParams] = useSearchParams()
  const requestedPly = plyNumber(params.get('ply'))
  const stepIndex = detail ? Math.min(requestedPly, detail.states.length - 1) : requestedPly
  const setStepIndex = useCallback((value: number | ((current: number) => number)) => {
    setParams(current => { const next = new URLSearchParams(current); next.set('ply', String(typeof value === 'function' ? value(plyNumber(current.get('ply'))) : value)); return next }, { replace: true })
  }, [setParams])
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

  useEffect(() => {
    setDetail(null)
    setError(null)
    setPlaying(false)
    let active = true
    api.getEvalReplay(runId, meta.replay_path)
      .then(replay => {
        if (!active) return
        if (!Array.isArray(replay.states) || !Array.isArray(replay.actions) || replay.states.length < 2 || replay.states.length !== replay.actions.length + 1 || replay.actions.length !== replay.game_length) {
          setError('Full recorded trajectory is unavailable. Playback is disabled; no game will be regenerated.')
          return
        }
        setDetail(replay)
      })
      .catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [runId, meta.replay_path])

  useEffect(() => {
    if (detail && requestedPly !== stepIndex) setStepIndex(stepIndex)
  }, [detail, requestedPly, stepIndex, setStepIndex])

  useEffect(() => {
    if (!detail || !playing) return
    const timer = window.setInterval(() => {
      if (stepIndex >= detail.states.length - 1) { setPlaying(false); return }
      setStepIndex(stepIndex + 1)
    }, 1000 / speed)
    return () => window.clearInterval(timer)
  }, [detail, playing, speed, stepIndex, setStepIndex])

  const handleKey = useCallback((e: KeyboardEvent) => {
    if (!detail) return
    if (e.target instanceof HTMLElement && (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(e.target.tagName) || e.target.isContentEditable)) return
    if (!['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End', ' '].includes(e.key)) return
    e.preventDefault()
    if (e.key === ' ') { setPlaying(p => !p); return }
    setPlaying(false)
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      setStepIndex(s => Math.min(s + 1, detail.states.length - 1))
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      setStepIndex(s => Math.max(s - 1, 0))
    } else if (e.key === 'Home') {
      setStepIndex(0)
    } else if (e.key === 'End') {
      setStepIndex(detail.states.length - 1)
    }
  }, [detail, setStepIndex])

  useEffect(() => {
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleKey])

  if (error) {
    return (
      <div style={viewerBox}>
        <button onClick={onClose} style={closeBtn}>✕ Close</button>
        <p style={{ color: '#dc3545' }}>Failed to load replay: {error}</p>
      </div>
    )
  }
  if (!detail) {
    return (
      <div style={viewerBox}>
        <button onClick={onClose} style={closeBtn}>✕ Close</button>
        <p>Loading replay…</p>
      </div>
    )
  }

  const totalStates = detail.states.length    // = game_length + 1
  const currentState: EvalReplayState = detail.states[stepIndex]
  // The action that produced currentState (stepIndex 0 = initial; action[i] -> state[i+1])
  const lastAction = stepIndex > 0 ? detail.actions[stepIndex - 1] : null
  const turn = replayTurn(currentState, detail.actions[stepIndex]?.player_id)
  const recordedSha = meta.checkpoint_sha ?? detail.checkpoint_sha256 ?? detail.checkpoint_sha ?? detail.source?.checkpoint_sha256
  const analysisSha = params.get('model_sha') || recordedSha || undefined
  const labQuery = contextQuery(runId, meta.replay_path, stepIndex, analysisSha)

  const dqnSeat = detail.dqn_player_id
  const oppSeat = detail.opponent_player_id

  return (
    <div style={viewerBox}>
      <ResearchContext context={{ position: {}, return_route: '', source: { run: { run_id: runId }, replay_path: meta.replay_path, ply: stepIndex, evaluation: { evaluation_id: meta.evaluation_id } }, recorded_model: recordedSha ? { sha256: recordedSha } : null, analysis_model: analysisSha ? { sha256: analysisSha } : null }} />
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
        <button onClick={onClose} style={closeBtn}>✕ Close</button>
        <span style={{ fontWeight: 600, fontSize: 14 }}>
          Episode {detail.episode ?? '—'} vs {detail.eval_opponent.name}
          {detail.eval_opponent.depth != null ? ` (d${detail.eval_opponent.depth})` : ''}
        </span>
        <span>Agent seat: <strong>{dqnSeat}</strong></span>
        <span>Opp seat: <strong>{oppSeat}</strong></span>
        {resultBadge(detail.result_from_dqn_perspective)}
        {detail.winner
          ? <span style={{ fontSize: 12, color: '#555' }}>Winner: {detail.winner}</span>
          : <span style={{ fontSize: 12, color: '#555' }}>Draw / Timeout</span>}
        <span style={{ fontSize: 12, color: '#777', marginLeft: 'auto' }}>
          Game #{detail.game_index} · {detail.game_length} turns
          {detail.illegal_acts > 0 ? ` · ⚠ ${detail.illegal_acts} illegal` : ''}
        </span>
      </div>

      {/* Board + step controls */}
      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        {/* Board */}
        <div>
          <EvalReplayBoard
            state={currentState}
            lastActionType={lastAction?.action_type}
            lastActionX={lastAction?.x}
            lastActionY={lastAction?.y}
            lastActionPlayer={lastAction?.player_id as 'P1' | 'P2' | undefined}
            size={400}
          />
          {/* Wall counts below board */}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginTop: 4, padding: '0 4px' }}>
            <span style={{ color: '#4a90e2' }}>P1 walls left: {currentState.walls_remaining_p1}</span>
            <span style={{ color: '#e24a4a' }}>P2 walls left: {currentState.walls_remaining_p2}</span>
          </div>
        </div>

        {/* Step controls + action log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minWidth: 200 }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>
            Step {stepIndex} / {totalStates - 1}
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <button aria-label="First position" style={navBtn} onClick={() => { setPlaying(false); setStepIndex(0) }} disabled={stepIndex === 0}>⏮</button>
            <button aria-label="Previous position" style={navBtn} onClick={() => { setPlaying(false); setStepIndex(s => Math.max(0, s - 1)) }} disabled={stepIndex === 0}>◀</button>
            <button aria-label="Next position" style={navBtn} onClick={() => { setPlaying(false); setStepIndex(s => Math.min(totalStates - 1, s + 1)) }} disabled={stepIndex === totalStates - 1}>▶</button>
            <button aria-label="Last position" style={navBtn} onClick={() => { setPlaying(false); setStepIndex(totalStates - 1) }} disabled={stepIndex === totalStates - 1}>⏭</button>
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button style={navBtn} onClick={() => setPlaying(value => !value)} disabled={stepIndex === totalStates - 1}>
              {playing ? 'Pause' : 'Auto play'}
            </button>
            <label style={{ fontSize: 12 }}>Speed {speed}× <input aria-label="Playback speed" type="range" min={0.5} max={4} step={0.5} value={speed} onChange={event => setSpeed(Number(event.target.value))} /></label>
          </div>
          <input aria-label="Replay progress" type="range" min={0} max={totalStates - 1} value={stepIndex}
            onChange={event => { setPlaying(false); setStepIndex(Number(event.target.value)) }} />
          <div className="recorded-identity"><strong>Actual recorded actor</strong><p>{detail.actions[stepIndex]?.actor_name ?? (stepIndex === totalStates - 1 ? 'Terminal position / no next actor' : 'Not recorded')} · {detail.actions[stepIndex]?.player_id ?? '—'}</p><p className="mono sha">Checkpoint: {recordedSha ?? 'SHA not recorded; never inferred from the analysis selection'}</p></div>
          <ModelPicker recordedSha={recordedSha} value={params.get('model_sha') ?? ''} onChange={sha => { const next = new URLSearchParams(params); if (sha) next.set('model_sha', sha); else next.delete('model_sha'); setParams(next, { replace: true }) }} />
          {turn != null ? <Link className="button-link primary" to={`/lab?${labQuery}`}>Analyze exact position in Lab →</Link> : <><button disabled>No recorded turn for analysis</button><p className="muted">No recorded current player or next actor; no turn is invented.</p></>}
          <Link to={replayRoute(runId, { replay: meta.replay_path, ply: stepIndex, checkpoint_sha: recordedSha, model_sha: params.get('model_sha') })}>Permalink · ply {stepIndex}</Link>
          <p style={{ fontSize: 11, color: '#888', margin: 0 }}>Space: pause/play · ← →: step · Home/End: jump</p>

          {/* Last action info */}
          {lastAction && (
            <div style={actionBox}>
              <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 13 }}>
                Turn {lastAction.turn_index}: {lastAction.player_id}
              </div>
              <div style={{ fontSize: 12 }}>
                <span style={{ color: lastAction.is_dqn_action ? '#4a90e2' : '#e24a4a' }}>
                  {lastAction.player_id === dqnSeat ? '🤖 Agent' : `⚔ ${lastAction.actor_name}`}
                </span>
              </div>
              <div style={{ fontSize: 12 }}>
                Action: {lastAction.action_type.toUpperCase()}
                {lastAction.action_type === 'pawn'
                  ? ` → (${lastAction.x}, ${lastAction.y})`
                  : ` @ (${lastAction.x}, ${lastAction.y})`}
              </div>
            </div>
          )}
          {!lastAction && (
            <div style={actionBox}>
              <div style={{ fontSize: 13, color: '#888' }}>Initial position</div>
            </div>
          )}

          {/* Pawn positions */}
          <div style={posBox}>
            <div style={{ fontSize: 12, color: '#4a90e2' }}>
              P1: ({currentState.p1[0]}, {currentState.p1[1]}) — {currentState.walls_remaining_p1}W
            </div>
            <div style={{ fontSize: 12, color: '#e24a4a' }}>
              P2: ({currentState.p2[0]}, {currentState.p2[1]}) — {currentState.walls_remaining_p2}W
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>
              H-walls: {currentState.h_walls.length} · V-walls: {currentState.v_walls.length}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// -------------------------------------------------------------------------
// Main EvalReplaysView
// -------------------------------------------------------------------------

const EvalReplaysView: React.FC<Props> = ({ runId, indexOnly = false, arena = false }) => {
  const [replaysResp, setReplaysResp] = useState<EvalReplaysResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [params, setParams] = useSearchParams()
  const selectedPath = indexOnly ? null : selectedReplay(params)
  const selectedMeta = selectedPath ? replaysResp?.replays.find(meta => meta.replay_path === selectedPath) ?? { replay_path: selectedPath } : null
  const closeReplay = () => { const next = new URLSearchParams(params); next.delete('replay'); next.delete('replay_path'); next.delete('ply'); setParams(next) }
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([])
  const [benchmarkWarning, setBenchmarkWarning] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  // Filtering
  const filterOpponent = params.get('opponent') ?? ''
  const filterResult = params.get('result') ?? ''
  const filterSha = params.get('checkpoint_sha') ?? ''
  const filterCheckpoint = params.get('checkpoint') ?? ''
  const filterEvaluation = params.get('evaluation') ?? ''
  const checkpointGroup = params.get('checkpoint_group')
  const evaluationGroup = params.get('evaluation_group')
  const filter = (key: string, value: string) => { const next = new URLSearchParams(params); next.delete('checkpoint_group'); next.delete('evaluation_group'); if (value) next.set(key, value); else next.delete(key); setParams(next) }

  useEffect(() => {
    let active = true
    setReplaysResp(null); setError(null); setBenchmarks([]); setBenchmarkWarning(null)
    api.evaluations(runId).then(data => {
      if (!Array.isArray(data.replays) || !Array.isArray(data.benchmarks)) throw new Error('Malformed evaluations response: expected benchmarks[] and replays[]')
      if (active) { setReplaysResp({ run_id: runId, replays: data.replays, count: data.replays.length, warnings: data.warnings }); setBenchmarks(data.benchmarks) }
    }).catch(async e => {
      // The platform endpoint can roll out independently of this frontend.
      if (!String(e).startsWith('Error: 404') && !String(e).startsWith('Error: 501')) { if (active) setError(String(e)); return }
      if (active) setBenchmarkWarning('Platform evaluation index unavailable; displaying existing replay and benchmark APIs.')
      const [replays, summaries] = await Promise.allSettled([api.listEvalReplays(runId), api.getBenchmarks(runId)])
      if (!active) return
      if (replays.status === 'fulfilled') setReplaysResp(replays.value)
      else setError(String(replays.reason))
      if (summaries.status === 'fulfilled') { setBenchmarks(summaries.value.benchmarks); setBenchmarkWarning(['Platform index unavailable; existing APIs used.', ...(summaries.value.warnings ?? [])].join(' ')) }
      else setBenchmarkWarning(`Aggregate summaries unavailable: ${summaries.reason}`)
    })
    return () => { active = false }
  }, [runId, retry])

  const viewer = selectedMeta && <div style={{ marginBottom: 20 }}><ReplayViewer
    key={`${runId}-${selectedMeta.replay_path}`}
    runId={runId}
    meta={selectedMeta}
    onClose={closeReplay}
  /></div>

  // Build filter options
  const replayRows = replaysResp?.replays ?? []
  const replayCount = replaysResp?.count ?? 0
  const opponents = Array.from(new Set(replayRows.map(r => r.opponent_name ?? r.opponent_type ?? '').filter(Boolean)))
  const results = ['win', 'loss', 'draw']

  const filtered = replayRows.filter(r => {
    if (filterSha && r.checkpoint_sha !== filterSha) return false
    if (filterCheckpoint && r.checkpoint !== filterCheckpoint) return false
    if (filterEvaluation && r.evaluation_id !== filterEvaluation) return false
    if (filterOpponent && (r.opponent_name ?? r.opponent_type) !== filterOpponent) return false
    if (filterResult && r.result_from_dqn_perspective !== filterResult) return false
    return true
  })
  const filteredBenchmarks = benchmarks.filter(b => (!filterOpponent || b.opponent === filterOpponent) && (!filterSha || b.checkpoint_sha === filterSha) && (!filterCheckpoint || b.checkpoint === filterCheckpoint) && (!filterEvaluation || b.evaluation_id === filterEvaluation))
  // Resolve legacy partial replay metadata against summaries before selecting an
  // exact group. Filtering raw replay fields would discard valid unique joins.
  const groups = evaluationGroups(filtered, filteredBenchmarks)
    .filter(checkpoint => !checkpointGroup || checkpoint.key === checkpointGroup)
    .map(checkpoint => ({ ...checkpoint, groups: checkpoint.groups.filter(group => !evaluationGroup || group.key === evaluationGroup) }))
    .filter(checkpoint => checkpoint.groups.length > 0)
  const visibleBenchmarks = groups.flatMap(checkpoint => checkpoint.groups.flatMap(group => group.summaries))
  const visibleGames = groups.reduce((count, checkpoint) => count + checkpoint.groups.reduce((n, group) => n + group.games.length, 0), 0)

  return (
    <div style={{ padding: '8px 0' }}>
      <p className="muted">Recorded games only. All groups start collapsed. Aggregate evaluation summaries are shown when available; indexed replay samples are not full-evaluation W/D/L. Missing trajectories cannot be regenerated.</p>
      {(replaysResp?.warnings ?? []).map((w, i) => <div className="warning" key={i}>{w}</div>)}
      {benchmarkWarning && <div className="warning">{benchmarkWarning}</div>}
      {/* Viewer overlay */}
      {viewer}
      {error && <div className="error" role="alert">Error loading replay index: {error}<div><button onClick={() => setRetry(value => value + 1)}>Retry replay index</button></div></div>}
      {!replaysResp && !error && <div className="loading" role="status">Loading evaluation replay index…</div>}

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#555' }}>
          {replayCount} replay{replayCount !== 1 ? 's' : ''}
        </span>
        <select
          value={filterOpponent}
          onChange={e => filter('opponent', e.target.value)}
          style={selectStyle}
        >
          <option value="">All opponents</option>
          {opponents.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <select
          value={filterResult}
          onChange={e => filter('result', e.target.value)}
          style={selectStyle}
        >
          <option value="">All results</option>
          {results.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
        {(filterOpponent || filterResult || filterSha || filterCheckpoint || filterEvaluation || checkpointGroup || evaluationGroup) && (
          <button
            onClick={() => { const next = new URLSearchParams(params); ['opponent', 'result', 'checkpoint_sha', 'checkpoint', 'evaluation', 'checkpoint_group', 'evaluation_group'].forEach(key => next.delete(key)); setParams(next) }}
            style={{ fontSize: 12, padding: '3px 8px', cursor: 'pointer' }}
          >Clear filters</button>
        )}
        <span style={{ fontSize: 12, color: '#888', marginLeft: 'auto' }}>
          Showing {visibleGames} of {replayCount}
        </span>
      </div>

      {(filterSha || filterCheckpoint || filterEvaluation) && <p className="muted sha">Exact filter: {filterSha || filterCheckpoint} {filterEvaluation && `· evaluation ${filterEvaluation}`}. Unknown checkpoint identities are not treated as matches.</p>}

      {arena && replaysResp && <section><h3>Aggregate evaluation results · {visibleBenchmarks.length}</h3><EvaluationTable rows={visibleBenchmarks} /><p className="muted">{visibleGames} matching recorded replay samples. Expand the evidence hierarchy below to inspect individual games.</p></section>}

      {replaysResp && !groups.length && <div className="empty">No recorded evaluation games or summaries match these filters.</div>}
      <details className="replay-group" open={evaluationGroup ? true : undefined}>
        <summary>Run: {runId} · {visibleGames} indexed games · {visibleBenchmarks.length} aggregate summaries</summary>
        {groups.map(checkpoint => <details className="replay-group" key={checkpoint.key} open={evaluationGroup ? true : undefined}>
          <summary>{checkpoint.label}</summary>
          {checkpoint.groups.map(group => <details className="replay-group" key={group.key} open={evaluationGroup ? true : undefined}>
            <summary>{group.label} · {group.summaries.length ? group.summaries.map(row => `Aggregate: ${row.games ?? '—'} games · W/D/L ${row.wins ?? '—'}/${row.draws ?? '—'}/${row.losses ?? '—'}`).join(' · ') : 'Aggregate W/D/L unknown'} · {group.games.length} sampled replay records</summary>
            <EvaluationTable rows={group.summaries} />
            {!group.summaries.length && <p className="muted">No matching aggregate summary. These games may be sampled replays; full evaluation W/D/L is unknown.</p>}
            {!group.games.length && <p className="muted">No indexed replay trajectories for this evaluation.</p>}
      <div style={{ overflowX: 'auto' }}>
        <table style={tableStyle}>
          <thead>
            <tr style={{ background: '#f0f0f0' }}>
              <th style={th}>Episode</th>
              <th style={th}>Opponent</th>
              <th style={th}>Agent Seat</th>
              <th style={th}>Game #</th>
              <th style={th}>Result</th>
              <th style={th}>Winner</th>
              <th style={th}>Length</th>
              <th style={th}></th>
            </tr>
          </thead>
          <tbody>
            {group.games.map((r, i) => (
              <tr
                key={i}
                style={{
                  background: selectedMeta?.replay_path === r.replay_path ? '#e8f4ff' : i % 2 === 0 ? '#fff' : '#fafafa',
                  cursor: 'default',
                }}
              >
                <td style={td}>{r.episode ?? '—'}</td>
                <td style={td}>{formatOpponent(r)}</td>
                <td style={td}>
                  <span style={{ color: r.dqn_player_id === 'P1' ? '#4a90e2' : '#e24a4a', fontWeight: 600 }}>
                    {r.dqn_player_id ?? '—'}
                  </span>
                </td>
                <td style={td}>{r.game_id ?? r.game_index ?? '—'}</td>
                <td style={td}>{resultBadge(r.result_from_dqn_perspective)}</td>
                <td style={td}>{r.winner ?? (r.result_from_dqn_perspective === 'draw' ? 'Draw' : 'Unknown')}</td>
                <td style={td}>{r.game_length ?? '—'}</td>
                <td style={td}>
                  {r.replay_available === false ? <button disabled title={r.replay_unavailable_reason ?? undefined}>Unavailable</button> : <Link className="button-link" to={replayRoute(runId, { replay: r.replay_path, ply: 0, model_sha: params.get('model_sha') })}>View recorded game</Link>}
                  {r.replay_available === false && <div className="muted">{r.replay_unavailable_reason ?? 'Full trajectory not available; playback disabled.'}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
          </details>)}
        </details>)}
      </details>
    </div>
  )
}

// -------------------------------------------------------------------------
// Styles
// -------------------------------------------------------------------------

const viewerBox: React.CSSProperties = {
  background: '#fff',
  border: '1px solid #ddd',
  borderRadius: 8,
  padding: 16,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
}

const closeBtn: React.CSSProperties = {
  padding: '4px 10px',
  fontSize: 13,
  cursor: 'pointer',
  background: '#6c757d',
  color: '#fff',
  border: 'none',
  borderRadius: 4,
}

const navBtn: React.CSSProperties = {
  padding: '4px 10px',
  fontSize: 16,
  cursor: 'pointer',
  background: '#4a90e2',
  color: '#fff',
  border: 'none',
  borderRadius: 4,
  minWidth: 36,
}

const actionBox: React.CSSProperties = {
  background: '#f8f9fa',
  border: '1px solid #e9ecef',
  borderRadius: 6,
  padding: '8px 10px',
}

const posBox: React.CSSProperties = {
  background: '#f8f9fa',
  border: '1px solid #e9ecef',
  borderRadius: 6,
  padding: '8px 10px',
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
}

const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: 13,
}

const th: React.CSSProperties = {
  padding: '6px 10px',
  textAlign: 'left',
  fontWeight: 600,
  borderBottom: '2px solid #ddd',
  whiteSpace: 'nowrap',
}

const td: React.CSSProperties = {
  padding: '5px 10px',
  borderBottom: '1px solid #eee',
}

const viewBtn: React.CSSProperties = {
  padding: '2px 8px',
  fontSize: 12,
  cursor: 'pointer',
  background: '#4a90e2',
  color: '#fff',
  border: 'none',
  borderRadius: 3,
}

const selectStyle: React.CSSProperties = {
  fontSize: 12,
  padding: '3px 6px',
  borderRadius: 4,
  border: '1px solid #ccc',
}

export default EvalReplaysView
