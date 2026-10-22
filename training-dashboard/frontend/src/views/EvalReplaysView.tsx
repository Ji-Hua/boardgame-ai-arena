/**
 * EvalReplaysView — shows a list of saved evaluation game replays for a run
 * and provides an inline step-by-step replay viewer with a Quoridor board.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../api'
import type { BenchmarkResult, EvalReplayMeta, EvalReplayDetail, EvalReplayState, EvalReplaysResponse } from '../types'
import { EvalReplayBoard } from '../components/EvalReplayBoard'
import { evaluationGroups, replayTurn } from '../runData'
import { EvaluationTable } from './BenchmarkView'

interface Props {
  runId: string
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
  meta: EvalReplayMeta
  onClose: () => void
}

export const ReplayViewer: React.FC<ReplayViewerProps> = ({ runId, meta, onClose }) => {
  const [detail, setDetail] = useState<EvalReplayDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stepIndex, setStepIndex] = useState(0)   // index into detail.states (0 = initial)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

  useEffect(() => {
    setDetail(null)
    setError(null)
    setStepIndex(0)
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
        setPlaying(true)
      })
      .catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [runId, meta.replay_path])

  useEffect(() => {
    if (!detail || !playing) return
    const timer = window.setInterval(() => {
      setStepIndex(current => {
        if (current >= detail.states.length - 1) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 1000 / speed)
    return () => window.clearInterval(timer)
  }, [detail, playing, speed])

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
  }, [detail])

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
  const openDecisionLab = () => {
    if (turn == null) return
    const position = {
      pawns: [currentState.p1, currentState.p2],
      walls_remaining: [currentState.walls_remaining_p1, currentState.walls_remaining_p2],
      walls: [
        ...currentState.h_walls.map(([x, y]) => ({ x, y, orientation: 'horizontal' })),
        ...currentState.v_walls.map(([x, y]) => ({ x, y, orientation: 'vertical' })),
      ],
      turn,
    }
    const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(position))))
    window.open(`http://${window.location.hostname}:8744/?position=${encodeURIComponent(encoded)}`, '_blank', 'noopener')
  }

  const dqnSeat = detail.dqn_player_id
  const oppSeat = detail.opponent_player_id

  return (
    <div style={viewerBox}>
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
          <button style={navBtn} onClick={openDecisionLab} disabled={turn == null} title={turn == null ? 'No recorded current player or next actor' : `Recorded turn: P${turn}`}>Analyze Decision</button>
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

const EvalReplaysView: React.FC<Props> = ({ runId }) => {
  const [replaysResp, setReplaysResp] = useState<EvalReplaysResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedMeta, setSelectedMeta] = useState<EvalReplayMeta | null>(null)
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([])
  const [benchmarkWarning, setBenchmarkWarning] = useState<string | null>(null)

  // Filtering
  const [filterOpponent, setFilterOpponent] = useState<string>('')
  const [filterResult, setFilterResult] = useState<string>('')

  useEffect(() => {
    let active = true
    setReplaysResp(null); setSelectedMeta(null); setError(null); setBenchmarks([]); setBenchmarkWarning(null)
    setFilterOpponent(''); setFilterResult('')
    api.listEvalReplays(runId)
      .then(r => { if (active) setReplaysResp(r) })
      .catch((e: Error) => { if (active) setError(e.message) })
    api.getBenchmarks(runId).then(r => { if (active) { setBenchmarks(r.benchmarks); setBenchmarkWarning(r.warnings?.join('\n') || null) } })
      .catch(e => { if (active) setBenchmarkWarning(`Aggregate summaries unavailable: ${e}`) })
    return () => { active = false }
  }, [runId])

  if (error) {
    return <div style={{ color: '#dc3545', padding: 16 }}>Error loading replays: {error}</div>
  }
  if (!replaysResp) {
    return <div style={{ padding: 16, color: '#888' }}>Loading evaluation replays…</div>
  }

  // Build filter options
  const opponents = Array.from(new Set(replaysResp.replays.map(r => r.opponent_name ?? r.opponent_type ?? '').filter(Boolean)))
  const results = ['win', 'loss', 'draw']

  const filtered = replaysResp.replays.filter(r => {
    if (filterOpponent && (r.opponent_name ?? r.opponent_type) !== filterOpponent) return false
    if (filterResult && r.result_from_dqn_perspective !== filterResult) return false
    return true
  })
  const groups = evaluationGroups(filtered, benchmarks.filter(b => !filterOpponent || b.opponent === filterOpponent))

  return (
    <div style={{ padding: '8px 0' }}>
      <p className="muted">Recorded games only. All groups start collapsed. Aggregate evaluation summaries are shown when available; indexed replay samples are not full-evaluation W/D/L. Missing trajectories cannot be regenerated.</p>
      {(replaysResp.warnings ?? []).map((w, i) => <div className="warning" key={i}>{w}</div>)}
      {benchmarkWarning && <div className="warning">{benchmarkWarning}</div>}
      {/* Viewer overlay */}
      {selectedMeta && (
        <div style={{ marginBottom: 20 }}>
          <ReplayViewer
            key={`${runId}-${selectedMeta.replay_path}`}
            runId={runId}
            meta={selectedMeta}
            onClose={() => setSelectedMeta(null)}
          />
        </div>
      )}

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#555' }}>
          {replaysResp.count} replay{replaysResp.count !== 1 ? 's' : ''}
        </span>
        <select
          value={filterOpponent}
          onChange={e => setFilterOpponent(e.target.value)}
          style={selectStyle}
        >
          <option value="">All opponents</option>
          {opponents.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <select
          value={filterResult}
          onChange={e => setFilterResult(e.target.value)}
          style={selectStyle}
        >
          <option value="">All results</option>
          {results.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
        {(filterOpponent || filterResult) && (
          <button
            onClick={() => { setFilterOpponent(''); setFilterResult('') }}
            style={{ fontSize: 12, padding: '3px 8px', cursor: 'pointer' }}
          >Clear filters</button>
        )}
        <span style={{ fontSize: 12, color: '#888', marginLeft: 'auto' }}>
          Showing {filtered.length} of {replaysResp.count}
        </span>
      </div>

      {!groups.length && <div className="empty">No recorded evaluation games or summaries available.</div>}
      <details className="replay-group">
        <summary>Run: {runId} · {filtered.length} indexed games · {benchmarks.length} aggregate summaries</summary>
        {groups.map(checkpoint => <details className="replay-group" key={checkpoint.key}>
          <summary>{checkpoint.label}</summary>
          {checkpoint.groups.map(group => <details className="replay-group" key={group.key}>
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
                  <button
                    disabled={r.replay_available === false}
                    title={r.replay_unavailable_reason ?? undefined}
                    onClick={() => setSelectedMeta(selectedMeta?.replay_path === r.replay_path ? null : r)}
                    style={viewBtn}
                  >
                    {r.replay_available === false ? 'Unavailable' : selectedMeta?.replay_path === r.replay_path ? 'Hide' : 'View recorded game'}
                  </button>
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
