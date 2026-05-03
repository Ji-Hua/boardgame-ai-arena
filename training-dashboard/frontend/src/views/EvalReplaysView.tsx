/**
 * EvalReplaysView — shows a list of saved evaluation game replays for a run
 * and provides an inline step-by-step replay viewer with a Quoridor board.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../api'
import type { EvalReplayMeta, EvalReplayDetail, EvalReplayState, EvalReplaysResponse } from '../types'
import { EvalReplayBoard } from '../components/EvalReplayBoard'

interface Props {
  runId: string
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

function resultBadge(result: string | null) {
  if (result === 'win') return <span style={badge('green')}>WIN</span>
  if (result === 'loss') return <span style={badge('red')}>LOSS</span>
  return <span style={badge('gray')}>DRAW</span>
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

const ReplayViewer: React.FC<ReplayViewerProps> = ({ runId, meta, onClose }) => {
  const [detail, setDetail] = useState<EvalReplayDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stepIndex, setStepIndex] = useState(0)   // index into detail.states (0 = initial)

  useEffect(() => {
    setDetail(null)
    setError(null)
    setStepIndex(0)
    api.getEvalReplay(runId, meta.replay_path)
      .then(setDetail)
      .catch((e: Error) => setError(e.message))
  }, [runId, meta.replay_path])

  const handleKey = useCallback((e: KeyboardEvent) => {
    if (!detail) return
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

  const dqnSeat = detail.dqn_player_id
  const oppSeat = detail.opponent_player_id

  return (
    <div style={viewerBox}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
        <button onClick={onClose} style={closeBtn}>✕ Close</button>
        <span style={{ fontWeight: 600, fontSize: 14 }}>
          Ep {detail.episode} vs {detail.eval_opponent.name}
          {detail.eval_opponent.depth != null ? ` (d${detail.eval_opponent.depth})` : ''}
        </span>
        <span>DQN seat: <strong>{dqnSeat}</strong></span>
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
            <button style={navBtn} onClick={() => setStepIndex(0)} disabled={stepIndex === 0}>⏮</button>
            <button style={navBtn} onClick={() => setStepIndex(s => Math.max(0, s - 1))} disabled={stepIndex === 0}>◀</button>
            <button style={navBtn} onClick={() => setStepIndex(s => Math.min(totalStates - 1, s + 1))} disabled={stepIndex === totalStates - 1}>▶</button>
            <button style={navBtn} onClick={() => setStepIndex(totalStates - 1)} disabled={stepIndex === totalStates - 1}>⏭</button>
          </div>
          <p style={{ fontSize: 11, color: '#888', margin: 0 }}>← → arrow keys also work</p>

          {/* Last action info */}
          {lastAction && (
            <div style={actionBox}>
              <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 13 }}>
                Turn {lastAction.turn_index}: {lastAction.player_id}
              </div>
              <div style={{ fontSize: 12 }}>
                <span style={{ color: lastAction.is_dqn_action ? '#4a90e2' : '#e24a4a' }}>
                  {lastAction.is_dqn_action ? '🤖 DQN' : `⚔ ${lastAction.actor_name}`}
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

  // Filtering
  const [filterOpponent, setFilterOpponent] = useState<string>('')
  const [filterResult, setFilterResult] = useState<string>('')

  useEffect(() => {
    api.listEvalReplays(runId)
      .then(setReplaysResp)
      .catch((e: Error) => setError(e.message))
  }, [runId])

  if (error) {
    return <div style={{ color: '#dc3545', padding: 16 }}>Error loading replays: {error}</div>
  }
  if (!replaysResp) {
    return <div style={{ padding: 16, color: '#888' }}>Loading evaluation replays…</div>
  }
  if (replaysResp.count === 0) {
    return (
      <div style={{ padding: 16, color: '#888' }}>
        <p>No evaluation replays found for this run.</p>
        <p style={{ fontSize: 13 }}>
          Replays are saved during periodic evaluation when <code>replay_sample_every</code> is
          set in the <code>evaluation</code> section of the training config.
        </p>
      </div>
    )
  }

  // Build filter options
  const opponents = Array.from(new Set(replaysResp.replays.map(r => r.opponent_name ?? r.opponent_type ?? '').filter(Boolean)))
  const results = ['win', 'loss', 'draw']

  const filtered = replaysResp.replays.filter(r => {
    if (filterOpponent && (r.opponent_name ?? r.opponent_type) !== filterOpponent) return false
    if (filterResult && r.result_from_dqn_perspective !== filterResult) return false
    return true
  })

  return (
    <div style={{ padding: '8px 0' }}>
      {/* Viewer overlay */}
      {selectedMeta && (
        <div style={{ marginBottom: 20 }}>
          <ReplayViewer
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

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={tableStyle}>
          <thead>
            <tr style={{ background: '#f0f0f0' }}>
              <th style={th}>Episode</th>
              <th style={th}>Opponent</th>
              <th style={th}>DQN Seat</th>
              <th style={th}>Game #</th>
              <th style={th}>Result</th>
              <th style={th}>Winner</th>
              <th style={th}>Length</th>
              <th style={th}></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
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
                <td style={td}>{r.game_index ?? '—'}</td>
                <td style={td}>{resultBadge(r.result_from_dqn_perspective)}</td>
                <td style={td}>{r.winner ?? 'Draw'}</td>
                <td style={td}>{r.game_length ?? '—'}</td>
                <td style={td}>
                  <button
                    onClick={() => setSelectedMeta(selectedMeta?.replay_path === r.replay_path ? null : r)}
                    style={viewBtn}
                  >
                    {selectedMeta?.replay_path === r.replay_path ? 'Hide' : 'View'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
