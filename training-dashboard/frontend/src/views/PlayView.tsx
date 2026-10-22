import React, { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Flag, MousePointer2, RectangleHorizontal, RefreshCw, X } from 'lucide-react'
import { Board, type WallPreview } from '../../../../frontend/src/components/board/Board'
import { PLAYER_COLORS } from '../../../../frontend/src/theme/playerColors'
import { api } from '../api'
import { projectPlay, ResearchPlayController, type PlaySnapshot } from '../play'
import type { PlatformModel } from '../platform'
import './PlayView.css'

export function PlayView() {
  const [params, setParams] = useSearchParams()
  const sha = params.get('model_sha') ?? ''
  const game = params.get('game') ?? ''
  const [controller] = useState(() => new ResearchPlayController())
  const [models, setModels] = useState<PlatformModel[]>([])
  const [snapshot, setSnapshot] = useState<PlaySnapshot | null>(null)
  const [seat, setSeat] = useState<1 | 2>(1)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [wallMode, setWallMode] = useState(false)
  const [preview, setPreview] = useState<WallPreview | null>(null)
  const [cellSize, setCellSize] = useState(48)
  const boardArea = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let active = true
    api.models().then(data => { if (active) setModels(data.models) }).catch(e => { if (active) setError(String(e)) })
    return () => { active = false }
  }, [])
  useEffect(() => {
    const element = boardArea.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setCellSize(Math.min(54, Math.max(18, (entry.contentRect.width - 36) / 9))))
    observer.observe(element)
    return () => observer.disconnect()
  }, [Boolean(snapshot)])
  useEffect(() => {
    if (!game) { setSnapshot(null); return }
    setSnapshot(null)
    let active = true
    let timer: ReturnType<typeof setTimeout>
    async function refresh() {
      try {
        const next = await controller.snapshot(game)
        if (active) { setSnapshot(next); setError('') }
      } catch (e) { if (active) setError(String(e)) }
      if (active) timer = setTimeout(refresh, 1200)
    }
    void refresh()
    return () => { active = false; clearTimeout(timer) }
  }, [game, controller])
  const selected = models.find(model => model.sha256 === sha)
  const updateSha = (value: string) => { const next = new URLSearchParams(params); next.set('model_sha', value); setParams(next) }
  async function perform(operation: () => Promise<PlaySnapshot>) {
    setBusy(true); setError('')
    try { setSnapshot(await operation()); setWallMode(false); setPreview(null) }
    catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }
  async function start() {
    setBusy(true); setError('')
    try {
      const data = await controller.start(sha, seat)
      setSnapshot(data)
      const next = new URLSearchParams(params); next.set('game', data.game_id); setParams(next)
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }
  async function close() {
    if (!snapshot) return
    setBusy(true)
    try { await controller.close(snapshot.game_id); const next = new URLSearchParams(params); next.delete('game'); setParams(next); setSnapshot(null) }
    catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }
  const render = snapshot ? projectPlay(snapshot) : null
  const humanTurn = Boolean(snapshot && snapshot.phase === 'running' && !snapshot.pending && snapshot.state.current_player === snapshot.human_seat && !busy)
  const legalMoves: [number, number][] = humanTurn && snapshot ? snapshot.legal_actions.filter(a => a.type === 'pawn').map(a => [a.target[1], a.target[0]]) : []
  const identity = snapshot?.identity
  let resumable = ''
  try { resumable = localStorage.getItem('quoridor-play-active') ?? '' } catch { /* Optional discovery only. */ }
  return <div className="unified-play">
    <div className="page-heading"><div><p className="eyebrow">Research / Play</p><h1>Human vs Checkpoint</h1></div><Link to="/models">Model library</Link></div>
    {error && <div className="error" role="alert">{error}</div>}
    {game && !snapshot && <div className="loading">Reconnecting to game {game}...</div>}
    {!game && <section className="play-setup">
      <label>Checkpoint<select aria-label="Play checkpoint" value={sha} onChange={e => updateSha(e.target.value)} disabled={busy}>
        <option value="">Select an approved CNN</option>
        {sha && !selected && <option value={sha}>{sha} - identity unavailable</option>}
        {models.map(model => <option key={model.sha256} value={model.sha256} disabled={!model.research_playable}>{model.aliases.join(' / ')} - {model.sha256}{!model.research_playable ? ` - unavailable: ${(model.reasons.research_playable ?? []).join(', ')}` : ''}</option>)}
      </select></label>
      {selected && <><p>{selected.aliases.join(' / ')} · {selected.architecture} · {String(selected.metadata['model_config.algorithm'] ?? 'Unknown algorithm')} · {selected.output_semantics}</p><p className="mono sha" data-testid="selected-sha">{sha}</p>{!selected.research_playable && <p className="warning">{(selected.reasons.research_playable ?? []).join(' · ')}</p>}</>}
      <div className="action-bar"><label>Your seat<select aria-label="Your seat" value={seat} onChange={e => setSeat(Number(e.target.value) as 1 | 2)} disabled={busy}><option value="1">P1 - first</option><option value="2">P2 - second</option></select></label><button disabled={busy || !selected?.research_playable} onClick={start}>{busy ? 'Loading exact checkpoint...' : 'Start game'}</button>{resumable && <Link to={`/play?game=${encodeURIComponent(resumable)}`}>Return to active game</Link>}</div>
    </section>}
    {snapshot && render && <>
      <section className="play-identity"><h2>{identity?.name}</h2><p>{identity?.architecture} · {identity?.algorithm} · {identity?.output_semantics}</p><p className="mono sha" data-testid="actual-sha">{identity?.actual_sha256}</p><p className="muted">{identity?.actual_load_verified ? 'Loaded SHA verified' : 'Load unverified'} · You: P{snapshot.human_seat} · Runtime {identity?.runtime_version}</p>{sha && sha !== identity?.sha256 && <p className="warning">This game remains pinned to its original checkpoint. The URL selection applies only to the next game.</p>}</section>
      <div className="play-layout">
        <div className="play-board-area" ref={boardArea}>
          <div className="play-seat"><strong>P2 {snapshot.human_seat === 2 ? 'You' : 'CNN'}</strong><span>Walls: {render.wallsRemaining[2]}</span></div>
          <div className="play-board" data-testid="live-board"><Board cellSize={cellSize} mode={humanTurn ? 'play' : 'replay'}
            pawns={([1, 2] as const).map(player => ({ playerId: player === 1 ? 'P1' : 'P2', row: render.pawns[player][0], col: render.pawns[player][1], color: PLAYER_COLORS[player === 1 ? 'P1' : 'P2'].primary }))}
            walls={render.walls.map(wall => ({ row: wall.row, col: wall.col, orientation: wall.orientation, color: '#59615d' }))}
            currentPlayer={render.currentSeat === 1 ? 'P1' : 'P2'} legalMoves={wallMode ? [] : legalMoves}
            wallPlacement={{ isActive: humanTurn && wallMode, playerId: snapshot.human_seat === 1 ? 'P1' : 'P2' }}
            wallPreview={preview} onUpdateWallPreview={setPreview}
            onMovePawn={(row, col) => { if (humanTurn) void perform(() => controller.pawn(snapshot, row, col)) }}
            onConfirmWall={wall => { if (humanTurn) void perform(() => controller.wall(snapshot, wall.orientation === 'Horizontal' ? 'H' : 'V', wall.logicalRow, wall.logicalCol)) }}
            onCancelWallPlacement={() => { setWallMode(false); setPreview(null) }} />
          </div>
          <div className="play-seat"><strong>P1 {snapshot.human_seat === 1 ? 'You' : 'CNN'}</strong><span>Walls: {render.wallsRemaining[1]}</span></div>
        </div>
        <aside className="play-status">
          <h2 data-testid="game-status">{snapshot.result ? `P${snapshot.result.winner_seat ?? '-'} wins` : snapshot.pending ? 'CNN deciding...' : humanTurn ? 'Your turn' : 'Agent turn'}</h2>
          <p data-testid="step-count">Ply {snapshot.step_count}</p>{snapshot.result && <p>{snapshot.result.termination}</p>}
          <div className="play-tools" role="group" aria-label="Action mode"><button title="Move pawn" aria-label="Move pawn" aria-pressed={!wallMode} disabled={!humanTurn} onClick={() => setWallMode(false)}><MousePointer2 size={19} /></button><button title="Place wall" aria-label="Place wall" aria-pressed={wallMode} disabled={!humanTurn || render.wallsRemaining[snapshot.human_seat] === 0} onClick={() => setWallMode(true)}><RectangleHorizontal size={19} /></button></div>
          {snapshot.error && <div className="error" role="alert">{snapshot.error}{snapshot.phase === 'running' && <button disabled={busy || snapshot.pending} onClick={() => void perform(() => controller.command(game, 'retry'))}><RefreshCw size={16} /> Retry Agent</button>}</div>}
          {snapshot.phase === 'running' && <button disabled={busy || snapshot.pending} onClick={() => { if (window.confirm('Surrender this game?')) void perform(() => controller.command(game, 'surrender')) }}><Flag size={16} /> Surrender</button>}
          <button disabled={busy || snapshot.pending} onClick={() => { if (snapshot.phase !== 'running' || window.confirm('Close this game?')) void close() }}><X size={16} /> Close game</button>
          <details><summary>Game identity & actions</summary><pre data-testid="game-evidence">{JSON.stringify(snapshot, null, 2)}</pre></details>
        </aside>
      </div>
    </>}
  </div>
}