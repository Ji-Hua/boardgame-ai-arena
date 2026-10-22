import type { GameStateWire } from '../../../frontend/src/api/roomAPI'
import { StateMapper } from '../../../frontend/src/core/StateMapper'
import { ActionTranslator } from '../../../frontend/src/core/ActionTranslator'
import type { Action } from '../../../frontend/src/types/Action'

export interface PlaySnapshot {
  game_id: string
  room_id: string
  phase: string
  step_count: number
  human_seat: 1 | 2
  pending: boolean
  error: string | null
  state: GameStateWire & { walls: { x: number; y: number; orientation: string }[]; wall_state: Record<string, string> }
  legal_actions: { player: 1 | 2; type: string; target: [number, number] }[]
  actions: { player: 1 | 2; type: string; target: [number, number] }[]
  identity: { name: string; architecture: string; algorithm: string; sha256: string; actual_sha256: string; output_semantics: string; runtime_version: string; actual_load_verified: boolean }
  result: { winner_seat: 1 | 2 | null; termination: 'goal' | 'surrender' | 'timeout' | 'forced' } | null
}

export function projectPlay(snapshot: PlaySnapshot) {
  const last = snapshot.actions[snapshot.actions.length - 1]
  return StateMapper.toRenderState({ gameState: snapshot.state,
    lastAction: last ? { seat: last.player, type: last.type as Action['type'], position: last.target } : null,
    walls: snapshot.state.walls.map(wall => ({ row: wall.y, col: wall.x, orientation: wall.orientation === 'horizontal' ? 'H' : 'V' })),
    stepCount: snapshot.step_count,
    actors: { 1: snapshot.human_seat === 1 ? 'human' : 'agent', 2: snapshot.human_seat === 2 ? 'human' : 'agent' },
    result: snapshot.result ? { winnerSeat: snapshot.result.winner_seat, termination: snapshot.result.termination, surrenderBySeat: null } : null })
}

const key = (game: string) => `quoridor-play:${game}`
export class ResearchPlayController {
  async request(path: string, method = 'GET', body?: unknown, token?: string) {
    const response = await fetch('/play-api/games' + path, { method,
      headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(45000) })
    const data = await response.json()
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
    return data
  }
  async start(sha256: string, human_seat: 1 | 2): Promise<PlaySnapshot> {
    const data = await this.request('', 'POST', { sha256, human_seat })
    try { localStorage.setItem(key(data.game_id), data.token); localStorage.setItem('quoridor-play-active', data.game_id) }
    catch { throw new Error('Browser storage is required to retain this game capability. Enable local storage before starting.') }
    return data
  }
  token(game: string) {
    const token = localStorage.getItem(key(game))
    if (!token) throw new Error('This browser has no capability for this game. No replacement game was created.')
    return token
  }
  snapshot(game: string): Promise<PlaySnapshot> { return this.request(`/${game}`, 'GET', undefined, this.token(game)) }
  async action(snapshot: PlaySnapshot, action: Action): Promise<PlaySnapshot> {
    return this.request(`/${snapshot.game_id}/actions`, 'POST', { expected_step: snapshot.step_count, type: action.type, target: action.position }, this.token(snapshot.game_id))
  }
  pawn(snapshot: PlaySnapshot, row: number, col: number) {
    return this.action(snapshot, ActionTranslator.translatePawnMove(snapshot.human_seat, [col, row]))
  }
  wall(snapshot: PlaySnapshot, orientation: 'H' | 'V', row: number, col: number) {
    return this.action(snapshot, ActionTranslator.translateWallPlacement(snapshot.human_seat, orientation, [col, row]))
  }
  command(game: string, command: 'retry' | 'surrender'): Promise<PlaySnapshot> {
    return this.request(`/${game}/${command}`, 'POST', {}, this.token(game))
  }
  async close(game: string) {
    await this.request(`/${game}`, 'DELETE', undefined, this.token(game))
    localStorage.removeItem(key(game))
    if (localStorage.getItem('quoridor-play-active') === game) localStorage.removeItem('quoridor-play-active')
  }
}