import React from 'react'
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { projectPlay, ResearchPlayController, type PlaySnapshot } from '../src/play'
import { PlayView } from '../src/views/PlayView'

afterEach(() => { cleanup(); localStorage.clear(); vi.unstubAllGlobals(); vi.restoreAllMocks() })
const sha = 'a'.repeat(64)
test('safe coordinate projection retains P2, walls and wall stocks', () => {
  const snapshot = { state: { current_player: 2, pawns: { '1': { row: 4, col: 3 }, '2': { row: 4, col: 5 } }, walls_remaining: { '1': 9, '2': 8 }, walls: [{ x: 7, y: 7, orientation: 'vertical' }], game_over: false }, actions: [], human_seat: 2, step_count: 7, result: null } as unknown as PlaySnapshot
  const state = projectPlay(snapshot)
  expect(state.pawns).toEqual({ 1: [3, 4], 2: [5, 4] })
  expect(state.walls).toEqual([{ row: 7, col: 7, orientation: 'V' }])
  expect(state.wallsRemaining).toEqual({ 1: 9, 2: 8 })
  expect(state.currentSeat).toBe(2)
  expect(state.actors).toEqual({ 1: 'agent', 2: 'human' })
})
test('controller converts UI row col to engine x y and sends revision', async () => {
  localStorage.setItem('quoridor-play:game', 'capability')
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) })
  vi.stubGlobal('fetch', fetch)
  const controller = new ResearchPlayController()
  await controller.wall({ game_id: 'game', step_count: 7, human_seat: 2 } as PlaySnapshot, 'V', 6, 1)
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ expected_step: 7, type: 'vertical', target: [1, 6] })
  expect(fetch.mock.calls[0][1].headers.Authorization).toBe('Bearer capability')
})
test('Play deep link selects exact SHA and never starts automatically', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ models: [{ sha256: sha, aliases: ['C-resnet_seed-42'], architecture: 'cnn', metadata: { 'model_config.algorithm': 'policy_distillation' }, research_playable: true, reasons: {}, output_semantics: 'policy_logits' }] }) }))
  render(<MemoryRouter initialEntries={[`/play?model_sha=${sha}`]}><PlayView /></MemoryRouter>)
  expect((await screen.findByLabelText('Play checkpoint') as HTMLSelectElement).value).toBe(sha)
  expect(await screen.findByTestId('selected-sha')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Start game' }).hasAttribute('disabled')).toBe(false)
  expect(vi.mocked(fetch).mock.calls.every(call => !call[1] || call[1].method !== 'POST')).toBe(true)
})
test('reconnecting to a pinned game does not depend on model catalog requests', () => {
  const fetch = vi.fn().mockRejectedValue(new Error('Offline catalog'))
  vi.stubGlobal('fetch', fetch)
  const snapshot = vi.spyOn(ResearchPlayController.prototype, 'snapshot').mockImplementation(() => new Promise(() => {}))
  render(<MemoryRouter initialEntries={['/play?game=existing-game']}><PlayView /></MemoryRouter>)
  expect(snapshot).toHaveBeenCalledWith('existing-game')
  expect(fetch).not.toHaveBeenCalled()
  expect(screen.queryByRole('alert')).toBeNull()
})