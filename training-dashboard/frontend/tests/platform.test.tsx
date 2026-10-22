import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import App from '../src/App'
import { api } from '../src/api'
import type { PlatformModel } from '../src/platform'

const sha = '0faa' + 'a'.repeat(60)
const runId = 'mlflow:opaque-run'
const replayPath = 'eval/a #?%/game+1.json'
const model: PlatformModel = {
  id: sha, sha256: sha, aliases: ['C-resnet'], architecture: 'cnn', output_semantics: 'policy_logits', metadata: {},
  available: true, sha_verified: false, analyzable: true, playable: false, deployed: false, actual_load_verified: false,
  requires_isolated_research_catalog: true, deployment_evidence: null, reasons: { analyzable: ['static_eligibility_only'], playable: ['no_pinned_top1_agent_spec'] }, conflicts: [], checkpoints: [],
  run_refs: [{ run_id: runId, source: 'mlflow', evidence: 'run_checkpoint_sha', checkpoint_sha256: sha }], evidence: [], registry: [], agent_specs: [],
}
const run = { run_id: runId, run_name: 'C-resnet training', source: 'mlflow', algorithm: 'policy_distillation', has_eval_replays: true, has_benchmarks: true }
const meta = { replay_path: replayPath, checkpoint_sha: sha, checkpoint: 'checkpoint.pt', episode: 0, opponent_name: 'Greedy', replay_available: true, result_from_dqn_perspective: 'win', evaluation_id: 'eval-1' }
const state = { p1: [4, 0], p2: [4, 8], h_walls: [], v_walls: [], walls_remaining_p1: 10, walls_remaining_p2: 10, current_player: 'P1' }
const detail = { game_length: 2, episode: 0, eval_opponent: { name: 'Greedy' }, actions: [0, 1].map(index => ({ turn_index: index, player_id: 'P1', action_type: 'pawn', x: 4, y: index + 1, actor_name: 'Recorded C-resnet', is_dqn_action: true })), states: [state, { ...state, p1: [4, 1] }, { ...state, p1: [4, 2] }], dqn_player_id: 'P1', opponent_player_id: 'P2', game_index: 0, illegal_acts: 0 }

function mount(route: string) {
  window.history.replaceState(null, '', route)
  return render(<BrowserRouter><App /></BrowserRouter>)
}
beforeEach(() => {
  localStorage.clear()
  vi.spyOn(api, 'models').mockResolvedValue({ models: [model, { ...model, id: 'b'.repeat(64), sha256: 'b'.repeat(64), aliases: ['Unsupported'], analyzable: false, reasons: { analyzable: ['unsupported_or_unknown_observation'] } }], count: 2, warnings: [], sources: [] })
  vi.spyOn(api, 'model').mockResolvedValue(model)
  vi.spyOn(api, 'listRuns').mockResolvedValue({ runs: [run], count: 1, warnings: [], artifact_roots_scanned: [] } as any)
  vi.spyOn(api, 'evaluations').mockResolvedValue({ run_id: runId, benchmarks: [], replays: [meta], warnings: [] } as any)
  vi.spyOn(api, 'getEvalReplay').mockResolvedValue(detail as any)
  vi.spyOn(api, 'getRun').mockResolvedValue({ ...run, checkpoints: [{ filename: 'checkpoint.pt', sha256: sha }], summary_stats: {}, config: { architecture: 'resnet' }, provenance: {}, warnings: [], metric_files: [], benchmark_files: [] } as any)
  vi.spyOn(api, 'getMetrics').mockResolvedValue({ columns: [], points: [], warnings: [], snapshots: [], metadata: {}, axes: [] } as any)
  vi.spyOn(api, 'getBenchmarks').mockResolvedValue({ benchmarks: [], warnings: [], count: 0, run_id: runId })
  vi.spyOn(api, 'summary').mockResolvedValue({ total_runs: 1, available_algorithms: [], available_reward_modes: [], available_model_arches: [] } as any)
})
afterEach(() => { cleanup(); vi.useRealTimers() })

describe('research shell and URL contracts', () => {
  it('offers all six navigation sections and persists accessible theme toggles', async () => {
    mount('/play')
    const nav = screen.getByRole('navigation', { name: 'Main navigation' })
    for (const name of ['Play', 'Training', 'Arena', 'Replay', 'Lab', 'Models']) expect(within(nav).getByRole('link', { name: new RegExp(name) })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }))
    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('dark'))
    expect(localStorage.getItem('quoridor-theme')).toBe('dark')
    expect(screen.queryByRole('iframe')).toBeNull()
    expect(screen.getByRole('heading', { name: 'Human vs Checkpoint' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Start game' }).hasAttribute('disabled')).toBe(true)
  })
  it('restores Run tabs after reload and browser back; preserves checkpoint full SHA links', async () => {
    const route = `/training/runs/${encodeURIComponent(runId)}?tab=checkpoints`
    const view = mount(route)
    const link = await screen.findByRole('link', { name: sha })
    expect(link.getAttribute('href')).toBe(`/models/${sha}`)
    const playLink = screen.getByRole('link', { name: 'Play against' })
    expect(new URL(playLink.getAttribute('href')!, 'http://localhost').searchParams.get('model_sha')).toBe(sha)
    const arenaLink = within(screen.getByRole('table')).getByRole('link', { name: 'Arena evaluations' })
    const arenaUrl = new URL(arenaLink.getAttribute('href')!, 'http://localhost')
    expect(arenaUrl.pathname).toBe('/arena')
    expect(arenaUrl.searchParams.get('run_id')).toBe(runId)
    expect(arenaUrl.searchParams.get('checkpoint_sha')).toBe(sha)
    fireEvent.click(screen.getByRole('button', { name: 'Config' }))
    expect(new URLSearchParams(window.location.search).get('tab')).toBe('config')
    await act(async () => { window.history.back() })
    await waitFor(() => expect(new URLSearchParams(window.location.search).get('tab')).toBe('checkpoints'))
    expect(await screen.findByRole('link', { name: sha })).toBeTruthy()
    view.unmount(); mount(route)
    expect(await screen.findByRole('link', { name: sha })).toBeTruthy()
  })
  it('keeps Compare selection in repeated run params, including an explicit empty selection', async () => {
    mount(`/training/compare?${new URLSearchParams({ run: runId })}`)
    const checkbox = await screen.findByRole('checkbox') as HTMLInputElement
    expect(checkbox.checked).toBe(true)
    fireEvent.click(checkbox)
    expect(new URLSearchParams(window.location.search).getAll('run')).toEqual([])
    fireEvent.click(checkbox)
    expect(new URLSearchParams(window.location.search).getAll('run')).toEqual([runId])
  })
  it('loads Models from its real contract and links full C-resnet SHA lineage to replays', async () => {
    mount(`/models/${sha}`)
    expect(await screen.findByRole('heading', { name: 'C-resnet' })).toBeTruthy()
    expect(api.model).toHaveBeenCalledWith(sha)
    const replay = screen.getByRole('link', { name: /Recorded evaluations & replays/ })
    expect(new URL(replay.getAttribute('href')!, 'http://localhost').searchParams.get('checkpoint_sha')).toBe(sha)
    expect(screen.getByText('playable: No')).toBeTruthy()
    expect(screen.getByText('no_pinned_top1_agent_spec')).toBeTruthy()
    const arena = new URL(screen.getByRole('link', { name: 'Arena evaluations →' }).getAttribute('href')!, 'http://localhost')
    expect(arena.searchParams.get('run_id')).toBe(runId)
    expect(arena.searchParams.get('checkpoint_sha')).toBe(sha)
  })
  it('rejects abbreviated model identities instead of resolving by display alias', async () => {
    mount('/models/0faa')
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(api.model).not.toHaveBeenCalled()
  })
  it('loads global run groups lazily and opens the reused checkpoint/evaluation index', async () => {
    mount('/replays')
    const summary = await screen.findByText(/C-resnet training · mlflow · Recorded games indexed/)
    expect(api.evaluations).not.toHaveBeenCalled()
    const group = summary.closest('details')!
    fireEvent(group, new Event('toggle'))
    group.open = true; fireEvent(group, new Event('toggle'))
    await waitFor(() => expect(api.evaluations).toHaveBeenCalledWith(runId))
    expect(await screen.findByText(/eval-1 · Protocol unspecified/)).toBeTruthy()
  })
  it('restores opaque replay and exact ply; playback is paused until requested, speed and end are bounded', async () => {
    const query = new URLSearchParams({ replay: replayPath, ply: '1' })
    mount(`/replays/${encodeURIComponent(runId)}?${query}`)
    expect(await screen.findByText('Step 1 / 2')).toBeTruthy()
    expect(api.getEvalReplay).toHaveBeenCalledWith(runId, replayPath)
    expect(screen.getByRole('button', { name: 'Auto play' })).toBeTruthy()
    const lab = screen.getByRole('link', { name: /Analyze exact position/ })
    const labUrl = new URL(lab.getAttribute('href')!, 'http://localhost')
    expect(labUrl.pathname).toBe('/lab')
    expect(labUrl.searchParams.get('replay_path')).toBe(replayPath)
    expect(labUrl.searchParams.get('ply')).toBe('1')
    expect(labUrl.searchParams.get('model_sha')).toBe(sha)
    expect(screen.getByRole('option', { name: `Use recorded model · ${sha}` })).toBeTruthy()
    expect(screen.getByText('Recorded C-resnet · P1')).toBeTruthy()
    const unsupported = await screen.findByRole('option', { name: /Unsupported/ }) as HTMLOptionElement
    expect(unsupported.disabled).toBe(true)
    vi.useFakeTimers()
    fireEvent.change(screen.getByRole('slider', { name: 'Playback speed' }), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Auto play' }))
    await act(async () => { vi.advanceTimersByTime(250) })
    expect(screen.getByText('Step 2 / 2')).toBeTruthy()
    expect(new URLSearchParams(window.location.search).get('ply')).toBe('2')
    await act(async () => { vi.advanceTimersByTime(250) })
    expect(screen.getByRole('button', { name: 'Auto play' }).hasAttribute('disabled')).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'First position' }))
    expect(screen.getByText('Step 0 / 2')).toBeTruthy()
  })
  it('clamps out of range plies and refuses incomplete recorded trajectories', async () => {
    const view = mount(`/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay: replayPath, ply: '999' })}`)
    expect(await screen.findByText('Step 2 / 2')).toBeTruthy()
    await waitFor(() => expect(new URLSearchParams(window.location.search).get('ply')).toBe('2'))
    view.unmount()
    vi.mocked(api.getEvalReplay).mockResolvedValue({ ...detail, states: [state] } as any)
    mount(`/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay: replayPath })}`)
    expect(await screen.findByText(/Full recorded trajectory is unavailable/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Auto play' })).toBeNull()
  })
  it('only resolves context on GET and passes exact safe context to the isolated Lab, never runs analysis', async () => {
    const context = { position: { pawns: [[4, 1], [4, 8]], turn: 2, label: '分析' }, position_hash: 'f'.repeat(64), source: { run: { run_id: runId, source: 'mlflow' }, game: { id: 'actual-game-hash' }, evaluation: { id: 'eval-hash', evaluation_id: 'arena-v2' }, replay_path: replayPath, ply: 1 }, recorded_model: { sha256: sha }, analysis_model: model, return_route: `/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay_path: replayPath, ply: '1' })}` }
    vi.spyOn(api, 'context').mockResolvedValue(context)
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    mount(`/lab?${new URLSearchParams({ run_id: runId, replay_path: replayPath, ply: '1', model_sha: sha })}`)
    const launch = await screen.findByRole('link', { name: /Open isolated Lab/ })
    const href = new URL(launch.getAttribute('href')!, 'http://localhost')
    expect(href.pathname).toBe('/research-lab/')
    const encoded = href.searchParams.get('context')!
    expect(encoded).toMatch(/^[A-Za-z0-9_-]+$/)
    expect(JSON.parse(Buffer.from(encoded, 'base64url').toString('utf8'))).toEqual(context)
    expect(vi.mocked(api.context).mock.calls[0][0].get('replay_path')).toBe(replayPath)
    expect(fetchSpy).not.toHaveBeenCalled()
    const strip = screen.getByRole('complementary', { name: 'Research context' })
    expect(within(strip).getByText('actual-game-hash')).toBeTruthy()
    expect(within(strip).getByText('arena-v2')).toBeTruthy()
    expect(within(strip).getByText('Analysis model · recorded model')).toBeTruthy()
    const provenance = screen.getByText('Full context provenance (JSON)').closest('details')!
    expect(provenance.open).toBe(false)
    fireEvent.click(screen.getByRole('link', { name: 'Return to source' }))
    expect(await screen.findByText('Step 1 / 2')).toBeTruthy()
    expect(api.getEvalReplay).toHaveBeenCalledWith(runId, replayPath)
  })
  it('shows context failures without offering an unvalidated Lab launch', async () => {
    vi.spyOn(api, 'context').mockResolvedValue({ position: {}, return_route: '//evil.example' })
    mount('/lab')
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.queryByRole('link', { name: /Open isolated Lab/ })).toBeNull()
  })
  it('falls back only on an unavailable platform endpoint and retains explicit rollout warnings', async () => {
    vi.mocked(api.evaluations).mockRejectedValue(new Error('404 Not Found'))
    vi.spyOn(api, 'listEvalReplays').mockResolvedValue({ run_id: runId, replays: [meta], count: 1 } as any)
    mount(`/replays/${encodeURIComponent(runId)}`)
    expect(await screen.findByText(/Platform index unavailable; existing APIs used/)).toBeTruthy()
    expect(api.listEvalReplays).toHaveBeenCalledWith(runId)
    expect(screen.getByRole('link', { name: 'View recorded game' })).toBeTruthy()
  })
  it('keeps a direct replay playable when the global index fails, without falling back on permission errors', async () => {
    vi.mocked(api.evaluations).mockRejectedValue(new Error('403 Forbidden'))
    const fallback = vi.spyOn(api, 'listEvalReplays')
    mount(`/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay: replayPath, ply: '1' })}`)
    expect(await screen.findByText('Step 1 / 2')).toBeTruthy()
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(fallback).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Auto play' })).toBeTruthy()
  })
  it('restores backend replay_path aliases, closes both aliases and honors explicit model overrides', async () => {
    const override = 'b'.repeat(64)
    mount(`/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay_path: replayPath, ply: '1', model_sha: override })}`)
    expect(await screen.findByText('Step 1 / 2')).toBeTruthy()
    const url = new URL(screen.getByRole('link', { name: /Analyze exact position/ }).getAttribute('href')!, 'http://localhost')
    expect(url.searchParams.get('model_sha')).toBe(override)
    expect(screen.getByText(`Checkpoint: ${sha}`)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Close/ }))
    expect(new URLSearchParams(window.location.search).has('replay_path')).toBe(false)
    expect(screen.queryByText('Step 1 / 2')).toBeNull()
  })
  it('uses recorded detail SHA even when the replay index is unavailable', async () => {
    vi.mocked(api.evaluations).mockRejectedValue(new Error('403 Forbidden'))
    vi.mocked(api.getEvalReplay).mockResolvedValue({ ...detail, source: { checkpoint_sha256: sha } } as any)
    mount(`/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay_path: replayPath, ply: '1' })}`)
    expect(await screen.findByText('Step 1 / 2')).toBeTruthy()
    const url = new URL(screen.getByRole('link', { name: /Analyze exact position/ }).getAttribute('href')!, 'http://localhost')
    expect(url.searchParams.get('model_sha')).toBe(sha)
  })
  it('renders real per-run Arena aggregates filtered by full SHA with exact evaluation replay links', async () => {
    const benchmark = { run_id: runId, checkpoint: 'checkpoint.pt', checkpoint_sha: sha, evaluation_id: 'eval-1', opponent: 'Greedy', protocol: 'paired-openings', games: 100, wins: 60, draws: 10, losses: 30, replay_available: true }
    vi.mocked(api.evaluations).mockResolvedValue({ run_id: runId, benchmarks: [benchmark, { ...benchmark, checkpoint_sha: 'b'.repeat(64), evaluation_id: 'wrong-checkpoint' }, { ...benchmark, checkpoint_sha: null, evaluation_id: 'unknown-checkpoint' }], replays: [meta], warnings: [] } as any)
    mount(`/arena?${new URLSearchParams({ run_id: runId, checkpoint_sha: sha })}`)
    expect(await screen.findByRole('heading', { name: 'Aggregate evaluation results · 1' })).toBeTruthy()
    expect(screen.queryByText('wrong-checkpoint')).toBeNull()
    expect(screen.queryByText('unknown-checkpoint')).toBeNull()
    expect(api.evaluations).toHaveBeenCalledWith(runId)
    expect(screen.getByRole('heading', { name: 'Run Arena evaluations' })).toBeTruthy()
    const links = screen.getAllByRole('link', { name: 'Evaluation replay list →' })
    const url = new URL(links[0].getAttribute('href')!, 'http://localhost')
    expect(url.pathname).toBe(`/replays/${encodeURIComponent(runId)}`)
    expect(url.searchParams.get('checkpoint_sha')).toBe(sha)
    expect(url.searchParams.get('evaluation')).toBe('eval-1')
    expect(screen.getByRole('link', { name: 'Latest Arena matrix' }).getAttribute('href')).toBe('/arena')
  })
  it('preserves the latest Arena endpoint and matrix when no run is selected', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ run_id: 'latest-arena', status: 'complete', agents: [], ranking: [], matrix: {}, forfeits: [], standard_controls: [], terminations: {}, ranked_games: 0, control_games: 0, pairs_per_match: 1, completed_blocks: 1, expected_blocks: 1, models: [], registry: [] }) } as any)
    mount('/arena')
    expect(await screen.findByText(/latest-arena/)).toBeTruthy()
    expect(fetch).toHaveBeenCalledWith('/api/arena/latest')
    expect(api.evaluations).not.toHaveBeenCalled()
  })
  it('opens only the exact protocol, side, seed and checkpoint progress replay group', async () => {
    const benchmark = { run_id: runId, checkpoint: 'checkpoint.pt', checkpoint_sha: sha, checkpoint_episode: 0, checkpoint_step: 10, evaluation_id: 'eval-1', opponent: 'Greedy', protocol: 'paired-openings', side: 'both', seed: 0, games: 100, wins: 60, draws: 10, losses: 30, replay_available: true }
    const other = { ...benchmark, protocol: 'standard-opening', seed: 42, wins: 1, draws: 2, losses: 97 }
    const selectedMeta = { ...meta, protocol: benchmark.protocol, side: benchmark.side, seed: benchmark.seed }
    const otherMeta = { ...meta, replay_path: 'other.json', protocol: other.protocol, side: other.side, seed: other.seed }
    vi.mocked(api.evaluations).mockResolvedValue({ run_id: runId, benchmarks: [benchmark, other], replays: [selectedMeta, otherMeta], warnings: ['Recorded source warning'] } as any)
    mount(`/arena?${new URLSearchParams({ run_id: runId, checkpoint_sha: sha })}`)
    expect(await screen.findByRole('heading', { name: 'Aggregate evaluation results · 2' })).toBeTruthy()
    expect(screen.getByText('Recorded source warning')).toBeTruthy()
    const section = screen.getByRole('heading', { name: 'Aggregate evaluation results · 2' }).closest('section')!
    const rows = within(section).getAllByRole('row')
    expect(within(rows[1]).getByText('60 / 10 / 30')).toBeTruthy()
    expect(within(rows[1]).getByText('Side both · seed 0')).toBeTruthy()
    expect(within(rows[1]).getByText('Greedy')).toBeTruthy()
    fireEvent.click(within(rows[1]).getByRole('link', { name: 'Evaluation replay list →' }))
    expect(await screen.findByText('Showing 1 of 2')).toBeTruthy()
    expect(screen.queryByText(/standard-opening/)).toBeNull()
    const gameLink = screen.getByRole('link', { name: 'View recorded game' })
    expect(new URL(gameLink.getAttribute('href')!, 'http://localhost').searchParams.get('replay')).toBe(replayPath)
    let parent = gameLink.closest('details')
    while (parent) { expect(parent.open).toBe(true); parent = parent.parentElement?.closest('details') ?? null }
    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
    expect(await screen.findByText('Showing 2 of 2')).toBeTruthy()
  })
  it('shows hashed nested evaluation identity without sending it as a provider evaluation filter', async () => {
    vi.spyOn(api, 'context').mockResolvedValue({ position: {}, source: { run: { run_id: runId, source: 'mlflow' }, evaluation: { id: 'canonical-evaluation-hash', evaluation_id: null }, game: { id: 'canonical-game-hash' }, replay_path: replayPath, ply: 0 }, return_route: `/replays/${encodeURIComponent(runId)}?${new URLSearchParams({ replay_path: replayPath })}` })
    mount(`/lab?${new URLSearchParams({ evaluation: 'stale-flat-id' })}`)
    const strip = await screen.findByRole('complementary', { name: 'Research context' })
    expect(within(strip).getByText('canonical-evaluation-hash')).toBeTruthy()
    expect(within(strip).getByText('canonical-game-hash')).toBeTruthy()
    const archive = new URL(within(strip).getByRole('link', { name: 'Replay archive' }).getAttribute('href')!, 'http://localhost')
    expect(archive.searchParams.has('evaluation')).toBe(false)
    expect(archive.searchParams.get('replay')).toBe(replayPath)
    expect(screen.queryByText('stale-flat-id')).toBeNull()
  })
})