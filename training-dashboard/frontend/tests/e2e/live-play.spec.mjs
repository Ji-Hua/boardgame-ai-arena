import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import path from 'node:path'

const sha = '0faaae51ded5b88ed5e0b3d8774a6e5de954bf30a5ec72e4d8007afe344a15fc'
const run = '40f0edbd60314c7ab822b6dfbd8522ef'
const python = process.env.QUORIDOR_PYTHON || '/home/jihua/workspace/boardgame-ai-arena/.venv/bin/python'
const root = path.resolve('../..')
const read = async page => JSON.parse(await page.getByTestId('game-evidence').textContent())
const errors = []
test.beforeEach(async ({ page }) => {
  errors.length = 0
  page.on('pageerror', error => errors.push(error.message))
})
test.afterEach(async ({}, info) => {
  await info.attach('browser-errors', { body: JSON.stringify(errors), contentType: 'application/json' })
  expect(errors).toEqual([])
})

async function ready(page) {
  await expect.poll(async () => {
    const data = await read(page)
    if (data.error) throw new Error(data.error)
    return data.phase === 'finished' || (!data.pending && data.state.current_player === data.human_seat)
  }, { timeout: 30000 }).toBe(true)
  return read(page)
}

async function clickPawn(page, target) {
  const svg = page.locator('[data-testid="live-board"] svg.board-container')
  await svg.scrollIntoViewIfNeeded()
  const rect = await svg.boundingBox()
  await page.mouse.click(rect.x + (target[0] + .5) * rect.width / 9,
    rect.y + (8 - target[1] + .5) * rect.height / 9)
}

async function placeHighWall(page) {
  await page.getByRole('button', { name: 'Place wall', exact: true }).click()
  const svg = page.locator('[data-testid="live-board"] svg.board-container')
  await svg.scrollIntoViewIfNeeded()
  const rect = await svg.boundingBox()
  const location = { x: rect.x + 8 * rect.width / 9 - 2, y: rect.y + .4 * rect.height / 9 }
  await page.mouse.move(location.x, location.y)
  await expect(svg.locator('rect[opacity="0.7"]')).toBeVisible()
  await page.mouse.click(location.x, location.y)
  await expect.poll(async () => (await read(page)).step_count).toBeGreaterThan(0)
  const state = await ready(page)
  expect(state.state.walls).toContainEqual({ x: 7, y: 7, orientation: 'vertical' })
  expect(BigInt(state.state.wall_state.vertical_heads) & (1n << 63n)).toBe(1n << 63n)
  expect(state.state.walls_remaining[String(state.human_seat)]).toBe(9)
}

function humanMove(snapshot) {
  const code = `import json,sys
from quoridor_engine import RuleEngine,calculation,Player
from agent_system.runtime.service.agents.greedy_agent import _build_state,_action_to_engine
data=json.load(sys.stdin)
engine=RuleEngine.standard()
state=_build_state(data['state'])
player=Player.P1 if data['human_seat']==1 else Player.P2
legal=[action for action in data['legal_actions'] if action['type']=='pawn']
def score(action):
    after=engine.apply_action(state,_action_to_engine(action))
    distance=calculation.shortest_path_len(after,player,engine.topology)
    return distance if distance is not None else 999
print(json.dumps(min(legal,key=score)))`
  return JSON.parse(execFileSync(python, ['-c', code], {
    input: JSON.stringify(snapshot), encoding: 'utf8', env: { ...process.env, PYTHONPATH: root }, timeout: 10000,
  }))
}

async function finish(page, info) {
  let snapshot = await ready(page)
  const start = snapshot.step_count
  for (let turn = 0; turn < 180 && snapshot.phase !== 'finished'; turn++) {
    const before = snapshot.step_count
    const action = humanMove(snapshot)
    await clickPawn(page, action.target)
    await expect.poll(async () => (await read(page)).step_count).toBeGreaterThan(before)
    snapshot = await ready(page)
    expect(snapshot.identity.sha256).toBe(sha)
    expect(snapshot.identity.actual_sha256).toBe(sha)
  }
  expect(snapshot.phase).toBe('finished')
  expect(snapshot.result.termination).toBe('goal')
  expect(snapshot.step_count).toBeGreaterThan(start + 4)
  expect(snapshot.actions.some(action => action.player !== snapshot.human_seat)).toBe(true)
  await info.attach(`complete-game-P${snapshot.human_seat}`, { body: JSON.stringify(snapshot, null, 2), contentType: 'application/json' })
  await page.screenshot({ path: info.outputPath(`terminal-P${snapshot.human_seat}.png`), fullPage: true })
  await page.getByRole('button', { name: 'Close game', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeVisible()
}

test('Training checkpoint to exact CNN: wall, moves, refresh, navigation, offline recovery and full P1 game', async ({ page, context }, info) => {
  const traffic = []
  page.on('response', response => { if (response.url().includes('/play-api/')) traffic.push({ method: response.request().method(), url: response.url(), status: response.status() }) })
  await page.goto(`/training/runs/${run}?tab=checkpoints`)
  await expect(page.getByText('C-resnet_seed-42', { exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'Play against', exact: true }).first().click()
  expect(new URL(page.url()).searchParams.get('model_sha')).toBe(sha)
  await expect(page.getByLabel('Play checkpoint')).toHaveValue(sha)
  await page.getByRole('button', { name: 'Start game', exact: true }).click()
  await expect(page.getByTestId('actual-sha')).toHaveText(sha)
  const gameUrl = page.url()
  await ready(page)
  await placeHighWall(page)
  await page.screenshot({ path: info.outputPath('desktop-wall.png'), fullPage: true })
  let snapshot = await ready(page)
  expect(snapshot.identity.actual_load_verified).toBe(true)
  expect(snapshot.identity.algorithm).toBe('policy_distillation')
  expect(snapshot.identity.output_semantics).toBe('policy_logits')
  expect(snapshot.identity.model_config.model_arch).toBe('cnn')
  expect(snapshot.identity.observation_version).toBe('dqn_obs_cnn_v1')
  expect(snapshot.identity.action_count).toBe(209)
  if (process.env.E2E_RUNTIME_VERSION) expect(snapshot.identity.runtime_version).toBe(process.env.E2E_RUNTIME_VERSION)
  expect(snapshot.identity.agent_spec.policy_params).toEqual({ k: 1 })
  await page.reload()
  await expect(page.getByTestId('actual-sha')).toHaveText(sha)
  expect((await ready(page)).state.walls).toEqual(snapshot.state.walls)
  await page.getByRole('navigation').getByRole('link', { name: 'Models', exact: true }).click()
  await page.getByRole('navigation').getByRole('link', { name: 'Play', exact: true }).click()
  await page.getByRole('link', { name: 'Return to active game', exact: true }).click()
  expect((await ready(page)).game_id).toBe(snapshot.game_id)
  await context.setOffline(true)
  await expect(page.getByRole('alert')).toBeVisible()
  await context.setOffline(false)
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect((await ready(page)).step_count).toBe(snapshot.step_count)
  const otherUrl = new URL(gameUrl)
  otherUrl.searchParams.set('model_sha', '09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4')
  await page.goto(otherUrl.href)
  await expect(page.getByTestId('actual-sha')).toHaveText(sha)
  expect((await ready(page)).identity.sha256).toBe(sha)
  await finish(page, info)
  await info.attach('play-network', { body: JSON.stringify(traffic, null, 2), contentType: 'application/json' })
  expect(traffic.every(entry => entry.status === 200)).toBe(true)
})

test('Models deep link, P2 opening, mobile/dark layout, illegal/stale rejection and full P2 game', async ({ page }, info) => {
  await page.goto(`/models/${sha}`)
  await page.getByRole('link', { name: 'Play against', exact: true }).click()
  await expect(page.getByLabel('Play checkpoint')).toHaveValue(sha)
  await page.getByLabel('Your seat').selectOption('2')
  await page.getByRole('button', { name: 'Start game', exact: true }).click()
  await expect(page.getByTestId('actual-sha')).toHaveText(sha)
  const snapshot = await ready(page)
  expect(snapshot.step_count).toBe(1)
  expect(snapshot.actions[0].player).toBe(1)
  expect(snapshot.state.current_player).toBe(2)
  const token = await page.evaluate(game => localStorage.getItem('quoridor-play:' + game), snapshot.game_id)
  const endpoint = `/play-api/games/${snapshot.game_id}/actions`
  const headers = { Authorization: `Bearer ${token}` }
  const illegal = await page.request.post(endpoint, { headers, data: { expected_step: 1, type: 'pawn', target: [8, 0] } })
  expect(illegal.status()).toBe(422)
  const stale = await page.request.post(endpoint, { headers, data: { expected_step: 0, type: 'pawn', target: [4, 7] } })
  expect(stale.status()).toBe(409)
  expect((await ready(page)).step_count).toBe(1)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Switch to dark theme' }).click()
  await page.screenshot({ path: info.outputPath('mobile-dark.png'), fullPage: true })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  const board = await page.locator('[data-testid="live-board"] svg.board-container').boundingBox()
  expect(board.width).toBeGreaterThan(220)
  expect(board.x + board.width).toBeLessThanOrEqual(390)
  await page.setViewportSize({ width: 1440, height: 1100 })
  await finish(page, info)
})

test('unapproved and incompatible checkpoints fail closed; existing services remain reachable', async ({ page, request }, info) => {
  const unknown = await request.post('/play-api/games', { data: { sha256: 'b'.repeat(64), human_seat: 1 } })
  expect(unknown.status()).toBe(503)
  expect(await unknown.text()).toContain('not uniquely approved')
  const listing = await (await request.get('/api/platform/models')).json()
  const unsupported = listing.models.find(model => model.architecture === 'mlp')
  expect(unsupported).toBeTruthy()
  await page.goto(`/play?model_sha=${unsupported.sha256}`)
  await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeDisabled()
  const rejected = await request.post('/play-api/games', { data: { sha256: unsupported.sha256, human_seat: 1 } })
  expect(rejected.status()).toBe(503)
  const checks = []
  for (const url of ['http://192.168.68.102:8765/', 'http://192.168.68.102:8764/health', 'http://192.168.68.88:8741/', 'http://192.168.68.88:8743/health', 'http://192.168.68.88:8744/health', 'http://192.168.68.88:8751/health']) {
    const response = await request.get(url)
    checks.push({ url, status: response.status() })
    expect(response.ok()).toBe(true)
  }
  await info.attach('existing-services', { body: JSON.stringify(checks, null, 2), contentType: 'application/json' })
})