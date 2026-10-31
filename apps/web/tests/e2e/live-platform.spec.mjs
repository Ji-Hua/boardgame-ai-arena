import { test as base, expect } from '@playwright/test'

const RUN = '40f0edbd60314c7ab822b6dfbd8522ef'
const SHA = '0faaae51ded5b88ed5e0b3d8774a6e5de954bf30a5ec72e4d8007afe344a15fc'
const LEGACY = 'cnn-architecture-replays-20261021'

// Observe real traffic only: no routes, response substitutions, server startup,
// training, registry mutations, deployment actions, or generated games.
const test = base.extend({
  audit: [async ({ page }, use, info) => {
    const audit = { requests: [], responses: [], consoleErrors: [], pageErrors: [], forceClicks: [] }
    page.on('request', r => audit.requests.push({ url: r.url(), method: r.method(), body: r.postData() }))
    page.on('response', r => audit.responses.push({ url: r.url(), status: r.status() }))
    page.on('pageerror', e => audit.pageErrors.push(e.message))
    page.on('console', m => { if (m.type() === 'error') audit.consoleErrors.push(m.text()) })
    await use(audit)
    await info.attach('live-network-and-browser-audit', { body: JSON.stringify(audit, null, 2), contentType: 'application/json' })
    const mutations = audit.requests.filter(r => !['GET', 'HEAD', 'OPTIONS'].includes(r.method))
    expect(mutations.filter(r => !/^\/api\/v1\/lab\/(position|analyze|branch|neural\/[^?]+)$/.test(new URL(r.url).pathname)), 'Only isolated position validation and explicitly requested analysis/branch diagnostics may POST').toEqual([])
  }, { auto: true }],
})

async function json(request, path) {
  const response = await request.get(path, { timeout: 120_000 })
  expect(response.ok(), `${path}: ${response.status()}`).toBeTruthy()
  return response.json()
}

async function evidence(info, name, value) {
  await info.attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' })
}

async function screenshot(page, info, name) {
  await info.attach(name, { body: await page.screenshot({ fullPage: true }), contentType: 'image/png' })
}

async function layoutClick(locator, audit, info) {
  // Never bypass application logic. Force is permitted only after a genuine
  // actionability failure, with elementFromPoint evidence of an obstruction.
  try { await locator.click({ timeout: 5000 }) }
  catch (error) {
    if (!/intercepts pointer events|outside of the viewport|not stable/i.test(error.message)) throw error
    const layout = await locator.evaluate(el => {
      const r = el.getBoundingClientRect()
      const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)
      return { target: el.outerHTML, rect: r.toJSON(), hit: hit?.outerHTML, obstructed: !!hit && hit !== el && !el.contains(hit) }
    })
    if (!layout.obstructed && !/not stable|outside of the viewport/i.test(error.message)) throw error
    const record = { error: error.message, layout }
    audit.forceClicks.push(record)
    await evidence(info, 'documented-layout-force-click', record)
    await locator.click({ force: true })
  }
}

async function exportCase(page) {
  await page.locator('#export-button').click()
  await expect(page.locator('#export-dialog')).toBeVisible()
  await page.locator('#data-scope').selectOption('case')
  const result = JSON.parse(await page.locator('#export-text').inputValue())
  await page.locator('#close-export').click()
  return result
}

test('legacy health GET and Training list retain both providers and exact C-resnet metrics', async ({ page, request }, info) => {
  const health = await json(request, '/health')
  expect(health).toMatchObject({ status: 'ok', service: 'training-dashboard', read_only: true })
  const runs = await json(request, '/api/v1/research/runs')
  expect(runs.runs).toEqual(expect.arrayContaining([
    expect.objectContaining({ run_id: RUN, run_name: 'C-resnet_seed-42', source: 'mlflow' }),
    expect.objectContaining({ run_id: LEGACY, source: 'legacy' }),
  ]))
  await page.goto('/training')
  await page.getByLabel('训练来源', { exact: true }).selectOption('legacy')
  await expect(page.locator(`.aw-run-choice button[title="${LEGACY}"]`)).toBeVisible()
  await screenshot(page, info, 'training-legacy')
  await page.getByLabel('训练来源', { exact: true }).selectOption('mlflow')
  await page.getByLabel('搜索训练').fill(RUN)
  const choice = page.locator('.aw-run-choice button').filter({ hasText: 'C-resnet_seed-42' })
  await expect(choice).toHaveAttribute('title', RUN)
  await choice.click()
  await expect(page.locator('.chart-card').first()).toBeVisible()
  await page.getByRole('link', { name: /完整详情/ }).click()
  await expect(page).toHaveURL(new RegExp(`/training/runs/${RUN}$`))
  await page.getByRole('button', { name: 'Metrics', exact: true }).click()
  const metrics = await json(request, `/api/v1/research/runs/${RUN}/metrics`)
  expect(metrics.run_id).toBe(RUN)
  expect(metrics.points).toHaveLength(60)
  await expect(page.locator('.chart-card').first()).toBeVisible()
  await evidence(info, 'health-and-exact-run-metrics', { health, metrics })
  await screenshot(page, info, 'c-resnet-metrics')
})

test('checkpoint → Arena Greedy → playback ply7 → isolated Lab exact import and explicit neural analysis', async ({ page, request, audit }, info) => {
  const detail = await json(request, `/api/v1/research/runs/${RUN}`)
  expect(detail.checkpoints).toEqual(expect.arrayContaining([expect.objectContaining({ sha256: SHA, filename: 'update012000.pt' })]))
  await page.goto(`/training/runs/${RUN}?tab=checkpoints`)
  const checkpoint = page.getByRole('row').filter({ hasText: SHA })
  await checkpoint.getByRole('link', { name: 'Arena evaluations', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`/arena\\?run_id=${RUN}&checkpoint_sha=${SHA}`))
  await expect(page.getByRole('heading', { name: /Aggregate evaluation results/ })).toBeVisible()
  const aggregate = page.getByRole('row').filter({ has: page.getByRole('cell', { name: /^greedy\s+Side/ }) }).filter({ hasText: 'arena_paired' })
  await expect(aggregate).toHaveCount(1)
  await expect(aggregate).toContainText('38 / 0 / 12')
  await aggregate.getByRole('link', { name: 'Evaluation replay list' }).click()
  await expect(page).toHaveURL(/opponent=greedy/)
  const games = page.getByRole('link', { name: 'View recorded game', exact: true })
  await expect(games).toHaveCount(50)
  await screenshot(page, info, 'greedy-evaluation-games')
  await games.first().click()
  await expect(page.getByLabel('Replay progress', { exact: true })).toHaveValue('0')
  await page.getByRole('button', { name: 'Auto play', exact: true }).click()
  await expect.poll(() => page.getByLabel('Replay progress', { exact: true }).inputValue()).not.toBe('0')
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  const paused = await page.getByLabel('Replay progress', { exact: true }).inputValue()
  await page.waitForTimeout(1300) // Observe more than one autoplay interval.
  await expect(page.getByLabel('Replay progress', { exact: true })).toHaveValue(paused)
  await page.getByLabel('Replay progress', { exact: true }).fill('7')
  await expect(page).toHaveURL(/ply=7/)
  const permalink = page.url()
  const replayPath = new URL(permalink).searchParams.get('replay')
  expect(replayPath).toBeTruthy()
  const replay = await json(request, `/api/v1/research/runs/${RUN}/eval-replays/${replayPath}`)
  expect(replay.states.length).toBe(replay.actions.length + 1)
  const state = replay.states[7]
  await page.reload()
  await expect(page.getByLabel('Replay progress', { exact: true })).toHaveValue('7')
  await page.getByRole('link', { name: 'Analyze exact position in Lab' }).click()
  await expect(page.getByRole('heading', { name: 'Decision Lab', exact:true })).toBeVisible()
  await expect(page.getByRole('button', {name:'Run analysis',exact:true})).toBeEnabled()
  await expect(page.locator('iframe')).toHaveCount(0)
  const context = await json(request, '/api/v1/research/context?' + new URLSearchParams({run_id:RUN,replay_path:replayPath,ply:'7',model_sha:SHA}))
  expect(context.position.pawns).toEqual([state.p1,state.p2])
  expect(context.source.ply).toBe(7)
  const imported = JSON.parse(await page.getByLabel('Position JSON').inputValue())
  expect(imported).toEqual(context.position)
  expect(audit.requests.filter(r=>new URL(r.url).pathname==='/api/v1/lab/analyze')).toHaveLength(0)
  const responsePromise=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/lab/analyze' && r.request().method()==='POST')
  await page.getByRole('button',{name:'Run analysis',exact:true}).click()
  const response=await responsePromise;const analysis=await response.json()
  expect(response.ok(),JSON.stringify(analysis)).toBe(true)
  expect(analysis.position).toEqual(context.position)
  expect(analysis.neural.model.sha256).toBe(SHA)
  await expect(page.getByRole('heading',{name:'Network diagnostics'})).toBeVisible()
  await screenshot(page,info,'native-lab-analysis')
  await page.getByRole('button',{name:'layers',exact:true}).click()
  await page.getByRole('button',{name:'Load tensor slice'}).click()
  await expect(page.locator('.lab-tensor svg')).toBeVisible()
  await screenshot(page,info,'native-lab-layer')
  await page.getByRole('button',{name:'Inspect',exact:false}).first().click()
  await expect(page.getByRole('heading',{name:/Branch:/})).toBeVisible()
  await page.getByRole('button',{name:'sensitivity',exact:true}).click()
  await expect(page.locator('.lab-tensors svg').first()).toBeVisible()
  await page.getByRole('link',{name:/Return to source/}).click()
  await expect(page.getByLabel('Replay progress',{exact:true})).toHaveValue('7')
  expect(audit.pageErrors).toEqual([])
  await evidence(info,'exact-native-lab',{context,analysis})
})

test('exact model reverse-links to Run; light/dark themes survive reload', async ({ page, request }, info) => {
  const model = await json(request, `/api/v1/models/${SHA}`)
  expect(model.sha256).toBe(SHA)
  expect(model.sha_verified).toBe(true)
  expect(model.run_refs.some(ref => ref.run_id === RUN && ref.checkpoint_sha256 === SHA)).toBeTruthy()
  await page.goto(`/models/${SHA}`)
  await expect(page.locator('.aw-hash').first()).toHaveText(SHA)
  await page.getByRole('tab', { name: /训练来源/ }).click()
  const lineage = page.getByRole('row').filter({ has: page.getByRole('link', { name: RUN, exact: true }) })
  await lineage.getByRole('link', { name: '完整详情', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`/training/runs/${RUN}$`))
  for (const theme of ['dark', 'light']) {
    if (await page.locator('html').getAttribute('data-theme') !== theme) await page.getByRole('button', { name: theme === 'dark' ? '深色' : '浅色' }).click()
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme)
    await page.reload()
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme)
    await expect(page.getByRole('button', { name: theme === 'dark' ? '浅色' : '深色' })).toBeVisible()
    await screenshot(page, info, `checkpoint-${theme}-persisted`)
  }
  await evidence(info, 'exact-model-lineage', model)
})