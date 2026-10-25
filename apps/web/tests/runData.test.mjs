import test from 'node:test'
import assert from 'node:assert/strict'
import { availableAxes, chartData, chartDomains, comparisonGroups, evaluationGroups, evaluationKey, replayTurn, semanticKey } from '../src/runData.ts'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

const point = (episode, value, step = null) => ({ episode, step, values: { metric: value } })
const metadata = { label: 'TD loss', semantic: 'td_loss', unit: 'loss', scope: 'training', step_type: 'env_steps' }
const series = (meta = metadata, axis = 'episode') => ({ points: [point(1, 2, 10), point(2, 5, 20)], columns: ['metric'], metadata: { metric: meta }, axes: [{ key: axis, label: axis }], snapshots: [], warnings: [], provenance: {} })

test('filter null/non-finite values, preserve real zero and sort actual axes', () => {
  assert.deepEqual(chartData([point(2, null), point(3, Infinity), point(1, 0), point(null, 4), point(0, 2)], 'episode', 'metric'), [{ x: 0, y: 2 }, { x: 1, y: 0 }])
})
test('all overlaid series use shared X/Y domains, never independent scaling', () => {
  assert.deepEqual(chartDomains([{ points: [point(1, 10), point(2, 20)], yKey: 'metric' }, { points: [point(100, -50), point(200, 1000)], yKey: 'metric' }], 'episode'), { xMin: 1, xMax: 200, yMin: -50, yMax: 1000 })
})
test('exact semantic/unit/scope/actual-axis required for cross-source overlay', () => {
  const entries = [{ runId: 'legacy-id', name: 'Legacy', series: series() }, { runId: 'opaque-mlflow-id', name: 'MLflow', series: series() }]
  assert.equal(comparisonGroups(entries).length, 1)
  for (const change of [{ semantic: 'policy_cross_entropy' }, { unit: 'logit' }, { scope: 'heldout' }, { unit: null }, { scope: 'unknown' }]) {
    assert.equal(comparisonGroups([entries[0], { ...entries[1], series: series({ ...metadata, ...change }) }]).length, 0)
  }
  assert.equal(comparisonGroups([entries[0], { ...entries[1], series: series(metadata, 'step') }]).length, 0)
  assert.equal(semanticKey(metadata, 'unknown_step'), null)
})
test('Q-values, logits, rewards, values and score/win rates stay distinct', () => {
  const semantics = ['action_value', 'policy_logits', 'episode_reward', 'state_value', 'score_rate', 'win_rate']
  assert.equal(new Set(semantics.map(semantic => semanticKey({ ...metadata, semantic }, 'episode'))).size, semantics.length)
})
test('only present axes are exposed; generic step is not environment steps', () => {
  const s = series({ ...metadata, step_type: 'unknown' }, 'step')
  s.axes.push({ key: 'gradient_updates', label: 'Gradient updates' })
  assert.deepEqual(availableAxes(s), [{ key: 'step', label: 'Recorded step (type unspecified)' }])
})
test('gradient update aliases can overlay but unknown MLflow step cannot', () => {
  const a = series({ ...metadata, step_type: 'gradient_updates' }, 'step')
  const b = series(metadata, 'update')
  b.points = b.points.map((p, i) => ({ ...p, update: (i + 1) * 10 }))
  assert.equal(comparisonGroups([{ runId: 'a', name: 'a', series: a }, { runId: 'b', name: 'b', series: b }]).length, 1)
  assert.equal(comparisonGroups([{ runId: 'a', name: 'a', series: series({ ...metadata, step_type: 'unknown' }, 'step') }, { runId: 'b', name: 'b', series: b }]).length, 0)
})

const replay = (extra = {}) => ({ episode: 100, checkpoint: 'ep100.pt', checkpoint_sha: 'sha', step: null, progress_type: 'episode', evaluation_id: 'eval-a', protocol: 'paired', opponent_name: 'minimax', opponent_depth: 1, game_type: 'evaluation', replay_path: 'game/1', result_from_dqn_perspective: 'win', ...extra })
const benchmark = (extra = {}) => ({ checkpoint_episode: 100, checkpoint_step: null, checkpoint: 'ep100.pt', checkpoint_sha: 'sha', progress_type: 'episode', evaluation_id: 'eval-a', protocol: 'paired', opponent: 'minimax', opponent_depth: 1, game_type: 'evaluation', games: 100, wins: 40, draws: 10, losses: 50, win_rate: 0.4, score_rate: 0.45, ...extra })
test('evaluation hierarchy matches summaries without treating replay samples as aggregate WDL', () => {
  const groups = evaluationGroups([replay()], [benchmark()])
  assert.equal(groups.length, 1)
  assert.equal(groups[0].groups[0].games.length, 1)
  assert.equal(groups[0].groups[0].summaries[0].games, 100)
  assert.equal(groups[0].groups[0].summaries[0].score_rate, 0.45)
})
test('checkpoint SHA/progress, evaluation ID, protocol and opponent separate groups', () => {
  const groups = evaluationGroups([replay(), replay({ protocol: 'control' }), replay({ evaluation_id: 'eval-b' }), replay({ opponent_name: 'random' }), replay({ checkpoint_sha: 'other-sha' })], [])
  assert.equal(groups.length, 2)
  assert.equal(groups[0].groups.length, 4)
  assert.ok(groups[0].groups.every(g => !g.summaries.length))
})
test('Decision Lab uses stored current_player, including a P2 opening, never step parity', () => {
  assert.equal(replayTurn({ current_player: 'P2' }, 'P1'), 2)
  assert.equal(replayTurn({ current_player: 1 }, 'P2'), 1)
  assert.equal(replayTurn({}, 'P2'), 2)
  assert.equal(replayTurn({}), null)
})

test('legacy replay with missing checkpoint step uses unique real aggregate, never invents totals', () => {
  const r = replay({ checkpoint_sha: null, step: null, progress_type: null, protocol: null, evaluation_id: null })
  const b = benchmark({ checkpoint_sha: null, checkpoint_step: 5000, progress_type: 'env_steps', protocol: null, evaluation_id: null })
  const groups = evaluationGroups([r], [b])
  assert.equal(groups.length, 1)
  assert.equal(groups[0].groups[0].games.length, 1)
  assert.equal(groups[0].groups[0].summaries[0].games, 100)
  const ambiguous = evaluationGroups([r], [b, { ...b, protocol: 'different' }])
  assert.ok(ambiguous.some(ck => ck.groups.some(g => g.games.length && !g.summaries.length)))
  const differentCheckpoint = evaluationGroups([r], [{ ...b, checkpoint: 'other.pt' }])
  assert.equal(differentCheckpoint.length, 2)
})
test('older legacy API still offers only actually recorded chart axes', () => {
  const s = series()
  delete s.axes
  assert.deepEqual(availableAxes(s).map(a => a.key), ['episode', 'step'])
})

const replayGroups = groups => groups.flatMap(checkpoint => checkpoint.groups).filter(group => group.games.length)
test('episode-only evidence never attaches, even when partial grouping keys are identical', () => {
  const r = replay({ checkpoint: null, checkpoint_sha: null })
  for (const b of [benchmark(), benchmark({ checkpoint: null, checkpoint_sha: null })]) {
    const groups = replayGroups(evaluationGroups([r], [b]))
    assert.equal(groups.length, 1)
    assert.equal(groups[0].summaries.length, 0)
  }
  assert.equal(replayGroups(evaluationGroups([replay({ checkpoint: null })], [benchmark({ checkpoint: null })]))[0].summaries.length, 1, 'exact SHA alone remains valid')
})
test('side, seed, manifest SHA and full protocol distinguish aggregate identities', () => {
  for (const change of [
    { side: 'P1' }, { seed: 0 }, { source_manifest: { sha256: 'manifest-a' } },
    { source_manifest: { protocol: { pairs: 50, opening: 'heldout' } } },
    { opponent_family: 'random' }, { opponent_depth: 2 }, { game_type: 'control' },
    { evaluation_id: 'eval-b' }, { protocol: 'control' },
  ]) {
    assert.notEqual(evaluationKey(benchmark()), evaluationKey(benchmark(change)))
    const summaries = [benchmark(), benchmark(change)]
    const groups = evaluationGroups([replay()], summaries).flatMap(checkpoint => checkpoint.groups)
    assert.equal(groups.filter(group => group.summaries.length).length, 2)
  }
  assert.equal(evaluationKey(benchmark({ source_manifest: { protocol: { a: 1, b: 2 } } })),
    evaluationKey(benchmark({ source_manifest: { protocol: { b: 2, a: 1 } } })))
})
test('unknown replay side or seed cannot choose between candidate summaries', () => {
  for (const variants of [
    [{ side: 'P1' }, { side: 'P2' }],
    [{ seed: 0 }, { seed: 42 }],
    [{ source_manifest: { sha256: 'a' } }, { source_manifest: { sha256: 'b' } }],
    [{ source_manifest: { protocol: { pairs: 10 } } }, { source_manifest: { protocol: { pairs: 20 } } }],
  ]) {
    const summaries = variants.map(extra => benchmark(extra))
    const unknown = replayGroups(evaluationGroups([replay()], summaries))
    assert.equal(unknown[0].summaries.length, 0)
    const known = replayGroups(evaluationGroups([replay(variants[0])], summaries))
    assert.equal(known[0].summaries.length, 1)
    assert.equal(known[0].summaries[0], summaries[0])
  }
  const summaries = [benchmark({ side: null }), benchmark({ side: 'P1' })]
  assert.equal(replayGroups(evaluationGroups([replay()], summaries))[0].summaries.length, 0, 'missing-value summary cannot bypass ambiguity via identical keys')
  assert.equal(replayGroups(evaluationGroups([replay()], [benchmark(), benchmark()]))[0].summaries.length, 0, 'duplicate aggregates are not assumed unique')
})
test('recorded checkpoint/protocol conflicts never attach to an aggregate', () => {
  for (const change of [{ checkpoint_sha: 'other' }, { checkpoint: 'other.pt' }, { step: 99 },
    { episode: 101 }, { side: 'P2' }, { seed: 42 }, { source_manifest: { sha256: 'other' } }]) {
    const summaries = [benchmark({ checkpoint_step: 100, side: 'P1', seed: 0, source_manifest: { sha256: 'manifest' } })]
    assert.equal(replayGroups(evaluationGroups([replay(change)], summaries))[0].summaries.length, 0, JSON.stringify(change))
  }
})

// Execute the real component with a minimal hook host, without adding a DOM or
// test framework dependency. Requests stay controlled to exercise stale replies.
test('compare refresh reloads every endpoint, clears recovered errors and never auto-retries', async () => {
  const states = [], effects = [], pendingEffects = [], requests = []
  let cursor = 0, tree
  let searchParams = new URLSearchParams('run=a&run=b')
  const react = {
    createElement: (type, props, ...children) => ({ type, props: props ?? {}, children: children.flat(Infinity) }),
    useState(initial) {
      const index = cursor++
      if (!(index in states)) states[index] = initial
      return [states[index], value => { states[index] = typeof value === 'function' ? value(states[index]) : value }]
    },
    useMemo(factory, deps) {
      const index = cursor++
      const previous = states[index]
      if (!previous || deps.some((value, i) => value !== previous.deps[i])) states[index] = { deps, value: factory() }
      return states[index].value
    },
    useEffect(effect, deps) {
      const index = cursor++, previous = effects[index]
      if (!previous || deps.some((value, i) => value !== previous.deps[i])) {
        previous?.cleanup?.()
        effects[index] = { deps }
        pendingEffects.push(() => { effects[index].cleanup = effect() })
      }
    },
  }
  const runs = ['a', 'b'].map(run_id => ({ run_id, run_name: run_id }))
  const api = { listRuns: async () => ({ runs, warnings: [] }) }
  for (const endpoint of ['getRun', 'getMetrics', 'getBenchmarks']) {
    api[endpoint] = id => new Promise((resolve, reject) => requests.push({ endpoint, id, resolve, reject }))
  }
  const exports = {}
  const require = createRequire(import.meta.url)
  const source = ts.transpileModule(readFileSync(new URL('../src/views/CompareView.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, esModuleInterop: true },
  }).outputText
  runInNewContext(source, { exports, URLSearchParams, require: name => name === 'react' ? react : name === 'react-router-dom' ? { useSearchParams: () => [searchParams, next => { searchParams = next }] } : name === '../api' ? { api } : name === '../runData' ? { comparisonGroups, display: String } : name.startsWith('.') ? {} : require(name) })
  const render = () => {
    cursor = 0
    tree = exports.CompareView({ onBack() {} })
    pendingEffects.splice(0).forEach(effect => effect())
  }
  const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); render() }
  const nodes = node => !node || typeof node !== 'object' ? [] : [node, ...node.children.flatMap(nodes)]
  const text = node => typeof node === 'string' ? node : node?.children?.map(text).join(' ') ?? ''
  const finish = (request, warning = []) => request.resolve(request.endpoint === 'getRun' ? { run_id: request.id, config: {} } : request.endpoint === 'getMetrics' ? { ...series(), warnings: warning } : { benchmarks: [benchmark()], warnings: warning })
  render(); await settle(); render()
  assert.equal(requests.length, 6)
  const initial = requests.splice(0)
  initial.forEach(request => request.endpoint === 'getMetrics' ? request.reject(new Error('offline')) : finish(request, ['old warning']))
  await settle()
  assert.match(text(tree), /offline/)
  render(); await settle()
  assert.equal(requests.length, 0, 'failed endpoints must not trigger a retry loop')
  const refresh = () => nodes(tree).find(node => node.type === 'button' && text(node) === 'Refresh selected runs').props.onClick()
  refresh(); render(); render()
  assert.equal(requests.length, 6)
  assert.ok(!nodes(tree).some(node => node.props.rows?.length), 'old benchmark evidence is invalidated')
  assert.doesNotMatch(text(tree), /old warning/)
  requests.splice(0).forEach(request => finish(request))
  await settle()
  assert.doesNotMatch(text(tree), /offline/)
  assert.ok(nodes(tree).some(node => node.type === 'button' && text(node) === 'Refresh selected runs' && !node.props.disabled))
  assert.equal(requests.length, 0)

  refresh(); render()
  const stale = requests.splice(0)
  const chip = nodes(tree).find(node => node.type === 'input' && node.props.type === 'checkbox')
  chip.props.onChange(); render()
  assert.equal(requests.length, 3)
  requests.splice(0).forEach(request => finish(request))
  await settle()
  stale.forEach(request => request.reject(new Error('obsolete generation')))
  await settle()
  assert.doesNotMatch(text(tree), /obsolete generation/)
  assert.equal(requests.length, 0)
})