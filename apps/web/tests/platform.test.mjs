import test from 'node:test'
import assert from 'node:assert/strict'
import { arenaRoute, contextQuery, encodeContext, externalPlayUrl, plyNumber, replayRoute, runRoute, runTab, safeReturnRoute, selectedReplay } from '../src/platform.ts'

test('backend replay aliases restore opaque paths and exact Arena filters preserve full SHA', () => {
  const replay = 'eval/a #?%/game+1.json'
  assert.equal(selectedReplay(new URLSearchParams({ replay_path: replay })), replay)
  assert.equal(selectedReplay(new URLSearchParams({ replay, replay_path: 'other' })), replay)
  const sha = 'a'.repeat(64)
  const route = new URL(arenaRoute('mlflow:opaque / run', sha), 'http://localhost')
  assert.equal(route.searchParams.get('run_id'), 'mlflow:opaque / run')
  assert.equal(route.searchParams.get('checkpoint_sha'), sha)
  assert.equal(route.searchParams.has('checkpoint'), false)
  assert.equal(new URL(arenaRoute('legacy', null, 'best.pt'), 'http://localhost').searchParams.get('checkpoint'), 'best.pt')
})

test('opaque run namespace and replay paths round trip without identity truncation', () => {
  const run = 'mlflow:123 / 中文'
  const replay = 'evals/x #?%/game+1.json'
  const sha = '0faa' + 'a'.repeat(60)
  const route = replayRoute(run, { replay, ply: 17, model_sha: sha })
  const parsed = new URL(route, 'https://research.test')
  assert.equal(decodeURIComponent(parsed.pathname.split('/')[2]), run)
  assert.equal(parsed.searchParams.get('replay'), replay)
  assert.equal(parsed.searchParams.get('ply'), '17')
  assert.equal(parsed.searchParams.get('model_sha'), sha)
  assert.equal(new URL(runRoute(run, 'checkpoints'), 'https://research.test').searchParams.get('tab'), 'checkpoints')
  const query = contextQuery(run, replay, 17, sha)
  assert.equal(query.get('replay_path'), replay)
  assert.equal(query.get('run_id'), run)
  assert.equal(query.get('model_sha'), sha)
})
test('URL tabs and plies reject malformed, negative and unsafe numbers', () => {
  assert.equal(runTab('metrics'), 'metrics')
  assert.equal(runTab('no-such-tab'), 'overview')
  for (const value of [null, '-1', '1.2', 'Infinity', '9007199254740993', '2x']) assert.equal(plyNumber(value), 0)
  assert.equal(plyNumber('0'), 0)
  assert.equal(plyNumber('32'), 32)
})
test('AnalysisContext is Unicode-safe unpadded base64url JSON with no evidence loss', () => {
  const context = { position: { label: '分析 🧱' }, hash: 'f'.repeat(64), source: { run_id: 'opaque', actual_model_sha: 'a'.repeat(64) }, return_route: '/replays/opaque?ply=4' }
  const encoded = encodeContext(context)
  assert.match(encoded, /^[A-Za-z0-9_-]+$/)
  assert.deepEqual(JSON.parse(Buffer.from(encoded, 'base64url').toString('utf8')), context)
})
test('return navigation cannot escape the platform and Play accepts only explicit HTTP(S)', () => {
  for (const route of ['//evil.test', '/\\evil.test', 'https://evil.test', '/research-lab/', '/models\n/evil']) assert.equal(safeReturnRoute(route), null)
  assert.equal(safeReturnRoute('/replays/run?replay=a%2Fb&ply=2'), '/replays/run?replay=a%2Fb&ply=2')
  for (const value of ['', undefined, 'javascript:alert(1)', '/relative', 'file:///tmp/x']) assert.equal(externalPlayUrl(value), null)
  assert.equal(externalPlayUrl('https://dev.example:8765'), 'https://dev.example:8765/')
})