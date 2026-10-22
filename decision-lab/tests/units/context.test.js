// TEST_CLASSIFICATION: UNIT
// Real Lab module, mocked browser/network boundaries; no inference or engine rewrite.
import {test} from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const appURL = new URL("../../web/app.js", import.meta.url);
const sha = "a".repeat(64), otherSHA = "a".repeat(63) + "b";
const position = {pawns: [[3, 5], [5, 2]], walls: [
  {orientation: "horizontal", x: 2, y: 3}, {orientation: "vertical", x: 6, y: 4},
], walls_remaining: [8, 10], turn: 2};
const context = () => ({
  schema_version: "analysis_context_v1", position: structuredClone(position), position_hash: "c".repeat(64),
  source: {type: "recorded_replay", ply: 17, run: {source: "mlflow", run_id: "来源/实验"},
    evaluation: {id: "eval-hash", manifest_sha256: "d".repeat(64)},
    game: {id: "game-hash", opening_hash: "opening"}, replay_path: "replays/棋局.json",
    evidence: {checkpoint_sha256: otherSHA}},
  actual_action: {type: "pawn", player: 2, target: [5, 1]},
  recorded_model: {sha256: otherSHA}, analysis_model: {id: sha, sha256: sha, aliases: ["not-a-lab-id"]},
  return_route: "/replays/mlflow%3Aexample?replay_path=games%2Fone.json&ply=17", warnings: ["来源警告 <script>"],
});
const encode = value => Buffer.from(JSON.stringify(value)).toString("base64url");
const entry = (id = "isolated-checkpoint", digest = sha) => ({id, sha256: digest, available: true, architecture: "cnn", metadata: {}});
let sequence = 0;

async function load(t, {encoded = encode(context()), models = [entry()], pathname = "/research-lab/", catalogError = false, rejectPosition = false} = {}) {
  const elements = new Map(), calls = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: id === "candidate-filter" ? "all" : id === "data-scope" ? "case" : "",
      innerHTML: "", textContent: "", selectedOptions: [], checked: false,
      classList: {add() {}, toggle() {}}, setAttribute() {}, showModal() {}, close() {},
    });
    return elements.get(id);
  };
  const prefix = pathname.startsWith("/research-lab") ? "/research-lab" : "";
  const globals = {
    document: {getElementById: element, querySelector: element, querySelectorAll: () => [], body: element("body")},
    window: {location: {pathname, search: `?context=${encoded}&position=${encode({turn: 1})}`}},
    fetch: async (path, options = {}) => {
      const body = options.body ? JSON.parse(options.body) : undefined;
      calls.push({path, body});
      assert.ok(path.startsWith(prefix + "/api/"), `API escaped isolated prefix: ${path}`);
      const route = path.slice(prefix.length);
      let data;
      if (route === "/api/meta") data = {provenance: {source_sha256: "0123456789abcdef"}, presets: [
        {id: "opening", title: "opening", position: {pawns: [[4, 0], [4, 8]], walls: [], walls_remaining: [10, 10], turn: 1}},
      ]};
      else if (route === "/api/models") {
        if (catalogError) throw new Error("isolated catalog offline");
        data = {models};
      } else if (route === "/api/position") {
        if (rejectPosition) return {ok: false, json: async () => ({detail: "Rust rejects position"})};
        data = {position: body, distances: [4, 3], legal_actions: [], paths: []};
      } else if (route === "/api/analyze") data = {winner: 1, candidates: []};
      else if (["/api/neural/tensor", "/api/neural/search", "/api/branch", "/api/apply"].includes(route)) data = {};
      else throw new Error(`Unexpected request ${path}`);
      return {ok: true, json: async () => data};
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, {configurable: true, writable: true, value});
    t.after(() => previous ? Object.defineProperty(globalThis, key, previous) : delete globalThis[key]);
  }
  const source = (await readFile(appURL, "utf8")).replace(/from "(\.\/[^\"]+)"/g,
    (_, path) => `from "${new URL(path, appURL).href}"`);
  const module = await import(`data:text/javascript;base64,${Buffer.from(source + `\nexport {model, api, safeReturnRoute};\n// ${sequence++}`).toString("base64")}`);
  await new Promise(resolve => setImmediate(resolve));
  return {...module, element, calls};
}

test("exact UTF8 context preserves turn, stocks, walls, provenance and distinct identities without analysis", async t => {
  const {model, calls, element} = await load(t);
  assert.deepEqual(model.context, context());
  assert.deepEqual(model.position, position);
  assert.deepEqual(calls.map(c => c.path), ["/research-lab/api/meta", "/research-lab/api/models", "/research-lab/api/position"]);
  assert.deepEqual(calls[2].body, position);
  assert.equal(model.modelId, "isolated-checkpoint");
  assert.equal(model.algorithm, "dqn");
  assert.equal(element("analyze-button").disabled, false);
  assert.equal(element("model-select").disabled, true);
  assert.match(element("context-banner").innerHTML, /Recorded model/);
  assert.match(element("context-banner").innerHTML, new RegExp(otherSHA));
  assert.match(element("context-banner").innerHTML, /&lt;script&gt;/);
  assert.match(element("context-banner").innerHTML, /返回来源/);
  assert.equal(element("turn-select").value, "2");
  assert.equal(element("p1-stock").textContent, "8 面墙");
  element("export-button").onclick();
  assert.deepEqual(JSON.parse(element("export-text").value).context, context());
  await element("analyze-button").onclick({});
  assert.deepEqual(calls.at(-1), {path: "/research-lab/api/analyze", body: {
    position, algorithm: "dqn", depth: 2, model_id: "isolated-checkpoint",
  }});
});

test("all injected neural and branch/apply requests use isolated prefix and pinned ID", async t => {
  const {api, calls} = await load(t);
  for (const path of ["/api/neural/tensor", "/api/neural/search", "/api/branch", "/api/apply"])
    await api(path, {position, algorithm: "dqn", model_id: "isolated-checkpoint"});
  assert.ok(calls.every(c => c.path.startsWith("/research-lab/api/")));
  const count = calls.length;
  await assert.rejects(api("/api/neural/search", {model_id: "not-a-lab-id"}), /精确 SHA/);
  await assert.rejects(api("/api/multi-compare", {model_ids: ["other"]}), /精确 SHA/);
  assert.equal(calls.length, count);
});

test("root-hosted Lab continues using its own root API", async t => {
  const {calls} = await load(t, {pathname: "/"});
  assert.deepEqual(calls.map(c => c.path), ["/api/meta", "/api/models", "/api/position"]);
});

for (const [name, models] of [
  ["alias/name matches but SHA differs", [entry(sha, otherSHA)]],
  ["SHA prefix collision", [entry("prefix", sha.slice(0, 12))]],
  ["same prefix different full SHA", [entry("near", otherSHA)]],
  ["exact SHA file unavailable", [{...entry(), available: false}]],
  ["duplicate ID maps to conflicting SHA", [entry(), entry("isolated-checkpoint", otherSHA)]],
  ["empty catalog", []],
]) test(`fails closed: ${name}`, async t => {
  const {model, element, calls} = await load(t, {models});
  assert.deepEqual(model.position, position);
  assert.equal(model.modelId, null);
  assert.equal(element("analyze-button").disabled, true);
  assert.match(element("context-banner").innerHTML, /神经分析已禁用/);
  await element("analyze-button").onclick({});
  assert.equal(calls.some(c => c.path.endsWith("/analyze")), false);
});

test("catalog failure preserves position but cannot silently select another checkpoint", async t => {
  const {model, element} = await load(t, {catalogError: true});
  assert.deepEqual(model.position, position);
  assert.equal(model.modelId, null);
  assert.equal(element("analyze-button").disabled, true);
});

for (const [name, encoded] of [
  ["empty", ""], ["invalid alphabet", "%%%"], ["padding", encode(context()) + "="],
  ["impossible length", "A"], ["noncanonical pad bits", "Zh"],
  ["invalid UTF8", Buffer.from([0xff]).toString("base64url")],
  ["invalid JSON", Buffer.from("{").toString("base64url")],
  ["oversized", "A".repeat(87_383)],
  ["wrong version", encode({...context(), schema_version: "v0"})],
  ["missing actual turn", encode({...context(), position: {...position, turn: undefined}})],
  ["partial analysis SHA", encode({...context(), analysis_model: {sha256: sha.slice(0, 12)}})],
  ["alias-only analysis model", encode({...context(), analysis_model: {id: "cnn"}})],
  ["duplicate context", encode(context()) + "&context=" + encode(context())],
]) test(`invalid context rejected without fallback or analysis: ${name}`, async t => {
  const {model, calls, element} = await load(t, {encoded});
  assert.equal(model.position, null);
  assert.equal(model.modelId, null);
  assert.equal(element("analyze-button").disabled, true);
  assert.equal(calls.some(c => c.path.endsWith("/position") || c.path.endsWith("/analyze")), false);
  assert.match(element("notice").textContent, /无法导入/);
});

test("no analysis_model never borrows recorded identity or default model", async t => {
  const {model, element} = await load(t, {encoded: encode({...context(), analysis_model: null})});
  assert.deepEqual(model.position, position);
  assert.equal(model.modelId, null);
  assert.equal(element("analyze-button").disabled, true);
});

test("engine validation rejection does not fall back or analyze", async t => {
  const {model, calls, element} = await load(t, {rejectPosition: true});
  assert.equal(model.position, null);
  assert.match(element("notice").textContent, /Rust rejects/);
  assert.equal(calls.some(c => c.path.endsWith("/analyze")), false);
});

test("return route is escaped and strictly internal; unsafe routes do not render a link", async t => {
  const {safeReturnRoute, element} = await load(t, {encoded: encode({...context(), return_route: "//evil.test/models"})});
  for (const route of ["//evil.test/models", "https://evil.test", "javascript:alert(1)", "/models/../../api/train", "/models/%2e%2e/%2e%2e/api/train", "/models/\\evil", "/models/%0aevil", "/api/models", "/models/%ZZ"])
    assert.equal(safeReturnRoute(route), null, route);
  assert.equal(safeReturnRoute(context().return_route), context().return_route);
  assert.doesNotMatch(element("context-banner").innerHTML, /href=/);
});

test("manual JSON import validates but never runs analysis", async t => {
  const {element, calls} = await load(t);
  element("import-text").value = JSON.stringify(position);
  await element("confirm-import").onclick();
  assert.equal(calls.filter(c => c.path.endsWith("/position")).length, 2);
  assert.equal(calls.some(c => c.path.endsWith("/analyze")), false);
});

test("HTML assets are prefix-relative for reverse proxy deployment", async () => {
  const html = await readFile(new URL("../../web/index.html", import.meta.url), "utf8");
  assert.match(html, /href="\.\/assets\/style.css"/);
  assert.match(html, /src="\.\/assets\/app.js/);
  assert.doesNotMatch(html, /(?:src|href)="\/assets\//);
});