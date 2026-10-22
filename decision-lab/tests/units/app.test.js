// TEST_CLASSIFICATION: UNSPECIFIED
// Regression guardrails for module startup and dashboard replay URL imports.
import {test} from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const appURL = new URL("../../web/app.js", import.meta.url);
const opening = {pawns: [[4, 0], [4, 8]], walls: [], walls_remaining: [10, 10], turn: 1};
let moduleId = 0;

async function loadApp(t, search = "") {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: id === "candidate-filter" ? "all" : "",
      textContent: "", innerHTML: "", checked: false, selectedOptions: [],
      classList: {add() {}, toggle() {}}, setAttribute() {},
    });
    return elements.get(id);
  };
  const calls = [];
  const globals = {
    document: {
      getElementById: element, querySelector: element, querySelectorAll: () => [],
      body: element("body"),
    },
    window: {location: {search}},
    fetch: async (path, options = {}) => {
      const body = options.body ? JSON.parse(options.body) : undefined;
      calls.push({path, body});
      let data;
      if (path === "/api/meta") data = {
        provenance: {source_sha256: "0123456789abcdef"},
        presets: [{id: "opening", title: "标准开局", position: opening}],
      };
      else if (path === "/api/models") data = {models: []};
      else if (path === "/api/position") data = {
        position: body, distances: [8, 8], legal_actions: [], paths: [],
      };
      // A terminal analysis avoids unrelated candidate/branch API requests.
      else if (path === "/api/analyze") data = {winner: 1, candidates: []};
      else throw new Error(`Unexpected request: ${path}`);
      return {ok: true, json: async () => data};
    },
  };
  for (const [name, value] of Object.entries(globals)) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, {configurable: true, writable: true, value});
    t.after(() => {
      if (original) Object.defineProperty(globalThis, name, original);
      else delete globalThis[name];
    });
  }
  // Execute the real ES module (including strict lexical scope), with real
  // imports and only browser/network boundaries replaced by lightweight fakes.
  const source = (await readFile(appURL, "utf8")).replace(
    /from "(\.\/[^\"]+)"/g, (_, path) => `from "${new URL(path, appURL).href}"`,
  );
  await import(`data:text/javascript;base64,${Buffer.from(`${source}\n// startup ${moduleId++}`).toString("base64")}`);
  // init() is intentionally fire-and-forget; drain its mocked async requests.
  await new Promise(resolve => setImmediate(resolve));
  return {element, calls};
}

test("module initialization binds multiCompare and reaches default position startup without a real DOM", async t => {
  const {element, calls} = await loadApp(t);
  assert.equal(typeof element("multi-compare-button").onclick, "function");
  assert.equal(typeof element("download-case").onclick, "function");
  assert.equal(element("connection").textContent, "Engine connected");
  assert.deepEqual(calls.map(call => call.path), ["/api/meta", "/api/models", "/api/position"]);
  assert.deepEqual(calls.find(call => call.path === "/api/position").body, opening);
  await element("multi-compare-button").onclick({});
  assert.equal(element("notice").textContent, "请选择至少一个 CNN checkpoint。");
});

test("startup imports the dashboard encoded position instead of the opening preset", async t => {
  const position = {
    pawns: [[3, 5], [5, 2]], walls: [{orientation: "horizontal", x: 2, y: 3}],
    walls_remaining: [9, 10], turn: 2,
  };
  const search = `?${new URLSearchParams({position: Buffer.from(JSON.stringify(position), "utf8").toString("base64")})}`;
  const {element, calls} = await loadApp(t, search);
  assert.deepEqual(calls.filter(call => call.path === "/api/position").map(call => call.body), [position]);
  assert.equal(calls.some(call => call.path === "/api/analyze"), false);
  assert.equal(element("position-title").textContent, "Dashboard Replay");
  assert.equal(element("turn-select").value, "2");
  assert.equal(element("notice").textContent, "已从 Training Dashboard 导入经回放记录还原的局面。");
  assert.equal(element("notice").className, "notice");
});