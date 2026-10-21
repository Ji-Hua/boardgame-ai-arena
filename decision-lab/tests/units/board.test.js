// TEST_CLASSIFICATION: SPECIFIED
// Spec: documents/agent/analysis/decision-analysis-interface.md
import {test} from "node:test";
import assert from "node:assert/strict";
import {cellPoint, wallRect, boardSVG, escapeHTML, numeric} from "../../web/board.js";

test("coordinate orientation keeps P1's goal at the top and walls on shared boundaries", () => {
  const a1 = cellPoint(0, 0), b1 = cellPoint(1, 0), a2 = cellPoint(0, 1);
  assert.ok(a1[0] < b1[0] && a1[1] > a2[1]);
  const h = wallRect("horizontal", 0, 0), v = wallRect("vertical", 0, 0);
  assert.equal(h[1]+h[3]/2, (a1[1]+a2[1])/2);
  assert.equal(v[0]+v[2]/2, (a1[0]+b1[0])/2);
  assert.ok(h[0] < a1[0] && h[0]+h[2] > b1[0]);
  assert.ok(v[1] < a2[1] && v[1]+v[3] > a1[1]);
});

test("a bound is visibly distinct from an exact score and preview removes root heatmap", () => {
  const input = {
    snapshot: {position: {pawns: [[4, 0], [4, 8]], walls: []}, paths: []},
    candidates: [{id: "pawn:4:1", action: {type: "pawn", target: [4, 1]}, bound: "upper", value: 0}],
  };
  assert.match(boardSVG(input), /≤0/);
  assert.match(boardSVG(input), /data-action="pawn:4:1"/);
  assert.doesNotMatch(boardSVG({...input, preview: true}), /data-action=/);
  assert.doesNotMatch(boardSVG({...input, editing: true}), /data-action=/);
});

test("JSON infinity sentinels remain ordered and user text is escaped", () => {
  assert.equal(numeric("+inf"), Infinity);
  assert.equal(numeric("-inf"), -Infinity);
  assert.equal(escapeHTML('<img onerror="x">'), "&lt;img onerror=&quot;x&quot;&gt;");
});
