// TEST_CLASSIFICATION: SPECIFIED
// Spec: documents/agent/analysis/decision-analysis-interface.md
import {test} from "node:test";
import assert from "node:assert/strict";
import {tensorSVG, hierarchyHTML, outputLabel, searchHTML} from "../../web/neural.js";
import {scoreText} from "../../web/board.js";

test("policy methods expose logits and DDQN exposes action values", () => {
  for (const algorithm of ["policy_distillation", "ppo", "ppo_imitation", "selfplay_ppo", "value_calibration", "alphazero_lite"]) {
    assert.equal(outputLabel({metadata: {model_config: {algorithm}}}), "logit");
  }
  assert.equal(outputLabel({metadata: {model_config: {algorithm: "double_dqn"}}}), "Q");
  assert.equal(outputLabel({metadata: {eval_summary: {output_semantics: "policy_logits"}}}), "logit");
});

test("spatial tensor maps x/y to board coordinates, vector keeps flat offsets", () => {
  const xy = [[1,2],[3,4]];
  const svg = tensorSVG(xy,"board_xy");
  assert.match(svg, /x="30" y="30"[^>]*><title>\[0,1\]: 2/);
  assert.match(svg, /x="62" y="62"[^>]*><title>\[1,0\]: 3/);
  const vector = tensorSVG([[1,2],[3]],"vector",256);
  assert.match(vector, /index 256: 1/);
  assert.match(vector, /index 272: 3/);
  assert.doesNotMatch(vector, /NaN|undefined/);
});

test("zero activations and divergent but finite Q values render without invalid numbers", () => {
  assert.doesNotMatch(tensorSVG([[0,0],[0,0]],"matrix"), /NaN|Infinity/);
  assert.equal(scoreText(1e24), "1.000e+24");
  assert.equal(scoreText("-inf"), "−∞");
});

test("hierarchical decisions expose type choice, conditional action and unavailable categories", () => {
  const html = hierarchyHTML({selected_type:"pawn", types:[
    {id:"pawn",label:"走子",q:.2,available:true,legal_actions:3,best_action:{label:"e2",value:.1}},
    {id:"wall",label:"放墙",q:100,available:false,legal_actions:0,best_action:null}
  ]});
  assert.match(html, /走子 · 选择/);
  assert.match(html, /类别内 Top1：e2/);
  assert.match(html, /无合法动作，类别已屏蔽/);
  assert.equal(hierarchyHTML(null), "");
});


test("PUCT exposes parent-view Q, visits and selected move separately from prior", () => {
  const html = searchHTML({perspective:2,root_value:-.2,seconds:.01,stats:{simulations:50,model_evaluations:39},
    selected_action_index:37,actions:[{action_index:37,id:"pawn:4:1",label:"e2",prior:.2,visits:31,visit_probability:.62,q:.3}]});
  assert.match(html, /P2 Value/);
  assert.match(html, /50 simulations/);
  assert.match(html, /width:62%/);
  assert.match(html, /chosen/);
  assert.doesNotMatch(html, /NaN|undefined/);
});
