import {boardSVG, escapeHTML as esc, numeric, scoreText} from "./board.js";
import {createNeuralPanel, hierarchyHTML, outputLabel} from "./neural.js";

const $ = id => document.getElementById(id);
const model = {
  meta: null, position: null, snapshot: null, analysis: null, branch: null,
  algorithm: "greedy", depth: 2, map: "pawn", tool: "p1", editing: false,
  selected: null, preview: null, previewLabel: "", pvIndex: null,
  history: [], comparisons: [], preset: "opening", title: "标准开局",
  revision: 0, branchRevision: 0, previewRevision: 0, busy: false, comparing: false,
  dataSnapshot: null,
  models: [], modelId: null, catalogMessage: "",
};
const kindName = {pawn: "走子", horizontal: "横墙", vertical: "竖墙"};
const boundName = {exact: "精确值", upper: "上界", lower: "下界"};
const boundValue = item => `${item.bound === "upper" ? "≤ " : item.bound === "lower" ? "≥ " : ""}${scoreText(item.value)}`;
const neuralScore = () => outputLabel(model.analysis?.neural?.model || model.models.find(m => m.id === model.modelId));
const candidate = id => model.analysis?.candidates.find(c => c.id === id);
const copy = object => structuredClone(object);
const signed = value => value > 0 ? `+${value}` : String(value);
const neuralPanel = createNeuralPanel({api, onCandidate: chooseCandidate});

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(Array.isArray(detail)
      ? detail.map(e => `${e.loc.join(".")}: ${e.msg}`).join("\n")
      : typeof detail === "string" ? detail : `请求失败 (${response.status})`);
  }
  return data;
}

function notice(message = "", error = false) {
  $("notice").hidden = !message;
  $("notice").textContent = message;
  $("notice").className = `notice${error ? " error" : ""}`;
}

function requestBody() {
  return {position: copy(model.position), algorithm: model.algorithm, depth: model.depth,
    ...(model.algorithm === "dqn" ? {model_id: model.modelId} : {})};
}

function controls() {
  document.body.classList.toggle("busy", model.busy);
  $("analyze-button").disabled = model.busy || !model.position || (model.algorithm === "dqn" && !model.modelId);
  $("analyze-button").textContent = model.busy ? "分析中…" : "运行分析";
  $("compare-button").disabled = model.busy || model.comparing || !model.position;
  $("compare-button").textContent = model.comparing ? "对比计算中…" : model.algorithm === "dqn" ? "对比当前模型与基准" : "对比 Greedy / Minimax D1–3";
  $("undo-button").disabled = !model.history.length || model.busy;
  $("export-button").disabled = !model.position;
  $("depth-control").hidden = model.algorithm !== "minimax";
  document.querySelectorAll("[data-depth]").forEach(b => {
    b.classList.toggle("active", Number(b.dataset.depth) === model.depth);
    b.disabled = model.algorithm !== "minimax";
    b.setAttribute("aria-pressed", String(Number(b.dataset.depth) === model.depth));
  });
  document.querySelectorAll("[data-algorithm]").forEach(b => {
    const active = b.dataset.algorithm === model.algorithm;
    b.classList.toggle("active", active);
    b.setAttribute("aria-pressed", String(active));
  });
  const entry = model.models.find(m => m.id === model.modelId);
  const hierarchical = Boolean(entry?.metadata?.model_config?.cnn_hierarchical);
  $("algorithm-description").textContent = model.algorithm === "greedy"
    ? "评分：Δ对手距离 − Δ自己距离"
    : model.algorithm === "minimax" ? "搜索：Minimax / Alpha–Beta"
    : hierarchical ? "决策：类别 Q → 类别内动作 Q" : `评分：${outputLabel(entry)} · 合法动作 Top1`;
  $("model-control").hidden = model.algorithm !== "dqn";
  $("model-select").value = model.modelId || "";
  $("tie-policy").textContent = model.algorithm === "dqn"
    ? (hierarchical ? "类别同分选走子；类内最小索引" : "最小动作索引") : "引擎顺序优先";
  $("model-meta").innerHTML = entry
    ? `<p>${esc(entry.architecture.toUpperCase())} · ep ${entry.metadata.episode_count ?? "?"}</p>${entry.training_status === "failed" ? "<p>训练运行中断 · 当前 checkpoint 已诊断</p>" : ""}<code title="${entry.sha256}">${entry.sha256.slice(0, 12)}</code><p>${entry.registry ? `${esc(entry.registry.name)} v${esc(entry.registry.version)}` : "未关联 Registry 版本"}</p>`
    : esc(model.catalogMessage || "无可用 checkpoint");
  $("edit-button").textContent = model.editing ? "完成编辑" : "编辑棋盘";
  $("edit-button").setAttribute("aria-pressed", String(model.editing));
  $("editor").hidden = !model.editing;
  $("turn-select").value = String(model.position?.turn || 1);
  document.querySelectorAll("[data-tool]").forEach(b => b.classList.toggle("active", b.dataset.tool === model.tool));
  $("presets").value = model.preset || "";
  document.querySelectorAll("[data-map]").forEach(b => b.classList.toggle("active", b.dataset.map === model.map));
}

function renderBoard() {
  if (!model.snapshot) return;
  const view = model.preview || model.snapshot, p = view.position;
  $("position-title").textContent = model.preview ? model.previewLabel : model.title;
  $("turn-badge").textContent = view.winner ? `P${view.winner} 已到终点` : `P${p.turn} 行动 ${p.turn === 1 ? "↑" : "↓"}`;
  [1, 2].forEach(n => $(`p${n}-stock`).textContent = `${p.walls_remaining[n-1]} 面墙`);
  $("board").innerHTML = boardSVG({
    snapshot: view, candidates: model.analysis?.candidates || [],
    selected: model.selected, recommended: model.analysis?.selected_id,
    map: model.map, paths: $("paths-toggle").checked, editing: model.editing,
    tool: model.tool, preview: Boolean(model.preview),
  });
  $("root-button").hidden = !model.preview;
  $("summary").innerHTML = `
    <div class="summary-item"><small>P1 最短通路</small><strong>${view.distances[0]}<em>步</em></strong></div>
    <div class="summary-item"><small>P2 最短通路</small><strong>${view.distances[1]}<em>步</em></strong></div>
    <div class="summary-item"><small>${model.preview ? "预览局面合法动作" : "当前合法动作"}</small><strong>${view.legal_actions.length}<em>个</em></strong></div>`;
}

function renderDecision() {
  const analysis = model.analysis;
  if (!analysis) {
    $("decision").innerHTML = `<p class="${model.busy ? "loading-dots" : "muted"}">${model.busy ? "计算中" : "无分析结果"}</p>`;
    return;
  }
  if (analysis.winner) {
    $("decision").innerHTML = `<strong>终局 · P${analysis.winner} 胜</strong><p>候选动作：0</p>`;
    return;
  }
  const chosen = candidate(analysis.selected_id), selected = candidate(model.selected) || chosen;
  const isChoice = selected.id === chosen.id;
  const before = selected.before_self_opp, after = selected.after_self_opp;
  const exact = model.branch?.action_id === selected.id ? model.branch : null;
  const value = exact || selected;
  let explanation = "";
  if (model.algorithm === "greedy") {
    explanation = `<div class="distance-row"><span>自己距离</span><b>${before[0]} → ${after[0]} (${signed(selected.delta_self)})</b></div>
      <div class="distance-row"><span>对手距离</span><b>${before[1]} → ${after[1]} (${signed(selected.delta_opp)})</b></div>
      <div class="equation"><span>Δ对手 − Δ自己</span><code>${selected.delta_opp} − (${selected.delta_self}) = ${scoreText(selected.greedy_score)}</code></div>
      <div class="distance-row"><span>终局奖励</span><b>无</b></div>`;
  } else if (model.algorithm === "minimax") {
    explanation = `<div class="distance-row"><span>视角 / 深度</span><b>P${analysis.position.turn} / ${analysis.depth} ply</b></div>
      <div class="distance-row"><span>叶子评价</span><b>对手距离 − 自己距离</b></div>
      <div class="distance-row"><span>终局评价</span><b>胜 +∞ / 负 −∞</b></div>
      <div class="distance-row"><span>落子后距离 · 自己 / 对手</span><b>${after[0]} / ${after[1]}</b></div>
      <div class="distance-row"><span>根搜索返回 · ${boundName[selected.bound]}</span><b>${boundValue(selected)}</b></div>
      <div class="equation"><span>全窗口分支值</span><code>${exact ? boundValue(exact) : "计算中…"}</code></div>`;
  } else {
    const n = analysis.neural;
    explanation = `${hierarchyHTML(n.hierarchy)}<div class="distance-row"><span>网络原始动作 ${neuralScore()}</span><b title="${selected.value}">${scoreText(selected.value)}</b></div>
      ${selected.route_eligible === false ? '<p class="small">此动作合法，但其类别未被第一级选中。</p>' : ""}
      <div class="distance-row"><span>Top1 − Top2（${n.hierarchy ? "选定类别内" : "合法"}）</span><b>${n.legal_margin === null ? "—" : scoreText(n.legal_margin)}</b></div>
      <div class="distance-row"><span>未屏蔽最高 ${neuralScore()} 动作</span><b>${esc(n.raw_best.label)} · ${n.raw_best.legal ? "合法" : "非法"}</b></div>
      <div class="distance-row"><span>高于 Top1 的非法输出</span><b>${n.illegal_above_selected}</b></div>
      <div class="distance-row"><span>落子后距离 · 自己 / 对手</span><b>${after[0]} / ${after[1]}</b></div>
      <p class="result-metrics">${neuralScore() === "logit" ? "logit 为策略分数，不是预期回报或胜率。" : "Q 是模型评分，不是胜率。"}距离只作为规则诊断。</p>
      <a class="text-button" href="#neural-panel">输入 / 激活 / 权重 / 敏感度</a>`;
  }
  $("decision").innerHTML = `<div class="decision-kicker"><span>${isChoice ? "算法选择" : "检查候选"} · ${kindName[selected.action.type]}</span><span>${isChoice ? "Top1" : `Top1: ${esc(chosen.label)}`}</span></div>
    <div class="decision-main"><strong>${esc(selected.label)}</strong><span>${boundValue(value)}</span></div>
    <div class="action-id"><code>${esc(selected.id)}</code><span>${model.algorithm === "dqn" ? "动作索引" : "引擎序号"} ${selected.order}</span></div>
    ${selected.immediate_win ? `<p>落子后终局：胜</p>` : ""}${explanation}
    <p class="result-metrics">${model.algorithm === "dqn" ? `${analysis.neural.model.parameters.toLocaleString()} 参数 · CPU · 诊断` : `根搜索：${analysis.stats.nodes.toLocaleString()} 节点 · ${analysis.stats.cutoffs} 剪枝`} · ${(analysis.stats.seconds*1000).toFixed(0)} ms</p>`;
}

function renderCandidates() {
  const list = model.analysis?.candidates || [];
  const filter = $("candidate-filter").value;
  const visible = list.filter(c => filter === "all" || c.action.type === filter);
  // Bounds stay as reported by the original search, even after inspecting one branch.
  // Their order is a display aid, not an exact ranking of inferior actions.
  visible.sort((a, b) => {
    if (a.route_eligible !== b.route_eligible) return a.route_eligible ? -1 : 1;
    const av = numeric(a.value), bv = numeric(b.value);
    return av === bv ? a.order-b.order : av > bv ? -1 : 1;
  });
  $("candidate-count").textContent = list.length ? `${visible.length} / ${list.length}` : "";
  $("candidate-order-label").textContent = model.algorithm === "dqn" ? "索引" : "序号";
  $("candidate-order-label").title = model.algorithm === "dqn" ? "DQN 动作编码 0–208" : "规则引擎顺序，从 0 开始";
  $("bounds-note").textContent = model.algorithm === "minimax"
    ? "按根搜索返回值排序；上界不代表精确排名。选择动作可全窗口重搜。序号为引擎原始顺序（从 0 开始）。"
    : model.algorithm === "dqn" ? (model.analysis?.neural?.hierarchy ? "先列第一级选定类别，再按类别内 Q 排序。其他类别仍可检查，推理时不参与第二级选择。" : `仅列合法动作，按 ${neuralScore()} 降序；同分取动作索引最小值。全部 209 个输出见网络诊断。`) : "按分数降序排列；同分取引擎序号最小的动作。序号从 0 开始。";
  $("candidates").innerHTML = visible.length ? visible.map(c => `<button class="candidate-row${c.id === model.selected ? " active" : ""}" data-candidate="${c.id}" aria-pressed="${c.id === model.selected}">
    <span class="order">${c.order}</span>
    <span class="move">${esc(c.label)}<small>${c.id === model.analysis.selected_id ? "◇ 选择" : c.route_eligible === false ? "类别未选" : kindName[c.action.type]}</small></span>
    <span class="value" title="${c.value}">${boundValue(c)}</span><span class="bound ${c.bound}">${model.algorithm === "dqn" ? neuralScore() : boundName[c.bound]}</span></button>`).join("")
    : `<div class="empty">${model.busy ? "计算中…" : "无候选动作"}</div>`;
}

function treeHTML(trace) {
  const nodes = new Map(trace.nodes.map(n => [n.id, n]));
  const render = n => {
    const omittedChildren = n.visited_children - n.children.length;
    const label = `<b>${esc(n.label)}</b> ${n.role} · ${boundValue(n)} <span class="bound">${boundName[n.bound]}</span>`;
    const inspect = `<button class="text-button" data-node="${n.id}">局面</button>`;
    if (n.leaf) return `<div class="tree-leaf">${label}${inspect}</div>`;
    return `<details${n.parent === null ? " open" : ""}><summary>${label}${inspect}${n.pruned ? `<span class="prune">剪去 ${n.pruned} 个直接子分支</span>` : ""}</summary>
      <div class="small muted">进入窗口 [${scoreText(n.alpha)}, ${scoreText(n.beta)}] · 剩余 ${n.depth_remaining} 半步</div>
      ${n.children.map(id => nodes.has(id) ? render(nodes.get(id)) : "").join("")}
      ${omittedChildren ? `<div class="tree-leaf">未记录已访问子分支：${omittedChildren}</div>` : ""}</details>`;
  };
  return trace.nodes.filter(n => n.parent === null).map(render).join("");
}

function renderBranch() {
  const branch = model.branch;
  if (!model.selected || !model.analysis) { $("branch-panel").innerHTML = ""; return; }
  if (!branch) {
    $("branch-panel").innerHTML = `<p class="small muted loading-dots">分支计算中…</p>`;
    return;
  }
  const trace = branch.trace;
  $("branch-panel").innerHTML = `<div class="branch-heading"><h3>${model.algorithm === "minimax" ? "主变化 PV" : "动作结果"}</h3><span class="pill">${model.algorithm === "dqn" ? neuralScore() : "精确值"} ${scoreText(branch.value)}</span></div>
    <div class="pv">${branch.principal_variation.map((step, i) => `<button data-pv="${i}" class="${model.pvIndex === i ? "active" : ""}"><small>${i+1} · P${step.action.player}</small>${esc(step.label)}</button>`).join("")}</div>
    <button id="apply-candidate" class="button secondary wide">应用动作 ${esc(candidate(model.selected).label)}</button>
    <p class="branch-metrics">${model.algorithm === "dqn" ? "输入梯度已计算 · 见网络诊断 / 敏感度" : `分支：${branch.stats.nodes.toLocaleString()} 节点 · ${branch.stats.cutoffs} 剪枝`} · ${(branch.stats.seconds*1000).toFixed(0)} ms</p>
    ${trace.nodes.length ? `<h3>搜索树</h3><p class="small muted">评分视角：根局面行动方。MAX 最大化，MIN 最小化。</p>
      <div class="tree">${treeHTML(trace)}</div>
      <p class="small muted">记录 ${trace.recorded} / ${trace.visited} 节点${trace.omitted ? `；省略 ${trace.omitted} 个记录，搜索值与 PV 完整` : ""}。</p>` : ""}`;
}

function render() {
  controls(); renderBoard(); renderDecision(); renderCandidates(); renderBranch();
  neuralPanel.setAnalysis(model.analysis); neuralPanel.setBranch(model.branch);
  $("source-version").textContent = (model.analysis?.provenance || model.meta?.provenance)?.source_sha256.slice(0, 12) || "…";
}

function clearPreview() {
  model.previewRevision++;
  model.preview = null; model.pvIndex = null;
}

async function chooseCandidate(id) {
  if (!candidate(id)) return;
  const revision = model.revision, branchRevision = ++model.branchRevision;
  const selected = candidate(id);
  model.selected = id; model.branch = null;
  model.map = selected.action.type; clearPreview(); render();
  try {
    const result = await api("/api/branch", {...requestBody(), action: selected.action});
    if (revision !== model.revision || branchRevision !== model.branchRevision) return;
    model.branch = result; renderDecision(); renderBranch(); neuralPanel.setBranch(result);
  } catch (error) {
    if (revision !== model.revision || branchRevision !== model.branchRevision) return;
    $("branch-panel").innerHTML = `<p class="error-text">${esc(error.message)}</p>`;
  }
}

async function analyze() {
  if (!model.position) return;
  const revision = ++model.revision;
  model.analysis = null; model.branch = null; model.selected = null;
  model.busy = true; model.editing = false; clearPreview(); notice(); render();
  if (model.algorithm === "dqn" && !model.modelId) {
    model.busy = false; render(); notice(model.catalogMessage || "请先选择可用 checkpoint。", true); return;
  }
  try {
    const result = await api("/api/analyze", requestBody());
    if (revision !== model.revision) return;
    model.analysis = result; model.busy = false; render();
    if (result.selected_id) await chooseCandidate(result.selected_id);
  } catch (error) {
    if (revision !== model.revision) return;
    model.busy = false; render(); notice(error.message, true);
  }
}

async function setPosition(position, title, {preset = null, history = true, run = true} = {}) {
  // Validate first: rejected edits leave the current position intact.
  const revision = ++model.revision;
  model.branchRevision++; model.busy = true; controls();
  let validated;
  try { validated = await api("/api/position", position); }
  catch (error) {
    if (revision === model.revision) { model.busy = false; controls(); }
    throw error;
  }
  if (revision !== model.revision) return;
  if (history && model.position) model.history.push({position: copy(model.position), title: model.title, preset: model.preset});
  model.position = validated.position; model.snapshot = validated; model.title = title;
  model.preset = preset; model.analysis = null; model.branch = null; model.selected = null;
  model.comparisons = []; $("comparison").hidden = true;
  model.busy = false; clearPreview(); notice(); render();
  if (run) await analyze();
}

async function editBoard(event) {
  if (model.busy) return;
  const target = event.target.closest("[data-cell], [data-anchor], [data-wall], [data-action]");
  if (!target) return;
  if (!model.editing) {
    if (target.dataset.action) await chooseCandidate(target.dataset.action);
    return;
  }
  const position = copy(model.position);
  const owner = position.turn - 1;
  if (target.dataset.wall) {
    const [orientation, x, y] = target.dataset.wall.split(":");
    position.walls = position.walls.filter(w => !(w.orientation === orientation && w.x === +x && w.y === +y));
    const refund = position.walls_remaining[owner] < 10 ? owner : 1-owner;
    position.walls_remaining[refund]++;
  } else if (target.dataset.anchor && ["horizontal", "vertical"].includes(model.tool)) {
    const [x, y] = target.dataset.anchor.split(",").map(Number);
    const spend = position.walls_remaining[owner] ? owner : 1-owner;
    if (!position.walls_remaining[spend]) { notice("所有 20 面墙已在棋盘上，请先移除一面。", true); return; }
    position.walls.push({orientation: model.tool, x, y});
    position.walls_remaining[spend]--;
  } else if (target.dataset.cell && ["p1", "p2"].includes(model.tool)) {
    position.pawns[model.tool === "p1" ? 0 : 1] = target.dataset.cell.split(",").map(Number);
  } else return;
  await setPosition(position, "自定义局面", {run: false});
}

function renderComparison() {
  $("comparison-results").innerHTML = model.comparisons.map((row, index) => {
    const chosen = row.result?.candidates.find(c => c.id === row.result.selected_id);
    return `<div class="compare-row"><span>${esc(row.name)}</span>${row.error
      ? `<span class="error-text">${esc(row.error)}</span>`
      : row.result ? `<strong>${chosen ? esc(chosen.label) : "终局"}</strong><span>${chosen ? boundValue(chosen) : ""}</span><button class="text-button" data-compare="${index}">检查</button>`
      : `<span class="small muted loading-dots">计算中</span>`}</div>`;
  }).join("");
}

async function compare() {
  const position = copy(model.position), positionKey = JSON.stringify(position);
  model.comparing = true; model.comparisons = [];
  $("comparison").hidden = false; controls();
  try {
    const configs = [["greedy", 1, "Greedy"], ["minimax", 1, "Minimax · 1 ply"], ["minimax", 2, "Minimax · 2 ply"], ["minimax", 3, "Minimax · 3 ply"]];
    if (model.algorithm === "dqn" && model.modelId) configs.unshift(["dqn", 1, model.modelId, model.modelId]);
    for (const [algorithm, depth, name, modelId] of configs) {
      if (JSON.stringify(model.position) !== positionKey) break;
      const row = {algorithm, depth, name, modelId};
      model.comparisons.push(row); renderComparison();
      try { row.result = await api("/api/analyze", {position, algorithm, depth, ...(modelId ? {model_id: modelId} : {})}); }
      catch (error) { row.error = error.message; }
      if (JSON.stringify(model.position) !== positionKey) break;
      renderComparison();
    }
  } finally { model.comparing = false; controls(); }
}

async function multiCompare() {
  const ids = [...$("multi-model-select").selectedOptions].map(option => option.value);
  if (!ids.length) throw new Error("请选择至少一个 CNN checkpoint。");
  model.comparing = true; $("comparison").hidden = false; controls();
  try {
    const result = await api("/api/multi-compare", {position: copy(model.position), model_ids: ids});
    const optimal = new Set(result.minimax_d2.equivalent_optimal_ids);
    $("comparison-results").innerHTML = [
      `<p><strong>D2 等价最优：</strong>${result.minimax_d2.actions.filter(a => optimal.has(a.id)).map(a => esc(a.id)).join("、")}</p>`,
      ...result.models.map(item => `<div class="compare-row"><span>${esc(item.model_id)}</span><strong>${esc(item.selected_id)}</strong><span>${item.selected_d2_optimal ? "D2 最优" : "D2 次优"}</span><button class="text-button" data-multi-action="${esc(JSON.stringify(item.top_actions[0].action))}">预览</button></div>`),
    ].join("");
    model.comparisons = [{name: "multi-agent", result}];
  } finally { model.comparing = false; controls(); }
}

function safe(fn) {
  return event => Promise.resolve().then(() => fn(event)).catch(error => notice(error.message, true));
}

$("analyze-button").onclick = safe(analyze);
$("compare-button").onclick = safe(compare);
$("multi-compare-button").onclick = safe(multiCompare);
$("model-select").onchange = safe(async event => { model.modelId = event.target.value; await analyze(); });
$("algorithm-tabs").onclick = safe(async event => {
  const button = event.target.closest("[data-algorithm]");
  if (!button || button.dataset.algorithm === model.algorithm) return;
  model.algorithm = button.dataset.algorithm; await analyze();
});
$("depth-control").onclick = safe(async event => {
  const button = event.target.closest("[data-depth]");
  if (!button || model.algorithm !== "minimax" || +button.dataset.depth === model.depth) return;
  model.depth = +button.dataset.depth; await analyze();
});
$("presets").onchange = safe(async event => {
  const item = model.meta.presets.find(p => p.id === event.target.value);
  if (item) await setPosition(item.position, item.title, {preset: item.id});
});
$("map-tabs").onclick = event => {
  const button = event.target.closest("[data-map]");
  if (button) { model.map = button.dataset.map; controls(); renderBoard(); }
};
$("candidates").onclick = safe(event => {
  const button = event.target.closest("[data-candidate]");
  if (button) return chooseCandidate(button.dataset.candidate);
});
$("candidate-filter").onchange = renderCandidates;
$("paths-toggle").onchange = renderBoard;
$("root-button").onclick = () => { clearPreview(); renderBoard(); renderBranch(); };
$("board").onclick = safe(editBoard);
$("board").onkeydown = event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); safe(editBoard)(event); }
};
$("branch-panel").onclick = safe(async event => {
  if (!model.branch) return;
  const step = event.target.closest("[data-pv]");
  const node = event.target.closest("[data-node]");
  if (step) {
    model.previewRevision++;
    model.pvIndex = +step.dataset.pv;
    model.preview = model.branch.principal_variation[model.pvIndex];
    model.previewLabel = `PV ${model.pvIndex+1} · ${model.preview.label}`;
    renderBoard(); renderBranch();
  } else if (node) {
    event.preventDefault();
    const treeNode = model.branch.trace.nodes.find(n => n.id === +node.dataset.node);
    const revision = model.revision, branchRevision = model.branchRevision, previewRevision = ++model.previewRevision;
    const result = await api("/api/position", treeNode.position);
    if (revision !== model.revision || branchRevision !== model.branchRevision || previewRevision !== model.previewRevision) return;
    model.preview = result; model.previewLabel = `搜索节点 ${treeNode.id} · ${treeNode.label}`;
    model.pvIndex = null; renderBoard();
  } else if (event.target.closest("#apply-candidate")) {
    const revision = model.revision, action = candidate(model.selected).action;
    const next = await api("/api/apply", {...requestBody(), action});
    if (revision !== model.revision) return;
    await setPosition(next.position, `${candidate(model.selected).label} 后局面`);
  }
});
$("edit-button").onclick = safe(async () => {
  if (model.busy) return;
  model.editing = !model.editing; clearPreview(); render();
  if (!model.editing) await analyze();
});
document.querySelector(".edit-tools").onclick = event => {
  const button = event.target.closest("[data-tool]");
  if (button) { model.tool = button.dataset.tool; controls(); renderBoard(); }
};
$("turn-select").onchange = safe(event => setPosition(
  {...copy(model.position), turn: Number(event.target.value)}, "自定义局面", {run: false}));
$("undo-button").onclick = safe(async () => {
  const item = model.history.pop();
  if (item) await setPosition(item.position, item.title, {preset: item.preset, history: false, run: !model.editing});
});
$("reset-button").onclick = safe(() => setPosition(model.meta.presets[0].position, "标准开局", {preset: "opening"}));
$("close-comparison").onclick = () => { $("comparison").hidden = true; };
$("comparison-results").onclick = safe(async event => {
  const preview = event.target.closest("[data-multi-action]");
  if (preview) {
    const next = await api("/api/apply", {position: copy(model.position), action: JSON.parse(preview.dataset.multiAction)});
    model.preview = next; model.previewLabel = "多模型建议后局面"; renderBoard(); return;
  }
  const button = event.target.closest("[data-compare]");
  if (!button) return;
  const row = model.comparisons[+button.dataset.compare];
  if (!row.result) return;
  model.revision++; model.busy = false; model.algorithm = row.algorithm;
  if (row.algorithm === "minimax") model.depth = row.depth;
  if (row.algorithm === "dqn") model.modelId = row.modelId;
  model.analysis = row.result; model.branch = null; model.selected = row.result.selected_id;
  model.editing = false; clearPreview(); render();
  if (model.selected) await chooseCandidate(model.selected);
});
$("import-button").onclick = () => { $("import-error").textContent = ""; $("import-dialog").showModal(); };
$("close-import").onclick = () => $("import-dialog").close();
$("import-example").onclick = () => { $("import-text").value = JSON.stringify(model.position, null, 2); };
$("confirm-import").onclick = async () => {
  $("confirm-import").disabled = true;
  try {
    const text = $("import-text").value;
    if (text.length > 2_000_000) throw new Error("文件超过 2 MB，请仅导入其中的 position 字段。");
    const data = JSON.parse(text);
    await setPosition(data.position || data, "导入局面", {run: false});
    $("import-dialog").close(); await analyze();
  } catch (error) { $("import-error").textContent = error.message; }
  finally { $("confirm-import").disabled = false; }
};
function renderData() {
  const data = model.dataSnapshot;
  const scope = $("data-scope").value;
  const value = scope === "case" ? data
    : ["agent_spec", "candidates", "neural"].includes(scope) ? data.analysis?.[scope] ?? null
    : data[scope];
  $("export-text").value = JSON.stringify(value, null, 2);
  $("export-status").textContent = value === null ? "尚无数据" : `${(new Blob([$("export-text").value]).size / 1024).toFixed(1)} KB · 打开面板时的数据快照`;
}
$("data-scope").onchange = renderData;
$("export-button").onclick = () => {
  model.dataSnapshot = copy({schema: "quoridor-decision-case-v1", exported_at: new Date().toISOString(),
    title: model.title, position: model.position, provenance: model.analysis?.provenance || model.meta.provenance,
    analysis: model.analysis, inspected_branch: model.branch,
    neural_tensor: neuralPanel.currentTensor(),
    neural_search: neuralPanel.currentSearch(),
    comparison: model.comparisons.map(({name, result, error}) => ({name, result, error}))});
  renderData();
  $("export-dialog").showModal();
};
$("close-export").onclick = () => $("export-dialog").close();
$("select-export").onclick = () => { $("export-text").focus(); $("export-text").select(); };
$("download-case").onclick = () => {
  const url = URL.createObjectURL(new Blob([$("export-text").value], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url; link.download = `quoridor-${$("data-scope").value}-${Date.now()}.json`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  $("export-status").textContent = "已请求下载。若当前浏览器未保存文件，可全选 JSON 后复制保存。";
};

async function init() {
  model.meta = await api("/api/meta");
  const catalog = await api("/api/models").catch(error => ({models: [], message: error.message}));
  model.models = catalog.models; model.catalogMessage = catalog.message;
  model.modelId = model.models.find(m => m.available && m.architecture === "cnn")?.id || model.models.find(m => m.available)?.id || null;
  $("model-select").innerHTML = model.models.map(m => `<option value="${esc(m.id)}" ${m.available ? "" : "disabled"}>${esc(m.id)} · ${esc(m.architecture.toUpperCase())}</option>`).join("");
  $("multi-model-select").innerHTML = model.models.filter(m => m.available).map(m =>
    `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("");
  $("connection").textContent = "Engine connected"; $("connection").classList.add("ready");
  $("source-version").textContent = model.meta.provenance.source_sha256.slice(0, 12);
  $("presets").innerHTML = `<option value="" disabled>自定义局面</option>` + model.meta.presets.map(p => `<option value="${p.id}">${esc(p.title)}</option>`).join("");
  const encodedPosition = new URLSearchParams(window.location.search).get("position");
  if (encodedPosition) {
    try {
      const position = JSON.parse(decodeURIComponent(escape(atob(encodedPosition))));
      await setPosition(position, "Dashboard Replay", {preset: "", history: false});
      notice("已从 Training Dashboard 导入经回放记录还原的局面。");
      return;
    } catch {
      notice("无法导入 Dashboard Replay 局面。", true);
    }
  }
  await setPosition(model.meta.presets[0].position, "标准开局", {preset: "opening", history: false});
}
controls();
init().catch(error => { $("connection").textContent = "连接失败"; notice(error.message, true); });
