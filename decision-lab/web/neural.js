import {escapeHTML as esc, scoreText as fmt} from "./board.js";

const modes = {input: "输入编码", outputs: "全部动作输出", activation: "层激活", weight: "权重", sensitivity: "输入敏感度", mcts: "PUCT / Value"};
export const outputLabel = entry => entry?.output_semantics === "policy_logits" || entry?.metadata?.output_semantics === "policy_logits" || entry?.metadata?.eval_summary?.output_semantics === "policy_logits" || ["policy_distillation", "ppo", "ppo_imitation", "selfplay_ppo", "value_calibration", "alphazero_lite"].includes(entry?.metadata?.model_config?.algorithm) ? "logit" : "Q";
const options = (items, selected) => items.map(([id, label]) => `<option value="${esc(id)}" ${String(id) === String(selected) ? "selected" : ""}>${esc(label)}</option>`).join("");

export function hierarchyHTML(hierarchy) {
  if (!hierarchy) return "";
  return `<div class="type-routing"><h3>第一级 · 动作类别 Q</h3>${hierarchy.types.map(t =>
    `<div class="type-choice ${t.id === hierarchy.selected_type ? "chosen" : ""}"><b>${esc(t.label)}${t.id === hierarchy.selected_type ? " · 选择" : ""}</b><code>${fmt(t.q)}</code>
    <small>${t.available ? `${t.legal_actions} 个合法动作 · 类别内 Top1：${esc(t.best_action.label)} (${fmt(t.best_action.value)})` : "无合法动作，类别已屏蔽"}</small></div>`).join("")}
    <p class="small muted">第二级只在选定类别内比较动作 Q；类别 Q 与动作 Q 均不是胜率。</p></div>`;
}

export function searchHTML(data) {
  if (!data) return "";
  return `<p class="small">P${data.perspective} Value = ${fmt(data.root_value)} · ${data.stats.simulations} simulations · ${data.stats.model_evaluations} model evaluations · ${(1000*data.seconds).toFixed(1)} ms</p>
    <p class="small muted">Q 为当前执手视角的搜索均值，未访问动作记 0。Value / Q 范围 [−1, 1]；访问比例用于训练策略。搜索路径重复局面按 0 回传。</p>
    <div class="q-table"><div class="q-row mcts-row q-head"><span>动作</span><span>Prior</span><span>N</span><span>N / ΣN</span><span>Q</span><span>访问比例</span></div>${data.actions.map(row=>
      `<button class="q-row mcts-row ${row.action_index === data.selected_action_index ? "chosen" : ""}" data-output-id="${esc(row.id)}"><span>${esc(row.label)}</span><span>${fmt(row.prior)}</span><span>${row.visits}</span><span>${fmt(row.visit_probability)}</span><span>${fmt(row.q)}</span><span class="visit-track"><i style="width:${100*row.visit_probability}%"></i></span></button>`).join("")}</div>`;
}

// Board tensors use [x,y]; dense vectors and kernels retain their tensor indexing.
export function tensorSVG(values, layout, offset=0) {
  const xy = layout === "board_xy", rows = xy ? values[0].length : values.length;
  const cols = xy ? values.length : Math.max(...values.map(row => row.length));
  const flat = values.flat(), magnitude = Math.max(...flat.map(Math.abs)), limit = magnitude || 1;
  const cell = 32, pad = 30;
  let out = `<svg viewBox="0 0 ${cols*cell+pad*2} ${rows*cell+pad*2}" role="img" aria-label="${xy ? "空间张量，a1 左下" : "张量切片，按索引排列"}">`;
  for (let r=0;r<rows;r++) for (let c=0;c<cols;c++) {
    const value = xy ? values[c][rows-1-r] : values[r][c];
    if (value === undefined) continue;
    const coordinate = xy ? `[${c},${rows-1-r}]` : layout === "vector" ? `index ${offset+r*16+c}` : `[${r},${c}]`;
    const alpha = Math.abs(value)/limit;
    const base = value >= 0 ? [35,108,86] : [181,89,54];
    const color = base.map(v=>Math.round(248+(v-248)*alpha));
    out += `<rect x="${pad+c*cell}" y="${pad+r*cell}" width="${cell-1}" height="${cell-1}" fill="rgb(${color.join(",")})"><title>${coordinate}: ${value}</title></rect>`;
    out += `<text x="${pad+c*cell+cell/2}" y="${pad+r*cell+cell/2+3}" text-anchor="middle" fill="${alpha > .65 ? "white" : "#34473e"}" font-family="ui-monospace,monospace" font-size="${String(fmt(value)).length > 6 ? 6 : 8}">${fmt(value)}</text>`;
  }
  for(let c=0;c<cols;c++) out += `<text x="${pad+c*cell+cell/2}" y="${rows*cell+pad+17}" text-anchor="middle" font-size="10" fill="#65736e">${xy ? String.fromCharCode(97+c) : c}</text>`;
  for(let r=0;r<rows;r++) out += `<text x="18" y="${pad+r*cell+cell/2+3}" text-anchor="middle" font-size="10" fill="#65736e">${xy ? rows-r : r}</text>`;
  return out+`</svg><div class="tensor-scale"><span>${magnitude ? `−${fmt(magnitude)}` : "0"}</span><i></i><span>${magnitude ? `+${fmt(magnitude)}` : "0 · 全零"}</span></div>`;
}

function histogramSVG(histogram) {
  if (!histogram) return "";
  const max = Math.max(...histogram.counts) || 1;
  return `<svg viewBox="0 0 420 100" class="tensor-histogram" role="img" aria-label="数值分布直方图">${histogram.counts.map((n,i) => `<rect x="${i*21}" y="${80-n/max*75}" width="19" height="${n/max*75}" fill="#699b83"><title>[${histogram.edges[i]}, ${histogram.edges[i+1]}]: ${n}</title></rect>`).join("")}<text x="0" y="97" font-size="10">${fmt(histogram.edges[0])}</text><text x="420" y="97" font-size="10" text-anchor="end">${fmt(histogram.edges.at(-1))}</text></svg>`;
}

function statistics(data, label="统计") {
  return `<h3>${label}</h3><dl class="tensor-stats">${[["min", data.min], ["max", data.max], ["mean", data.mean], ["std", data.std], ["零值占比", `${(data.zero_fraction*100).toFixed(1)}%`]].map(([key,value]) => `<dt>${key}</dt><dd>${typeof value === "number" ? fmt(value) : value}</dd>`).join("")}</dl>${histogramSVG(data.histogram)}`;
}

export function createNeuralPanel({api, onCandidate}) {
  const root = document.getElementById("neural-panel");
  const s = {analysis: null, branch: null, mode: "input", view: 0, objective: "margin", layer: "", channel: 0,
    inputChannel: 0, offset: 0, tensor: null, search: null, simulations: 50, cPuct: 1.5, revision: 0, pending: false, error: ""};

  function render() {
    const n = s.analysis?.neural;
    root.hidden = !n;
    if (!n) return;
    const score = outputLabel(n.model);
    let controls = "", content = "";
    if (s.mode === "mcts") {
      controls = `<form id="search-form"><label for="search-simulations">Simulations</label><input id="search-simulations" type="number" min="1" max="200" step="1" value="${s.simulations}" required>
        <label for="search-cpuct">c_puct</label><input id="search-cpuct" type="number" min="0.01" max="10" step="0.01" value="${s.cPuct}" required>
        <button class="button secondary wide" ${s.pending ? "disabled" : ""}>${s.pending ? "搜索中…" : "运行 PUCT"}</button></form>
        <p class="small">Value = ${fmt(n.value)} · P${n.input.perspective}</p>
        <p class="small mono">${esc(n.model.metadata.eval_summary?.value_head_status || "")}</p>
        <p class="small muted">无根噪声；每次重建搜索树。主棋盘保持无搜索策略的分析结果，表中高亮为 MCTS 选择。点击动作检查其策略梯度。</p>`;
      content = s.pending ? `<p class="muted">搜索中…</p>` : s.error ? `<p class="error-text">${esc(s.error)}</p>` : s.search ? searchHTML(s.search) : `<p class="muted">运行搜索后显示先验、访问分布和动作价值。</p>`;
    } else if (s.mode === "input" || s.mode === "sensitivity") {
      const sensitivity = s.branch?.neural?.sensitivities.find(v=>v.objective === s.objective) || s.branch?.neural?.sensitivities[0];
      const views = s.mode === "input" ? n.input.views : sensitivity?.views;
      if (views) {
        s.view = Math.min(s.view, views.length-1);
        controls += `<label for="neural-view">输入分量</label><select id="neural-view">${options(views.map((v,i)=>[i,`${i} · ${v.name}`]),s.view)}</select>`;
        if (s.mode === "sensitivity") {
          controls += `<label for="neural-objective">梯度目标</label><select id="neural-objective">${options(s.branch.neural.sensitivities.map(v=>[v.objective,({q:`动作 ${score}(a)`,margin:`动作 ${score}(a) − ${score}(b)`,type_q:"类别 Q(t)",type_margin:"类别 Q(t) − Q(另一类)"})[v.objective]]), sensitivity.objective)}</select>
            <p class="small">a = ${esc(s.analysis.candidates.find(c=>c.id === s.branch.action_id)?.label || "")} · ${score} = ${fmt(s.branch.value)}</p>
            ${sensitivity.reference_label ? `<p class="small">b = ${esc(sensitivity.reference_label)}（${n.hierarchy ? "同类别内" : "其余"}合法动作最高 ${score}）</p>` : ""}
            <p class="small">目标值 = ${fmt(sensitivity.value)}</p>
            <p class="small muted">∂目标 / ∂输入。局部梯度，不是因果贡献；离散特征和广播平面不能逐格独立改变。</p>`;
        } else {
          controls += `<dl class="config"><dt>Shape</dt><dd>${n.input.shape.join(" × ")}</dd><dt>视角</dt><dd>P${n.input.perspective}</dd></dl>
            <p class="small mono">${esc(n.model.metadata.observation_version)}</p><p class="small muted">原始 [x,y] 坐标，不翻转棋盘。墙平面表示锚点。</p>
            ${n.model.architecture === "mlp" ? `<p class="small muted">此 MLP 编码没有目标方向输入。</p>` : ""}`;
        }
        const view = views[s.view];
        content = `<div class="tensor-chart">${tensorSVG(view.values,view.layout)}</div>${s.mode === "sensitivity" && view.values.flat().every(v=>v === 0) ? `<p class="small">当前目标对该输入分量的梯度全为 0。</p>` : ""}<p class="small muted">${s.mode === "sensitivity" ? "正值 / 负值表示增大该输入时目标值的局部变化方向。" : "显示实际传入网络的数值。"} 色阶按当前切片的最大绝对值缩放，悬停查看完整数值。</p>`;
      } else content = `<p class="muted">${s.analysis.winner ? "终局无合法动作，无动作梯度。" : "选择合法候选动作后计算输入梯度。"}</p>`;
    } else if (s.mode === "outputs") {
      controls = `${hierarchyHTML(n.hierarchy)}<p class="small">209 个动作输出</p><p class="small muted">按原始 ${score} 降序排列。${n.hierarchy ? "未选类别和" : ""}非法动作在选步前置为 −∞。${score === "logit" ? "logit 为策略分数，不是预期回报或胜率。" : "Q 不归一化为概率。"}</p>
        <dl class="config"><dt>合法动作</dt><dd>${s.analysis.candidates.length}</dd><dt>未屏蔽 argmax</dt><dd>${esc(n.raw_best.label)}</dd><dt>argmax 合法</dt><dd>${n.raw_best.legal ? "是" : "否"}</dd></dl>`;
      content = `<div class="q-table"><div class="q-row q-head"><span>ID</span><span>动作</span><span>原始 ${score}</span><span>合法</span><span>选步后 ${score}</span></div>${[...n.outputs].sort((a,b)=>b.value-a.value||a.action_index-b.action_index).map(row => `<button class="q-row ${row.id === s.analysis.selected_id ? "chosen" : ""}" ${row.legal ? `data-output-id="${row.id}"` : "disabled"}><span>${row.action_index}</span><span>${esc(row.label)}</span><span title="${row.value}">${fmt(row.value)}</span><span>${row.legal ? "是" : "否"}</span><span>${row.legal && row.route_eligible !== false ? fmt(row.value) : "−∞"}</span></button>`).join("")}</div>`;
    } else {
      const layers = n.layers.filter(l=>s.mode !== "weight" || l.weight);
      const layer = layers.find(l=>l.name === s.layer) || layers[0];
      s.layer = layer.name;
      const shape = s.mode === "weight" ? layer.weight.shape : layer.shape;
      const hasChannel = s.mode === "weight" || shape.length === 3;
      const hasInput = s.mode === "weight" && shape.length === 4;
      const vector = s.mode === "weight" ? shape.length === 2 : shape.length === 1;
      controls = `<form id="tensor-form"><label for="neural-layer">层</label><select id="neural-layer">${options(layers.map(l=>[l.name,`${l.name} · ${l.type} · ${l.shape.join("×")}`]),s.layer)}</select>
        ${hasChannel ? `<label for="tensor-channel">${s.mode === "weight" ? "输出单元 / 通道" : "通道"}</label><input id="tensor-channel" type="number" value="${s.channel}" min="0" max="${shape[0]-1}" step="1" required>` : ""}
        ${hasInput ? `<label for="tensor-input">输入通道</label><input id="tensor-input" type="number" value="${s.inputChannel}" min="0" max="${shape[1]-1}" step="1" required>` : ""}
        ${vector ? `<label for="tensor-offset">向量起始索引（最多 256 项）</label><input id="tensor-offset" type="number" value="${s.offset}" min="0" max="${(s.mode === "weight" ? shape[1] : shape[0])-1}" step="1" required>` : ""}
        <button class="button secondary wide" ${s.pending ? "disabled" : ""}>${s.pending ? "读取中…" : "读取切片"}</button></form>
        <p class="small mono">${s.mode === "weight" ? `Weight ${shape.join(" × ")}` : `Output ${shape.join(" × ")}`}</p>
        ${statistics(s.mode === "weight" ? layer.weight.stats : layer.activation, s.mode === "weight" ? "全层权重" : "本局面全层激活")}
        <p class="small muted">${s.mode === "weight" ? "卷积核按张量行列排列；全连接层每次显示一个输出单元的部分输入权重。" : "激活为实际前向输出。零值占比只描述当前局面，不代表神经元永久失活。"}</p>`;
      content = s.pending ? `<p class="muted">读取中…</p>` : s.error ? `<p class="error-text">${esc(s.error)}</p>` : s.tensor
        ? `<div class="tensor-chart">${tensorSVG(s.tensor.values,s.tensor.layout,s.tensor.window.offset)}</div>
          <p class="small muted">${s.tensor.layout === "vector" ? `索引 ${s.tensor.window.offset}–${s.tensor.window.offset+s.tensor.window.shown-1} / ${s.tensor.window.total} 项；按 16 列排列，无棋盘含义。` : s.tensor.layout === "board_xy" ? "[x,y] 空间坐标，a1 左下。" : "核索引 [行,列]，不对应棋盘格。"} 当前切片独立色阶。</p>${statistics(s.tensor.stats, "当前通道 / 输出单元统计（含未展示项）")}`
        : `<p class="muted">选择层和切片参数后读取。</p>`;
      if (s.mode === "activation") content += `<div class="layer-table"><div><b>层</b><b>输出 shape</b><b>零值占比</b></div>${n.layers.map(l=>`<div><span>${esc(l.name)} · ${l.type}</span><span>${l.shape.join("×")}</span><span>${(l.activation.zero_fraction*100).toFixed(1)}%</span></div>`).join("")}</div>`;
    }
    root.innerHTML = `<div class="neural-heading"><h2>网络诊断</h2><span class="small mono">${esc(n.model.id)} · ${n.model.sha256.slice(0,12)} · 根局面 P${n.input.perspective}${n.value != null ? ` · Value ${fmt(n.value)}` : ""}</span></div>
      <div class="segmented neural-tabs">${Object.entries(modes).filter(([id])=>id !== "mcts" || n.value != null).map(([id,label])=>`<button data-nmode="${id}" class="${s.mode === id ? "active" : ""}" aria-pressed="${s.mode === id}">${label}</button>`).join("")}</div>
      <div class="neural-body"><div class="neural-controls">${controls}</div><div class="neural-content">${content}</div></div>`;
  }

  async function fetchTensor() {
    const revision = ++s.revision;
    s.pending = true; s.error = ""; s.tensor = null; render();
    try {
      const result = await api("/api/neural/tensor", {position: s.analysis.position, model_id: s.analysis.model_id,
        kind: s.mode, layer: s.layer, channel: s.channel, input_channel: s.inputChannel, offset: s.offset});
      if (revision !== s.revision) return;
      s.tensor = result;
    } catch (error) { if (revision === s.revision) s.error = error.message; }
    finally { if (revision === s.revision) {s.pending = false; render();} }
  }

  async function fetchSearch() {
    const revision = ++s.revision;
    s.pending = true; s.error = ""; s.search = null; render();
    try {
      const result = await api("/api/neural/search", {position: s.analysis.position, model_id: s.analysis.model_id,
        simulations: s.simulations, c_puct: s.cPuct});
      if (revision === s.revision) s.search = result;
    } catch (error) { if (revision === s.revision) s.error = error.message; }
    finally { if (revision === s.revision) { s.pending = false; render(); } }
  }

  root.onclick = event => {
    const tab = event.target.closest("[data-nmode]"), output = event.target.closest("[data-output-id]");
    if (output) { void onCandidate(output.dataset.outputId); return; }
    if (!tab) return;
    s.mode = tab.dataset.nmode; s.revision++; s.pending = false; s.error = ""; s.tensor = null;
    s.layer = ""; s.channel = s.inputChannel = s.offset = 0; render();
    if (["activation", "weight"].includes(s.mode)) void fetchTensor();
  };
  root.onchange = event => {
    if (event.target.id === "neural-view") s.view = +event.target.value;
    else if (event.target.id === "neural-objective") s.objective = event.target.value;
    else if (event.target.id === "neural-layer") {
      s.layer = event.target.value; s.channel = s.inputChannel = s.offset = 0; void fetchTensor(); return;
    } else return;
    render();
  };
  root.onsubmit = event => {
    event.preventDefault();
    if (event.target.id === "search-form") {
      s.simulations = Number(root.querySelector("#search-simulations").value);
      s.cPuct = Number(root.querySelector("#search-cpuct").value);
      void fetchSearch(); return;
    }
    s.channel = Number(root.querySelector("#tensor-channel")?.value || 0);
    s.inputChannel = Number(root.querySelector("#tensor-input")?.value || 0);
    s.offset = Number(root.querySelector("#tensor-offset")?.value || 0);
    void fetchTensor();
  };
  return {
    setAnalysis(value) {
      if (s.analysis === value) return;
      s.analysis = value; s.branch = null; s.revision++; s.pending = false; s.tensor = null; s.search = null;
      s.error = ""; s.mode = "input"; s.view = 0; render();
    },
    setBranch(value) { if (s.branch === value) return; s.branch = value; if (s.mode === "sensitivity") render(); },
    currentTensor() { return s.tensor; },
    currentSearch() { return s.search; },
  };
}
