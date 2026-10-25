import React, { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { safeReturnRoute, type AnalysisContext } from '../platform'
import { ResearchContext } from '../components/ResearchContext'
import { EvalReplayBoard } from '../components/EvalReplayBoard'
import { labRequest, opening, positionBoard, type Position } from '../lab'
import './LabView.css'

type Data = Record<string, any>
const fmt = (value: any) => typeof value === 'number' ? Number(value.toPrecision(5)).toString() : String(value ?? 'unknown')
function Json({value}: {value: unknown}) { return <pre className="lab-json">{JSON.stringify(value, null, 2)}</pre> }
export function Tensor({data}: {data: Data}) {
  const values: number[][] = data.values ?? []
  if (!values.length) return null
  const xy = data.layout === 'board_xy'
  const rows = xy ? values[0].length : values.length
  const cols = xy ? values.length : Math.max(...values.map(r => r.length))
  const limit = Math.max(1e-12, ...values.flat().map(Math.abs))
  return <figure className="lab-tensor"><svg viewBox={`0 0 ${cols*32} ${rows*32}`} role="img" aria-label={xy ? 'Spatial tensor, a1 bottom left' : 'Tensor slice by index'}>
    {Array.from({length: rows}, (_,r) => Array.from({length: cols}, (_,c) => {
      const value = xy ? values[c]?.[rows-1-r] : values[r]?.[c]
      if (value === undefined) return null
      const alpha = Math.abs(value)/limit
      return <g key={`${r}:${c}`}><rect x={c*32} y={r*32} width="31" height="31" fill={value >= 0 ? `rgba(35,108,86,${.1+.9*alpha})` : `rgba(181,89,54,${.1+.9*alpha})`}><title>{xy ? `[${c},${rows-1-r}]` : `[${r},${c}]`}: {value}</title></rect><text x={c*32+16} y={r*32+18} textAnchor="middle" fill={alpha>.65?'white':'currentColor'} fontSize="7">{fmt(value)}</text></g>
    }))}
  </svg><figcaption>{data.name ?? data.layer} · {data.layout} · symmetric range ±{fmt(limit)} (this slice only)</figcaption></figure>
}
export function LabView() {
  const [params] = useSearchParams(); const query = params.toString()
  const [context,setContext] = useState<AnalysisContext|null>(null)
  const [position,setPosition] = useState<Position>(opening)
  const [root,setRoot] = useState<Position>(opening)
  const [draft,setDraft] = useState(JSON.stringify(opening,null,2))
  const [models,setModels] = useState<Data[]>([]); const [presets,setPresets] = useState<Data[]>([])
  const [model,setModel] = useState(''); const [algorithm,setAlgorithm] = useState('greedy'); const [depth,setDepth] = useState(2)
  const [result,setResult] = useState<Data|null>(null); const [branch,setBranch] = useState<Data|null>(null)
  const [tensor,setTensor] = useState<Data|null>(null); const [search,setSearch] = useState<Data|null>(null)
  const [comparison,setComparison] = useState<Data|null>(null); const [multi,setMulti] = useState<string[]>([])
  const [tab,setTab] = useState('outputs'); const [layer,setLayer] = useState(''); const [channel,setChannel] = useState(0)
  const [inputChannel,setInputChannel] = useState(0); const [offset,setOffset] = useState(0); const [simulations,setSimulations] = useState(50)
  const [error,setError] = useState(''); const [busy,setBusy] = useState(false); const [ready,setReady] = useState(false)
  const [editTool,setEditTool] = useState('none');
  const [filter,setFilter] = useState('all'); const [undo,setUndo] = useState<Position[]>([])
  const generation = useRef(0); const pending = useRef<AbortController|null>(null)
  function invalidate() { generation.current++; pending.current?.abort(); setBusy(false); setResult(null);setBranch(null);setTensor(null);setSearch(null);setComparison(null);setError('') }
  function changePosition(next: Position, remember=true) {
    invalidate(); if (remember) setUndo(old=>[...old,position]); setPosition(next);setDraft(JSON.stringify(next,null,2))
  }
  useEffect(() => {
    const controller=new AbortController(); const version=++generation.current
    pending.current?.abort();setBusy(false);setReady(false);setError('');setResult(null);setBranch(null);setTensor(null);setSearch(null);setComparison(null)
    Promise.all([labRequest('meta',undefined,controller.signal),labRequest('models',undefined,controller.signal),
      params.get('snapshot') ? Promise.resolve(JSON.parse(sessionStorage.getItem('quoridor-lab:'+params.get('snapshot')) ?? 'null')) : api.context(new URLSearchParams(query))
    ]).then(async ([meta,catalog,ctx])=>{
      if (!ctx?.position) throw new Error('Analysis context is missing')
      if (!safeReturnRoute(ctx.return_route) && !(ctx.source?.type==='live_snapshot' && /^\/play\?game=[a-f0-9-]{36}$/.test(ctx.return_route))) throw new Error('Unsafe context return route')
      const checked=await labRequest('position',ctx.position,controller.signal)
      if (version!==generation.current) return
      setPresets(meta.presets);setModels(catalog.models);setContext(ctx);setRoot(checked.position);setPosition(checked.position);setDraft(JSON.stringify(checked.position,null,2));setUndo([])
      const sha=ctx.analysis_model?.sha256; const selected=catalog.models.find((m:Data)=>m.sha256===sha)
      setModel(selected?.id ?? '');setAlgorithm(sha?'dqn':'greedy');setReady(true)
      if (sha && !selected) setError('Exact selected SHA is not available in the Lab catalog; no substitute model was selected.')
    }).catch(e=>{if (version===generation.current && e.name!=='AbortError') setError(String(e))})
    return ()=>{controller.abort();generation.current++;pending.current?.abort()}
  },[query])
  async function run(operation: (signal: AbortSignal)=>Promise<any>, apply: (value:any)=>void) {
    pending.current?.abort();const controller=new AbortController();pending.current=controller
    const version=++generation.current;setBusy(true);setError('')
    try { const value=await operation(controller.signal);if(version===generation.current) apply(value) }
    catch(e:any){if(version===generation.current && e.name!=='AbortError')setError(String(e))}
    finally {if(version===generation.current)setBusy(false)}
  }
  const payload = {position,algorithm,depth,...(algorithm==='dqn'?{model_id:model}:{})}
  const selected=models.find(m=>m.id===model)
  const layers:Data[]=result?.neural?.layers ?? []
  const canSearch=['trained','calibrated'].includes(result?.neural?.model?.metadata?.eval_summary?.value_head_status)
  const candidates:Data[]=result?.candidates ?? []
  const shown=candidates.filter(r=>filter==='all'||r.action?.type===filter)
  const returnRoute=safeReturnRoute(context?.return_route) ?? (context?.source?.type==='live_snapshot' && typeof context.return_route==='string' && /^\/play\?game=[a-f0-9-]{36}$/.test(context.return_route)?context.return_route:null)
  function analyze(){void run(signal=>labRequest('analyze',payload,signal),value=>{setResult(value);setLayer(value.neural?.layers?.[0]?.name ?? '');setBranch(null);setTensor(null);setSearch(null)})}
  function inspect(action:any){void run(signal=>labRequest('branch',{...payload,action},signal),setBranch)}
  function importPosition(){void run(signal=>{const parsed=JSON.parse(draft);return labRequest('position',parsed.position??parsed,signal)},data=>changePosition(data.position))}
  function editCell(x:number,y:number) {
    if(busy||editTool==='none')return
    const next:Position=JSON.parse(JSON.stringify(position))
    if(editTool==='p1'||editTool==='p2')next.pawns[editTool==='p1'?0:1]=[x,y]
    else {if(x>7||y>7)return;next.walls.push({x,y,orientation:editTool as 'horizontal'|'vertical'});next.walls_remaining[next.turn-1]--}
    void run(signal=>labRequest('position',next,signal),data=>changePosition(data.position))
  }
  function removeWall(x:number,y:number,orientation:'horizontal'|'vertical') {
    if(busy||editTool==='none')return
    const next:Position=JSON.parse(JSON.stringify(position))
    next.walls=next.walls.filter(w=>!(w.x===x&&w.y===y&&w.orientation===orientation));next.walls_remaining[next.turn-1]++
    void run(signal=>labRequest('position',next,signal),data=>changePosition(data.position))
  }
  function download(){const blob=new Blob([JSON.stringify({context,position,root,result,branch,tensor,search,comparison},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='quoridor-analysis.json';a.click();URL.revokeObjectURL(url)}
  return <div className="native-lab">
    <div className="page-heading"><div><p className="eyebrow">Research / Decision Lab</p><h1>Decision Lab</h1><p className="muted">Analyze a private position copy. Live games and recorded evidence remain unchanged.</p></div><button onClick={download}>Export analysis JSON</button></div>
    {context && <ResearchContext context={{...context,analysis_model:selected?{sha256:selected.sha256}:null}}/>}
    {returnRoute && <Link to={returnRoute}>Return to source{context?.source?.ply!==undefined?` · ply ${context.source.ply}`:''}</Link>}
    {context?.source?.type==='live_snapshot' && <p className="warning">Frozen snapshot only. The live game is not paused.</p>}
    {context?.warnings?.map((w,i)=><p className="warning" key={i}>{w}</p>)}
    {error && <div className="error" role="alert">{error}</div>}
    {!ready && !error && <p role="status">Resolving validated context…</p>}
    <div className="lab-workspace">
      <section className="panel lab-controls"><h2>Agent and position</h2>
        <label>Algorithm<select aria-label="Analysis algorithm" value={algorithm} onChange={e=>{invalidate();setAlgorithm(e.target.value)}}><option value="greedy">Greedy</option><option value="minimax">Minimax</option><option value="dqn">Neural model</option></select></label>
        {algorithm==='minimax' && <label>Depth<select value={depth} onChange={e=>{invalidate();setDepth(+e.target.value)}}>{[1,2,3].map(d=><option key={d}>{d}</option>)}</select></label>}
        {algorithm==='dqn' && <><label>Analysis checkpoint<select aria-label="Analysis checkpoint" value={model} onChange={e=>{invalidate();setModel(e.target.value)}}><option value="">Select exact checkpoint</option>{models.map(m=><option key={m.id} value={m.id} disabled={!m.available}>{m.aliases?.join(' / ') || m.id} · {m.sha256.slice(0,12)}{!m.available?' · unavailable':''}</option>)}</select></label>{selected && <p className="mono sha">{selected.sha256}</p>}</>}
        <button disabled={!ready||busy||(algorithm==='dqn'&&!model)} onClick={analyze}>{busy?'Working…':'Run analysis'}</button>
        <button disabled={!ready||busy} onClick={()=>void run(async signal=>{const rows=[];for(const [a,d] of [['greedy',1],['minimax',1],['minimax',2],['minimax',3]])rows.push(await labRequest('analyze',{position,algorithm:a,depth:d},signal));return {rows}},setComparison)}>Compare Greedy / D1–D3</button>
        <label>Preset<select aria-label="Position preset" defaultValue="" onChange={e=>{const preset=presets.find(p=>p.id===e.target.value);if(preset)changePosition(preset.position)}}><option value="">Custom position</option>{presets.map(p=><option key={p.id} value={p.id}>{p.title}</option>)}</select></label>
        <label>Board editor<select aria-label="Board editor" value={editTool} onChange={e=>setEditTool(e.target.value)}><option value="none">Read only</option><option value="p1">Place P1</option><option value="p2">Place P2</option><option value="horizontal">Place horizontal wall</option><option value="vertical">Place vertical wall</option></select></label>{editTool!=='none' && <p className="muted">Click a cell to edit; wall anchors use its lower-left coordinate. Click an existing wall to remove it. Wall stock is charged/refunded to the current player.</p>}<label>Side to move<select aria-label="Side to move" value={position.turn} onChange={e=>changePosition({...position,turn:+e.target.value as 1|2})}><option value="1">P1</option><option value="2">P2</option></select></label>
        <details><summary>Edit or import position</summary><p>Coordinates use x,y with a1 at the bottom left. The engine validates walls, stocks and paths.</p><textarea aria-label="Position JSON" rows={12} value={draft} onChange={e=>setDraft(e.target.value)}/><button disabled={busy} onClick={importPosition}>Validate and import</button></details>
        <div className="action-bar"><button disabled={!undo.length||busy} onClick={()=>{const prev=undo[undo.length-1];setUndo(undo.slice(0,-1));changePosition(prev,false)}}>Undo</button><button disabled={busy} onClick={()=>changePosition(root)}>Return to root</button><button disabled={busy} onClick={()=>changePosition(opening)}>Standard opening</button></div>
      </section>
      <section className="panel lab-board"><h2>{JSON.stringify(position)===JSON.stringify(root)?'Root position':'Derived position'} · P{position.turn} to move</h2><EvalReplayBoard state={positionBoard(position)} size={450} onCellClick={editTool!=='none'?editCell:undefined} onWallClick={editTool!=='none'?removeWall:undefined}/><p>P1 walls: {position.walls_remaining[0]} · P2 walls: {position.walls_remaining[1]}</p>{result && <p>Selected: {result.selected_id ?? 'none'} · {result.neural?.model?.output_semantics ?? 'heuristic score'} · perspective P{position.turn}</p>}{Boolean(context?.actual_action) && <details><summary>Recorded action (separate from recommendation)</summary><Json value={context?.actual_action}/></details>}</section>
      <section className="panel lab-candidates"><h2>Candidate actions ({shown.length})</h2><label>Action type<select value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All</option><option value="pawn">Pawn</option><option value="horizontal">Horizontal wall</option><option value="vertical">Vertical wall</option></select></label><p className="muted">Q values, logits and heuristic scores have different meanings and scales.</p><div className="lab-scroll"><table><thead><tr><th>Action</th><th>Value</th><th>Bound</th><th>Inspect</th></tr></thead><tbody>{shown.map(r=><tr key={r.id} className={r.id===result?.selected_id?'lab-selected':''}><td>{r.label ?? r.id}</td><td>{fmt(r.value)}</td><td>{r.bound}</td><td><button disabled={busy} onClick={()=>inspect(r.action)}>Inspect {r.label ?? r.id}</button></td></tr>)}</tbody></table></div></section>
    </div>
    {branch && <section className="panel"><h2>Branch: {branch.action_id}</h2><p>Value {fmt(branch.value)} · {branch.bound} · {branch.complete?'complete':'partial'}</p><button disabled={busy} onClick={()=>{if(branch.after?.position)changePosition(branch.after.position)}}>Explore resulting position</button><details><summary>Principal variation and search trace</summary><Json value={{principal_variation:branch.principal_variation,trace:branch.trace,stats:branch.stats}}/></details></section>}
    {comparison && <section className="panel"><h2>Agent comparison</h2><p>Compare decisions; values across algorithms are not on a common scale.</p><Json value={comparison}/></section>}
    {result?.neural && <section className="panel lab-network"><h2>Network diagnostics</h2><p>{result.neural.model.architecture} · {result.neural.model.output_semantics} · {result.neural.model.parameters} parameters</p><div className="action-bar" role="group" aria-label="Network diagnostics">{['outputs','inputs','layers','weights','sensitivity','search','compare'].map(t=><button key={t} aria-pressed={tab===t} onClick={()=>{setTab(t);setTensor(null)}}>{t}</button>)}</div>
      {tab==='outputs' && <><p>Illegal actions are shown for diagnosis, never offered as moves.</p><div className="lab-scroll"><table><thead><tr><th>Action</th><th>Output</th><th>Legal</th><th>Route eligible</th></tr></thead><tbody>{result.neural.outputs.map((r:Data)=><tr key={r.id}><td>{r.label}</td><td>{fmt(r.value)}</td><td>{String(r.legal)}</td><td>{String(r.route_eligible??r.legal)}</td></tr>)}</tbody></table></div>{result.neural.hierarchy && <Json value={result.neural.hierarchy}/>}</>}
      {tab==='inputs' && <div className="lab-tensors">{result.neural.input.views.map((v:Data)=><Tensor key={v.name} data={v}/>)}</div>}
      {(tab==='layers'||tab==='weights') && <><div className="lab-tensor-controls"><label>Layer<select aria-label="Network layer" value={layer} onChange={e=>{setLayer(e.target.value);setTensor(null);setChannel(0);setInputChannel(0);setOffset(0)}}>{layers.map(l=><option key={l.name} value={l.name}>{l.name} · {l.type} [{l.shape.join(',')}]</option>)}</select></label><label>Output channel<input aria-label="Output channel" type="number" min="0" value={channel} onChange={e=>setChannel(+e.target.value)}/></label>{tab==='weights' && <label>Input channel<input type="number" min="0" value={inputChannel} onChange={e=>setInputChannel(+e.target.value)}/></label>}<label>Vector offset<input type="number" min="0" value={offset} onChange={e=>setOffset(+e.target.value)}/></label><button disabled={busy||!layer||(tab==='weights'&&!layers.find(l=>l.name===layer)?.weight)} onClick={()=>void run(signal=>labRequest('neural/tensor',{position,model_id:model,kind:tab==='layers'?'activation':'weight',layer,channel,input_channel:tab==='weights'?inputChannel:0,offset},signal),setTensor)}>Load tensor slice</button></div><details><summary>Layer statistics</summary><Json value={layers.find(l=>l.name===layer)}/></details>{tensor && <><Tensor data={tensor}/><Json value={{stats:tensor.stats,window:tensor.window}}/></>}</>}
      {tab==='sensitivity' && <><p>Inspect a candidate action first. Input gradients measure local sensitivity, not causal contribution or a legal counterfactual.</p>{branch?.neural?.sensitivities?.map((s:Data)=><section key={s.objective}><h3>{s.objective} · {fmt(s.value)}</h3><div className="lab-tensors">{s.views.map((v:Data)=><Tensor key={v.name} data={v}/>)}</div></section>)}</>}
      {tab==='search' && <><p>PUCT requires a trained or calibrated value head. Search Q is distinct from DQN Q.</p><label>Simulations<input type="number" min="1" max="200" value={simulations} onChange={e=>setSimulations(+e.target.value)}/></label><button disabled={busy||!canSearch} onClick={()=>void run(signal=>labRequest('neural/search',{position,model_id:model,simulations,c_puct:1.5},signal),setSearch)}>Run PUCT</button>{!canSearch && <p className="warning">This model has no verified trained/calibrated value head.</p>}{search && <Json value={search}/>}</>}
      {tab==='compare' && <><label>Compare models (up to 8)<select multiple aria-label="Compare models" value={multi} onChange={e=>setMulti(Array.from(e.target.selectedOptions,o=>o.value).slice(0,8))}>{models.filter(m=>m.available).map(m=><option key={m.id} value={m.id}>{m.aliases?.join(' / ')||m.id}</option>)}</select></label><button disabled={busy||!multi.length} onClick={()=>void run(signal=>labRequest('multi-compare',{position,model_ids:multi,actual_action:context?.actual_action??null},signal),setComparison)}>Compare selected models</button></>}
    </section>}
    <details className="panel"><summary>Analysis provenance and exact response</summary><Json value={{context,result}}/></details>
  </div>
}
