import React, {useEffect, useRef, useState} from 'react'
import {Link, Navigate, Route, Routes, useParams, useSearchParams} from 'react-router-dom'
import {api} from '../api'
import {labRequest, positionBoard, type Position} from '../lab'
import {EvalReplayBoard} from '../components/EvalReplayBoard'
import {Tensor} from './LabView'
import {MetricsView} from './MetricsView'
import type {RunDetail, RunSummary} from '../types'
import '../App.css'
import './AgentWorkspace.css'
import './Refinement.css'

type Obj = Record<string, any>
export async function evidence(path:string, options:RequestInit={}) {
  const response=await fetch('/api/agents'+path,options)
  const data=await response.json()
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail??data))
  return data
}
function useData(path:string|null){
 const [data,setData]=useState<Obj|null>(null),[error,setError]=useState(''),[retry,setRetry]=useState(0)
 useEffect(()=>{setData(null);setError('');if(!path)return;const c=new AbortController();evidence(path,{signal:c.signal}).then(d=>{if(!c.signal.aborted)setData(d)}).catch(e=>{if(!c.signal.aborted)setError(String(e))});return()=>c.abort()},[path,retry])
 return {data,error,retry:()=>setRetry(v=>v+1)}
}
const enc=encodeURIComponent
const kind=(a:Obj)=>({neural:'神经网络',heuristic:'启发式',random:'随机',human:'玩家'}[a.kind as string]??a.kind)
const fmt=(x:any,n=2)=>typeof x==='number'?x.toFixed(n):'—'
const pct=(x:any)=>typeof x==='number'?(100*x).toFixed(1)+'%':'—'
const short=(s?:string)=>s?s.slice(0,12):'—'
export function route(path:string,season:string,values:Record<string,string|number|undefined>={}){const q=new URLSearchParams({season});Object.entries(values).forEach(([k,v])=>{if(v!==undefined)q.set(k,String(v))});return path+'?'+q}
function ErrorBox({text,retry}:{text:string;retry?:()=>void}){return text?<div className="error" role="alert">{text}{retry&&<button onClick={retry}>重试</button>}</div>:null}
function Json({value}:{value:unknown}){return <pre className="aw-json">{JSON.stringify(value,null,2)}</pre>}
function Tag({agent}:{agent:Obj}){return <span className={'aw-tag '+agent.kind}>{kind(agent)}</span>}
function agentVersion(a:Obj){const sha=a.actual_sha256??a.sha256;return sha?'checkpoint '+short(sha):a.spec?.algo?.type==='minimax'?'depth '+a.spec.algo.params?.depth:a.spec?.algo?.type??'版本未记录'}

export default function AgentWorkspace(){
 const [params]=useSearchParams();const seasons=useData('/seasons');const season=params.get('season')??seasons.data?.seasons?.[0]?.id??''
 const roster=useData(season?'/seasons/'+enc(season):null)
 const [theme,setTheme]=useState(()=>localStorage.getItem('quoridor-theme')??'light')
 useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem('quoridor-theme',theme)},[theme])
 return <div className="app agent-workspace"><header className="aw-header"><Link to={route('/ladder',season)} className="aw-brand">Quoridor</Link><nav aria-label="主要入口"><Link to={route('/ladder',season)}>天梯</Link><Link to={route('/play',season)}>对战</Link></nav><span className="aw-preview">独立预览</span><button onClick={()=>setTheme(t=>t==='light'?'dark':'light')}>{theme==='light'?'深色':'浅色'}</button></header><main className="aw-main">
 <ErrorBox text={seasons.error||roster.error} retry={seasons.error?seasons.retry:roster.retry}/>
 {!seasons.data&&!seasons.error&&<p role="status">加载评测批次…</p>}
 {seasons.data?.seasons?.length===0&&<p>暂无已完成的天梯报告</p>}
 {season&&!roster.data&&!roster.error&&<p role="status">加载 Agent…</p>}
 {roster.data&&<Routes><Route path="/" element={<Navigate replace to={route('/ladder',season)}/>}/><Route path="/ladder" element={<Ladder season={season} roster={roster.data} seasons={seasons.data?.seasons??[]}/>}/><Route path="/agents/:agentId" element={<AgentDetail season={season} roster={roster.data}/>}/><Route path="/games/:gameId" element={<Replay season={season} roster={roster.data}/>}/><Route path="/play" element={<Play season={season} roster={roster.data}/>}/><Route path="/training" element={<TrainingIndex season={season}/>}/><Route path="*" element={<Navigate replace to={route('/ladder',season)}/>}/></Routes>}
 </main></div>
}

function Ladder({season,roster,seasons}:{season:string;roster:Obj;seasons:Obj[]}){
 const [params,setParams]=useSearchParams();const type=params.get('type')??'all';const q=params.get('q')??''
 const rows=[...roster.agents].sort((a,b)=>(a.ranking?.rank??Infinity)-(b.ranking?.rank??Infinity)).filter(a=>(type==='all'||a.kind===type)&&[a.name,a.id,a.sha256??''].join(' ').toLowerCase().includes(q.toLowerCase()))
 function update(k:string,v:string){const p=new URLSearchParams(params);p.set(k,v);setParams(p)}
 return <><div className="aw-toolbar"><h1>天梯榜</h1><label>评测批次<select aria-label="评测批次" value={season} onChange={e=>setParams({season:e.target.value})}>{seasons.map(s=><option key={s.id} value={s.id}>{s.id}</option>)}</select></label><Link to={route('/training',season)}>训练记录</Link></div><div className="aw-toolbar"><label>类型<select aria-label="Agent 类型" value={type} onChange={e=>update('type',e.target.value)}><option value="all">全部</option><option value="neural">神经网络</option><option value="heuristic">启发式</option><option value="random">随机</option></select></label><input aria-label="搜索 Agent" placeholder="名称 / SHA" value={q} onChange={e=>update('q',e.target.value)}/><small>{roster.protocol} · {rows.length} Agent</small></div>
 {roster.warning&&<details className="aw-note"><summary>排名适用范围</summary>{roster.warning}<p>{roster.rating}</p></details>}
 <div className="aw-table"><table><thead><tr><th>#</th><th>Agent / 版本</th><th>类型</th><th>Elo</th><th>胜 / 平 / 负</th><th></th></tr></thead><tbody>{rows.map(a=><tr key={a.id}><td>{a.ranking?.rank??'—'}</td><td><Link to={route('/agents/'+enc(a.id),season)}>{a.name}</Link><small>{agentVersion(a)}</small></td><td><Tag agent={a}/></td><td>{fmt(a.ranking?.elo,0)}</td><td>{a.ranking?`${a.ranking.wins} / ${a.ranking.draws} / ${a.ranking.losses}`:'—'}</td><td><Link to={route('/play',season,{p1:a.id,p2:'human'})}>对战</Link></td></tr>)}</tbody></table></div>{rows.length===0&&<p>没有匹配的 Agent</p>}</>
}

function AgentDetail({season,roster}:{season:string;roster:Obj}){
 const {agentId=''}=useParams();const [params,setParams]=useSearchParams();const a=roster.agents.find((x:Obj)=>x.id===agentId);const opponent=params.get('opponent');
 const training=useData(a?'/seasons/'+enc(season)+'/training/'+enc(agentId):null)
 if(!a)return <p>Agent 不存在于此评测批次</p>
 const pairs=roster.matrix[agentId]??{}
 return <><div className="aw-breadcrumb"><Link to={route('/ladder',season)}>天梯</Link> / {a.name}</div><div className="aw-toolbar"><h1>{a.name}</h1><Tag agent={a}/><small>{agentVersion(a)}</small><Link className="button-link" to={route('/play',season,{p1:a.id,p2:'human'})}>发起对战</Link></div><div className="aw-toolbar"><span>Elo {fmt(a.ranking?.elo,0)}</span><span>排名 {a.ranking?.rank??'—'}</span><span>胜 / 平 / 负 {a.ranking?`${a.ranking.wins} / ${a.ranking.draws} / ${a.ranking.losses}`:'—'}</span></div>
 <div className="aw-detail-grid"><section><h2>交手关系</h2><div className="aw-table"><table><thead><tr><th>对手</th><th>胜 / 平 / 负</th><th>得分率</th><th></th></tr></thead><tbody>{Object.entries(pairs).filter(([,v])=>(v as Obj)?.games>0).map(([id,value])=>{const v=value as Obj;return <tr key={id}><td><Link to={route('/agents/'+enc(id),season)}>{roster.agents.find((x:Obj)=>x.id===id)?.name??id}</Link><small>P1 {v.p1_games??'—'} / P2 {v.p2_games??'—'}</small></td><td>{v.wins} / {v.draws} / {v.losses}</td><td>{pct(v.score)}</td><td><button onClick={()=>{const p=new URLSearchParams(params);p.set('opponent',id);setParams(p)}}>交手记录</button></td></tr>})}</tbody></table></div></section><section className="aw-panel"><h2>{a.kind==='neural'?'模型配置':'搜索 / 策略配置'}</h2><dl><dt>算法</dt><dd>{a.spec.algo.type}</dd><dt>策略</dt><dd>{JSON.stringify(a.spec.policy)}</dd><dt>Spec</dt><dd className="aw-hash">{a.identity}</dd>{a.sha256&&<><dt>SHA</dt><dd className="aw-hash">{a.sha256}</dd></>}</dl><details><summary>完整配置</summary><Json value={a.spec}/></details></section></div>
 {opponent&&<Matches key={season+agentId+opponent} season={season} agent={a} opponent={opponent} roster={roster}/>}
 <section className="aw-section"><h2>训练记录</h2><ErrorBox text={training.error} retry={training.retry}/>{!training.data&&!training.error&&<p role="status">查找 SHA 关联记录…</p>}{training.data?.reason&&<p>{training.data.reason}</p>}{training.data&&[...new Set<string>(training.data.runs.map((r:Obj)=>r.run_id))].map(id=><TrainingRun key={id} runId={id} sha={a.sha256}/>)}</section>
 </>
}

function TrainingRun({runId,sha}:{runId:string;sha?:string}){
 const [data,setData]=useState<RunDetail|null>(null),[error,setError]=useState(''),[retry,setRetry]=useState(0)
 useEffect(()=>{let active=true;setError('');setData(null);api.getRun(runId).then(d=>{if(active)setData(d)}).catch(e=>{if(active)setError(String(e))});return()=>{active=false}},[runId,retry])
 return <div className="aw-panel aw-training"><ErrorBox text={error} retry={()=>setRetry(n=>n+1)}/>{!data&&!error&&<p role="status">加载训练运行…</p>}{data&&<><div className="aw-toolbar"><h3>{data.run_name??runId}</h3><small>{data.source} · {data.status??'状态未知'} · {data.algorithm}</small></div><small className="aw-hash">Run {runId}</small>{data.source==='mlflow'&&data.experiment_id&&<a target="_blank" rel="noreferrer" href={'http://192.168.68.88:8742/#/experiments/'+enc(data.experiment_id)+'/runs/'+enc(runId)}>在 MLflow 查看 ↗</a>}<div className="aw-toolbar"><span>创建 {data.created_at??'—'}</span><span>版本 {data.checkpoints.length}</span></div>{data.warnings.map((w,i)=><details key={i}><summary>来源提示</summary>{w}</details>)}<details><summary>训练指标</summary><MetricsOnOpen runId={runId}/></details><details><summary>训练配置与来源</summary><Json value={{config:data.config,provenance:data.provenance,summary:data.summary_stats}}/></details><details><summary>Checkpoint 记录</summary><div className="aw-table"><table><thead><tr><th>文件</th><th>进度</th><th>SHA</th></tr></thead><tbody>{data.checkpoints.map((c,i)=><tr key={i} className={sha&&c.sha256===sha?'aw-selected':''}><td>{c.filename}{sha&&c.sha256===sha?' · 当前 Agent':''}</td><td>{c.progress_type??'未知'} {c.step??c.episode??'—'}</td><td className="aw-hash">{c.sha256??'未记录'}</td></tr>)}</tbody></table></div></details>{data.notes&&<details><summary>运行备注</summary><pre>{data.notes}</pre></details>}</>}</div>
}
function MetricsOnOpen({runId}:{runId:string}){const ref=useRef<HTMLDivElement>(null);const [open,setOpen]=useState(false);useEffect(()=>{const d=ref.current?.closest('details');const fn=()=>{if(d?.open)setOpen(true)};d?.addEventListener('toggle',fn);fn();return()=>d?.removeEventListener('toggle',fn)},[]);return <div ref={ref}>{open&&<MetricsView runId={runId} embedded/>}</div>}
function TrainingIndex({season}:{season:string}){
 const [runs,setRuns]=useState<RunSummary[]|null>(null),[error,setError]=useState(''),[q,setQ]=useState('');const [params,setParams]=useSearchParams()
 useEffect(()=>{let active=true;api.listRuns().then(d=>{if(active)setRuns(d.runs)}).catch(e=>{if(active)setError(String(e))});return()=>{active=false}},[])
 return <><div className="aw-breadcrumb"><Link to={route('/ladder',season)}>天梯</Link> / 训练记录</div><div className="aw-toolbar"><h1>训练记录</h1><input aria-label="搜索训练" value={q} onChange={e=>setQ(e.target.value)} placeholder="名称 / Run ID"/></div><ErrorBox text={error}/>{!runs&&!error&&<p role="status">加载训练记录…</p>}<div className="aw-table"><table><thead><tr><th>运行</th><th>来源</th><th>算法</th><th>状态</th><th></th></tr></thead><tbody>{runs?.filter(r=>[r.run_id,r.run_name].join(' ').toLowerCase().includes(q.toLowerCase())).map(r=><tr key={r.run_id}><td>{r.run_name??r.run_id}</td><td>{r.source}</td><td>{r.algorithm}</td><td>{r.status??'—'}</td><td><button onClick={()=>setParams({season,run:r.run_id})}>详情</button></td></tr>)}</tbody></table></div>{params.get('run')&&<TrainingRun key={params.get('run')} runId={params.get('run')!}/>}</>
}

function Matches({season,agent,opponent,roster}:{season:string;agent:Obj;opponent:string;roster:Obj}){
 const result=useData('/seasons/'+enc(season)+'/games?'+new URLSearchParams({agent:agent.id,opponent}));const [filter,setFilter]=useState('all');const [page,setPage]=useState(0)
 const rows=(result.data?.games??[]).filter((g:Obj)=>!g.control&&(filter==='all'||(filter==='win'?g.winner===agent.id:filter==='loss'?g.winner&&g.winner!==agent.id:!g.winner)))
 return <section className="aw-section"><div className="aw-toolbar"><h2>{agent.name} × {roster.agents.find((a:Obj)=>a.id===opponent)?.name??opponent}</h2><select aria-label="对局结果" value={filter} onChange={e=>{setFilter(e.target.value);setPage(0)}}><option value="all">全部结果</option><option value="win">胜</option><option value="loss">负</option><option value="draw">平</option></select><Link to={route('/play',season,{p1:agent.id,p2:opponent})}>双方再战</Link></div><ErrorBox text={result.error} retry={result.retry}/>{!result.data&&!result.error&&<p role="status">读取原始对局…</p>}{result.data?.warnings?.map((w:string)=><p key={w} className="warning">{w}</p>)}<div className="aw-table"><table><thead><tr><th>对局 / 开局</th><th>P1 / P2</th><th>结果</th><th>步数</th><th></th></tr></thead><tbody>{rows.slice(page*20,page*20+20).map((g:Obj)=><tr key={g.id}><td>{g.id.slice(0,8)}:{g.id.split(':')[1]}<small>开局 {g.opening}</small></td><td>{g.p1} / {g.p2}</td><td>{g.winner??'平局'}<small>{g.termination}</small></td><td>{g.steps}</td><td>{g.available?<Link to={route('/games/'+enc(g.id),season,{agent:agent.id,opponent,ply:0})}>回放</Link>:'无棋谱'}</td></tr>)}</tbody></table></div><div className="aw-toolbar"><small>{rows.length} 局</small><button disabled={page===0} onClick={()=>setPage(p=>p-1)}>上一页</button><button disabled={(page+1)*20>=rows.length} onClick={()=>setPage(p=>p+1)}>下一页</button></div></section>
}

export function Participant({agent,seat,active=false,winner}:{agent:Obj;seat:number;active?:boolean;winner?:number|null}){
 const sha=agent.actual_sha256??agent.sha256
 return <div className={'aw-participant aw-p'+seat} aria-label={'P'+seat+' 身份'}>
  <span className="aw-seat-dot">P{seat}</span><div className="aw-participant-name"><strong>{agent.kind==='human'?'本机玩家':agent.name??agent.id}</strong><small>{kind(agent)} · {agent.kind==='human'?'你控制此方':agentVersion(agent)}</small></div>
  <span className="aw-turn">{winner===seat?'获胜':active?'行棋中':''}</span>
  {agent.kind!=='human'&&<details><summary>版本</summary><dl><dt>Agent</dt><dd>{agent.id}</dd><dt>策略</dt><dd>{agent.spec?.policy?.type??'—'} · k={agent.spec?.policy?.k??'—'}</dd>{sha&&<><dt>Checkpoint</dt><dd className="aw-hash">{sha}</dd></>}</dl><Json value={agent.spec??agent}/></details>}
 </div>
}

function Replay({season,roster}:{season:string;roster:Obj}){
 const {gameId=''}=useParams();const [params,setParams]=useSearchParams();const r=useData('/seasons/'+enc(season)+'/replay/'+enc(gameId))
 const [derived,setDerived]=useState<Position|null>(null),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[panel,setPanel]=useState('analysis')
 const total=r.data?.actions.length??0
 const ply=Math.max(0,Math.min(Math.trunc(Number(params.get('ply'))||0),total));const position=r.data?.positions[ply]
 const participant=(id:string)=>roster.agents.find((a:Obj)=>a.id===id)??{id,name:id,kind:'unknown'}
 const actor=position?participant(r.data?.[position.turn===1?'p1':'p2']):null
 const actual=r.data?.actions[ply];const occurrence=r.data?.actions.slice(0,ply).filter((a:Obj)=>a.player===position?.turn).length??0
 const latency=actor?r.data?.decision_milliseconds?.[actor.id]?.[occurrence]:undefined
 function seek(n:number,pause=true){if(pause)setPlaying(false);const p=new URLSearchParams(params);p.set('ply',String(Math.max(0,Math.min(n,total))));setParams(p,{replace:true})}
 function toggle(){if(!total)return;if(ply===total)seek(0,false);setPlaying(v=>!v)}
 useEffect(()=>{setDerived(null)},[ply,gameId])
 useEffect(()=>{setPlaying(false)},[gameId,season])
 useEffect(()=>{if(!playing||!r.data)return;if(ply>=total){setPlaying(false);return}const t=setTimeout(()=>seek(ply+1,false),1000/speed);return()=>clearTimeout(t)},[playing,ply,total,speed,r.data])
 useEffect(()=>{function key(e:KeyboardEvent){const el=e.target as HTMLElement;if(e.altKey||e.ctrlKey||e.metaKey||el.closest?.('input,select,textarea,button,a,[contenteditable="true"],[role="tab"]'))return;if(['ArrowLeft','ArrowRight','Home','End',' '].includes(e.key)){e.preventDefault();if(e.key===' ')toggle();else seek(e.key==='Home'?0:e.key==='End'?total:ply+(e.key==='ArrowLeft'?-1:1))}}window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key)},[ply,total,playing,params])
 return <><div className="aw-breadcrumb"><Link to={route('/ladder',season)}>天梯</Link> / <Link to={route('/agents/'+enc(params.get('agent')??r.data?.p1??''),season,{opponent:params.get('opponent')??r.data?.p2})}>交手记录</Link> / {gameId.slice(0,12)}</div><ErrorBox text={r.error} retry={r.retry}/>{!r.data&&!r.error&&<p role="status">验证原始棋谱…</p>}{r.data&&position&&<>
 <div className="aw-toolbar"><h1>对局回放</h1><small>{r.data.winner?participant(r.data.winner).name+' 胜':'平局'} · {r.data.termination}</small><Link to={route('/play',season,{p1:r.data.p1,p2:r.data.p2})}>双方再战</Link></div>
 <div className="aw-replay aw-workbench"><section className="aw-board-column">
 <Participant agent={participant(r.data.p2)} seat={2} active={!derived&&!!actual&&position.turn===2}/>
 <EvalReplayBoard state={positionBoard(derived??position)} size={430}/>
 <Participant agent={participant(r.data.p1)} seat={1} active={!derived&&!!actual&&position.turn===1}/>
 <div className="aw-transport" aria-label="回放控制">
 <input type="range" aria-label="回放步数" min={0} max={total} value={ply} onChange={e=>seek(+e.target.value)}/>
 <div className="aw-toolbar"><button aria-label="回到开头" title="Home" disabled={ply===0} onClick={()=>seek(0)}>⏮</button><button aria-label="上一步" title="←" disabled={ply===0} onClick={()=>seek(ply-1)}>◀</button><button aria-label={playing?'暂停回放':'播放回放'} title="空格" disabled={!total} onClick={toggle}>{playing?'暂停':'播放'}</button><button aria-label="下一步" title="→" disabled={ply===total} onClick={()=>seek(ply+1)}>▶</button><button aria-label="跳到结束" title="End" disabled={ply===total} onClick={()=>seek(total)}>⏭</button><select aria-label="播放速度" value={speed} onChange={e=>setSpeed(+e.target.value)}>{[0.5,1,2,4].map(v=><option key={v} value={v}>{v}×</option>)}</select><label><input aria-label="跳转步数" type="number" min={0} max={total} value={ply} onChange={e=>seek(Math.trunc(+e.target.value))}/> / {total}</label></div>
 <div className="aw-toolbar"><small>{derived?'候选分支 · 原始第 '+ply+' 步':'原始回放'}</small>{derived&&<button onClick={()=>setDerived(null)}>返回原局面</button>}</div></div>
 </section><section className="aw-panel aw-inspector">
 <div className="aw-tabs" role="tablist" aria-label="回放视图">{[['analysis','逐步分析'],['moves','棋谱'],['source','证据']].map(([id,label])=><button key={id} role="tab" aria-selected={panel===id} onClick={()=>setPanel(id)}>{label}</button>)}</div>
 <div className="aw-inspector-body"><div hidden={panel!=='analysis'}><h2>第 {ply} 步 · P{position.turn}</h2><div className="aw-record"><strong>{actor?.name}</strong> · 实际动作 {actual?notation(actual):'终局'}<small>耗时 {fmt(latency)} ms · 候选分数 / 搜索树：未记录</small></div>
 <div hidden={!actual}><StepAnalysis key={gameId} position={position} actor={actor} roster={roster} onBranch={setDerived} onAnalyze={()=>setPlaying(false)}/></div>{!actual&&<p>对局已结束</p>}</div>
 {panel==='moves'&&<div className="aw-move-list">{r.data.actions.map((a:Obj,i:number)=><button key={i} aria-current={ply===i?'step':undefined} onClick={()=>seek(i)}><span>{i+1}</span><span>P{a.player}</span><strong>{notation(a)}</strong></button>)}</div>}
 {panel==='source'&&<><h3>原始证据</h3><Json value={{source:r.data.source,decision_diagnostics:r.data.decision_diagnostics,scope:r.data.diagnostics_scope}}/></>}
 </div></section></div></>}</>
}

function notation(a:Obj){return 'abcdefghi'[a.target[0]]+(a.target[1]+1)+({horizontal:' H',vertical:' V',pawn:''}[a.type as string]??'')}

export function StepAnalysis({position,actor,roster,onBranch,onAnalyze}:{position:Position;actor:Obj;roster:Obj;onBranch:(p:Position|null)=>void;onAnalyze?:()=>void}){
 const [id,setId]=useState(actor?.id??roster.agents.find((a:Obj)=>a.spec.algo.type==='greedy')?.id??'')
 const [result,setResult]=useState<Obj|null>(null),[branch,setBranch]=useState<Obj|null>(null),[tensor,setTensor]=useState<Obj|null>(null),[layer,setLayer]=useState(''),[channel,setChannel]=useState(0),[error,setError]=useState(''),[busy,setBusy]=useState(false)
 const [tab,setTab]=useState('decision'),[limit,setLimit]=useState(10),[network,setNetwork]=useState('activation'),[input,setInput]=useState(0)
 const gen=useRef(0),abort=useRef<AbortController|null>(null);const selected=roster.agents.find((a:Obj)=>a.id===id);const algo=selected?.spec.algo.type;const supported=['dqn','greedy','minimax'].includes(algo)
 const payload={position,algorithm:algo,depth:selected?.spec.algo.params?.depth??1,...(algo==='dqn'?{model_id:selected.sha256}:{})}
 const positionKey=JSON.stringify(position)
 useEffect(()=>()=>{gen.current++;abort.current?.abort()},[])
 useEffect(()=>{clear()},[positionKey])
 function clear(){gen.current++;abort.current?.abort();setResult(null);setBranch(null);setTensor(null);setError('');setBusy(false);onBranch(null)}
 async function run(path:string,body:Obj,apply:(v:Obj)=>void){onAnalyze?.();abort.current?.abort();const c=new AbortController();abort.current=c;const token=++gen.current;setBusy(true);setError('');try{const d=await labRequest(path,body,c.signal);if(token===gen.current)apply(d)}catch(e:any){if(token===gen.current&&e.name!=='AbortError')setError(String(e))}finally{if(token===gen.current)setBusy(false)}}
 const score=(v:any)=>v==='+inf'?Infinity:v==='-inf'?-Infinity:Number(v??-Infinity)
 const candidates=[...(result?.candidates??[])].sort((a,b)=>score(b.value)-score(a.value))
 const pv=branch?.principal_variation??[]
 return <div className="aw-analysis"><div className="aw-toolbar"><label>分析 Agent<select aria-label="分析 Agent" value={id} onChange={e=>{clear();setId(e.target.value);setTab('decision')}}>{roster.agents.map((a:Obj)=><option key={a.id} value={a.id}>{a.name}</option>)}</select></label><button disabled={busy||!supported} onClick={()=>void run('analyze',payload,d=>{setResult(d);setBranch(null);setTensor(null);onBranch(null);if(!d.neural?.layers?.some((l:Obj)=>l.name===layer)){setLayer(d.neural?.layers?.[0]?.name??'');setChannel(0)}})}>{busy?'计算中…':'运行分析'}</button></div>
 {!supported&&<p>此 Agent 暂无诊断接口</p>}<ErrorBox text={error}/>{!result&&supported&&!busy&&<small>当前局面未分析</small>}
 {result&&<><div className="aw-toolbar"><small>{result.neural?.model?.output_semantics??'启发式评估分'} · 重新计算 · 原始局面</small>{branch&&<button onClick={()=>{setBranch(null);onBranch(null)}}>返回原局面</button>}</div>
 <div className="aw-tabs" role="tablist" aria-label="分析视图">{[['decision','决策'],...(algo==='minimax'?[['search','搜索']]:[]),...(result.neural?[['network','神经网络诊断']]:[]),['raw','原始数据']].map(([key,label])=><button key={key} role="tab" aria-selected={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
 {tab==='decision'&&<><div className="aw-toolbar"><span>{candidates.length} 个合法候选</span><select aria-label="候选数量" value={limit} onChange={e=>setLimit(+e.target.value)}><option value={5}>前 5</option><option value={10}>前 10</option><option value={999}>全部</option></select></div><div className="aw-candidates"><table><thead><tr><th>候选</th><th>数值 / 界</th><th></th></tr></thead><tbody>{candidates.slice(0,limit).map((c:Obj)=><tr key={c.id} className={branch?.action_id===c.id?'aw-selected':''}><td>{c.label??c.id}{c.id===result.selected_id?' ✓':''}</td><td>{c.bound==='upper'?'≤ ':c.bound==='lower'?'≥ ':''}{typeof c.value==='number'?fmt(c.value,4):c.value}</td><td><button disabled={busy} onClick={()=>void run('branch',{...payload,action:c.action},d=>{setBranch(d);onBranch(d.after?.position??null)})}>变化</button></td></tr>)}</tbody></table></div>{branch&&<small>候选分支 · {branch.action_id} · {branch.bound} · {typeof branch.value==='number'?fmt(branch.value,4):branch.value}</small>}</>}
 {tab==='search'&&<>{!branch?<p>在决策面板选择候选变化</p>:<><div className="aw-toolbar"><span>节点 {branch.stats?.nodes??'—'}</span><span>遗漏 {branch.trace?.omitted??0}</span></div><h3>主变化</h3><div className="aw-moves">{pv.map((p:Obj,i:number)=><button key={i} disabled={!p.position&&!p.after?.position} onClick={()=>onBranch(p.position??p.after?.position)}>{i+1}. {p.action?notation(p.action):p.action_id??'局面'}</button>)}</div><div className="aw-table"><table><thead><tr><th>节点 / 父节点</th><th>值 / 界</th><th></th></tr></thead><tbody>{branch.trace?.nodes?.map((n:Obj)=><tr key={n.id}><td>{n.id} / {n.parent??'根'}</td><td>{n.score??n.value??'—'} · {n.bound}</td><td><button disabled={!n.position} onClick={()=>onBranch(n.position)}>查看局面</button></td></tr>)}</tbody></table></div></>}</>}
 {tab==='network'&&result.neural&&<><div className="aw-toolbar"><small>{result.neural.model.architecture} · {result.neural.model.parameters} 参数</small><select aria-label="网络诊断内容" value={network} onChange={e=>{setNetwork(e.target.value);setTensor(null)}}><option value="activation">层激活</option><option value="input">输入平面</option><option value="sensitivity">候选敏感度</option></select></div>
 {network==='input'&&<><select aria-label="输入平面" value={input} onChange={e=>setInput(+e.target.value)}>{result.neural.input?.views?.map((v:Obj,i:number)=><option key={i} value={i}>{v.name}</option>)}</select>{result.neural.input?.views?.[input]&&<Tensor data={result.neural.input.views[input]}/>}</>}
 {network==='activation'&&<><div className="aw-toolbar"><label>层<select aria-label="网络层" value={layer} onChange={e=>{setLayer(e.target.value);setTensor(null);setChannel(0)}}>{result.neural.layers?.map((l:Obj)=><option key={l.name} value={l.name}>{l.name} [{l.shape.join(',')}]</option>)}</select></label><label>通道<input aria-label="激活通道" type="number" min={0} value={channel} onChange={e=>{setChannel(+e.target.value);setTensor(null)}}/></label><button disabled={busy||!layer} onClick={()=>void run('neural/tensor',{position,model_id:selected.sha256,kind:'activation',layer,channel,input_channel:0,offset:0},setTensor)}>读取激活</button></div>{tensor&&<Tensor data={tensor}/>}</>}
 {network==='sensitivity'&&(branch?.neural?.sensitivities?.map((v:Obj)=><details key={v.objective}><summary>{v.objective}</summary>{v.views?.map((t:Obj)=><Tensor key={t.name} data={t}/>)}</details>)??<p>先选择候选变化</p>)}</>}
 {tab==='raw'&&<Json value={{analysis:result,branch,tensor}}/>}</>}
 </div>
}

function Play({season,roster}:{season:string;roster:Obj}){
 const [params,setParams]=useSearchParams();const [game,setGame]=useState<Obj|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[auto,setAuto]=useState(false),[analysis,setAnalysis]=useState(false),[derived,setDerived]=useState<Position|null>(null),[mode,setMode]=useState('pawn')
 const gameId=params.get('game');const p1=params.get('p1')??'human',p2=params.get('p2')??roster.agents.find((a:Obj)=>a.spec.algo.type==='greedy')?.id??'human'
 const token=()=>sessionStorage.getItem('agent-preview:'+gameId)??''
 useEffect(()=>{let active=true;setGame(null);setError('');setAuto(false);setAnalysis(false);if(gameId)evidence('/play/'+enc(gameId),{headers:{'X-Game-Token':token()}}).then(d=>{if(active)setGame(d)}).catch(e=>{if(active)setError(String(e))});return()=>{active=false}},[gameId])
 async function create(){setBusy(true);setError('');try{const d=await evidence('/play',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({season,p1,p2})});sessionStorage.setItem('agent-preview:'+d.id,d.token);const p=new URLSearchParams(params);p.set('game',d.id);setParams(p);setGame(d)}catch(e){setError(String(e))}finally{setBusy(false)}}
 async function step(action?:Obj){if(!game||busy)return;setBusy(true);setError('');try{const d=await evidence('/play/'+enc(gameId!)+'/step',{method:'POST',headers:{'Content-Type':'application/json','X-Game-Token':token()},body:JSON.stringify({revision:game.revision,...(action?{action}:{})})});setGame(d)}catch(e){setError(String(e));setAuto(false)}finally{setBusy(false)}}
 const actor=game?.identities[game.position.turn-1];const ended=game&&(game.winner!==null||game.limit)
 useEffect(()=>{if(!auto||busy||!game||ended||actor?.kind==='human'||analysis)return;const t=setTimeout(()=>void step(),350);return()=>clearTimeout(t)},[auto,busy,game,analysis])
 async function close(){setAuto(false);setBusy(true);try{await evidence('/play/'+enc(gameId!),{method:'DELETE',headers:{'X-Game-Token':token()}});sessionStorage.removeItem('agent-preview:'+gameId);const p=new URLSearchParams(params);p.delete('game');setParams(p);setGame(null)}catch(e){setError(String(e))}finally{setBusy(false)}}
 function move(x:number,y:number){if(!game||actor?.kind!=='human'||ended||busy||analysis)return;const action=game.legal_actions.find((a:Obj)=>a.type===mode&&a.target[0]===x&&a.target[1]===y);if(action)void step(action);else setError('该位置不可执行所选动作')}
 return <><div className="aw-toolbar"><h1>对战</h1><small>独立预览 · 不计入天梯 · 会话内存保存</small></div><ErrorBox text={error}/>{!gameId?<><div className="aw-seats">{[1,2].map(n=><section key={n} className="aw-panel"><h2>P{n} · {n===1?'先手':'后手'}</h2><Participant seat={n} agent={(n===1?p1:p2)==='human'?{id:'human',kind:'human'}:roster.agents.find((a:Obj)=>a.id===(n===1?p1:p2))??{id:n===1?p1:p2,name:'不可用 Agent',kind:'unknown'}}/><select aria-label={'P'+n+' 对战方'} value={n===1?p1:p2} onChange={e=>{const p=new URLSearchParams(params);p.set('p'+n,e.target.value);setParams(p)}}><option value="human">本机玩家（你）</option>{roster.agents.map((a:Obj)=><option key={a.id} value={a.id} disabled={!a.playable}>{a.name} · {kind(a)} · {agentVersion(a)}</option>)}</select></section>)}</div><div className="aw-toolbar"><button disabled={busy} onClick={()=>{const p=new URLSearchParams(params);p.set('p1',p2);p.set('p2',p1);setParams(p)}}>交换</button><button disabled={busy} onClick={()=>void create()}>{busy?'加载确切 Agent…':'开始对局'}</button></div></>:!game&&!error?<p role="status">加载对局…</p>:null}
 {game&&<><div className="aw-toolbar"><span>{game.identities[0].name} × {game.identities[1].name}</span><span>{ended?(game.winner?'P'+game.winner+' 胜':'达到步数上限'):'P'+game.position.turn+' 行棋'} · {game.revision} 步</span><button disabled={busy} onClick={()=>void close()}>关闭对局</button></div><div className="aw-replay aw-workbench"><section className="aw-board-column"><Participant seat={2} agent={{...roster.agents.find((a:Obj)=>a.id===game.identities[1].id),...game.identities[1]}} active={!ended&&game.position.turn===2} winner={game.winner}/><EvalReplayBoard state={positionBoard(derived??game.position)} size={430} onCellClick={move}/><Participant seat={1} agent={{...roster.agents.find((a:Obj)=>a.id===game.identities[0].id),...game.identities[0]}} active={!ended&&game.position.turn===1} winner={game.winner}/><div className="aw-toolbar"><select aria-label="动作类型" value={mode} onChange={e=>setMode(e.target.value)} disabled={analysis}><option value="pawn">移动棋子</option><option value="horizontal">横墙</option><option value="vertical">竖墙</option></select>{actor.kind!=='human'&&!ended&&<><button disabled={busy||analysis} onClick={()=>void step()}>Agent 下一手</button><label><input type="checkbox" checked={auto} disabled={analysis} onChange={e=>setAuto(e.target.checked)}/>自动行棋</label></>}</div>{actor.kind==='human'&&!ended&&!analysis&&<select aria-label="合法动作" value="" disabled={busy} onChange={e=>{if(e.target.value)void step(JSON.parse(e.target.value))}}><option value="">选择合法动作</option>{game.legal_actions.map((a:Obj)=><option key={JSON.stringify(a)} value={JSON.stringify(a)}>{notation(a)}</option>)}</select>}</section><section className="aw-panel aw-inspector aw-play-inspector">{analysis?<><button onClick={()=>{setAnalysis(false);setDerived(null)}}>返回对局</button><small>冻结局面 · 未修改对局</small><StepAnalysis key={gameId+':'+game.revision} position={game.position} actor={actor} roster={roster} onBranch={setDerived}/></>:<><h2>对局信息</h2><dl><dt>状态</dt><dd>{busy?'计算中':'就绪'}</dd>{game.identities.map((a:Obj,i:number)=><React.Fragment key={i}><dt>P{i+1}</dt><dd>{a.name} · {kind(a)}<small className="aw-hash">{a.actual_sha256??''}</small></dd></React.Fragment>)}</dl><button disabled={busy||ended} onClick={()=>{setAuto(false);setAnalysis(true)}}>分析当前局面</button><details><summary>走子记录</summary><div className="aw-moves">{game.actions.map((a:Obj,i:number)=><span key={i}>{i+1}. P{a.player} {notation(a)}</span>)}</div></details></>}</section></div></>}
 </>
}
