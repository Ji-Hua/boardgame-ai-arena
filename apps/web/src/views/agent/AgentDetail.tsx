import {useEffect, useState} from 'react'
import {Link, useParams, useSearchParams} from 'react-router-dom'
import {RunInspector, trainingUrl} from '../TrainingWorkspace'
import {type Obj, useData, enc, fmt, pct, route, ErrorBox, Json, Tag, agentVersion} from './shared'
import {Matches} from './Matches'
import {Failures} from './Failures'

export function AgentDetail({season,roster}:{season:string;roster:Obj}){
 const {agentId=''}=useParams();const [params,setParams]=useSearchParams();const [query,setQuery]=useState('');
 const a=roster.agents.find((x:Obj)=>x.id===agentId)
 const training=useData(a?'/seasons/'+enc(season)+'/training/'+enc(agentId):null)
 useEffect(()=>setQuery(''),[agentId,season])
 if(!a)return <p>Agent 不存在于此评测批次</p>
 const pairs=Object.entries(roster.matrix[agentId]??{}).filter(([,v])=>(v as Obj)?.games>0)
 const opponent=pairs.some(([id])=>id===params.get('opponent'))?params.get('opponent')!:pairs[0]?.[0]
 const panel=['matches','failures','training','config'].includes(params.get('panel')??'')?params.get('panel')!:'matches'
 const runs=[...new Set<string>((training.data?.runs??[]).map((r:Obj)=>r.run_id))]
 const run=runs.includes(params.get('run')??'')?params.get('run')!:runs[0]
 function update(values:Record<string,string>){const p=new URLSearchParams(params);Object.entries(values).forEach(([k,v])=>p.set(k,v));setParams(p)}
 const filtered=pairs.filter(([id])=>((roster.agents.find((x:Obj)=>x.id===id)?.name??id)+' '+id).toLowerCase().includes(query.toLowerCase()))
 return <><div className="aw-breadcrumb"><Link to={route('/ladder',season)}>天梯</Link> / {a.name}</div>
 <div className="aw-toolbar"><h1>{a.name}</h1><Tag agent={a}/><small>{agentVersion(a)}</small><Link className="button-link" to={route('/play',season,{p1:a.id,p2:'human'})}>发起对战</Link></div>
 <div className="aw-toolbar aw-metrics"><span>Elo {fmt(a.ranking?.elo,0)}</span><span>排名 {a.ranking?.rank??'—'}</span><span>胜 / 平 / 负 {a.ranking?`${a.ranking.wins} / ${a.ranking.draws} / ${a.ranking.losses}`:'—'}</span></div>
 <div className="aw-agent-desk">
 <section className="aw-panel aw-opponents"><div className="aw-toolbar"><h2>交手关系</h2><small>{filtered.length} / {pairs.length}</small></div><input aria-label="筛选对手" placeholder="筛选对手" value={query} onChange={e=>setQuery(e.target.value)}/>
 <div className="aw-opponent-list">{filtered.map(([id,value])=>{const v=value as Obj;return <button key={id} className="aw-opponent" aria-pressed={opponent===id} onClick={()=>update({opponent:id,panel:'matches'})}><strong>{roster.agents.find((x:Obj)=>x.id===id)?.name??id}</strong><b>{pct(v.score)}</b><small>{v.wins} / {v.draws} / {v.losses} · P1 {v.p1_games??'—'} / P2 {v.p2_games??'—'}</small><span>交手记录 →</span></button>})}{!filtered.length&&<p className="muted">无匹配对手</p>}</div></section>
 <section className="aw-panel aw-agent-inspector"><div className="aw-tabs" role="tablist" aria-label="Agent 详情"><button role="tab" aria-selected={panel==='matches'} onClick={()=>update({panel:'matches'})}>交手记录</button><button role="tab" aria-selected={panel==='failures'} onClick={()=>update({panel:'failures'})}>失败样本</button><button role="tab" aria-selected={panel==='training'} onClick={()=>update({panel:'training'})}>训练记录{runs.length?` · ${runs.length}`:''}</button><button role="tab" aria-selected={panel==='config'} onClick={()=>update({panel:'config'})}>配置</button></div>
 <div className="aw-agent-content" role="tabpanel">
 {panel==='matches'&&(opponent?<><Link className="aw-opponent-link" to={route('/agents/'+enc(opponent),season)}>查看对手 Agent ↗</Link><Matches key={season+agentId+opponent} season={season} agent={a} opponent={opponent} roster={roster}/></>:<p className="muted">暂无交手记录</p>)}
 {panel==='failures'&&<Failures key={season+agentId} season={season} agent={a} roster={roster}/>}
 {panel==='config'&&<><h2>{a.kind==='neural'?'模型配置':'搜索 / 策略配置'}</h2><dl><dt>算法</dt><dd>{a.spec.algo.type}</dd><dt>策略</dt><dd>{JSON.stringify(a.spec.policy)}</dd><dt>Spec</dt><dd className="aw-hash">{a.identity}</dd>{a.sha256&&<><dt>SHA</dt><dd className="aw-hash">{a.sha256}</dd></>}</dl><Json value={a.spec}/></>}
 {panel==='training'&&<><ErrorBox text={training.error} retry={training.retry}/>{!training.data&&!training.error&&<p role="status">查找 SHA 关联记录…</p>}{training.data?.reason&&<p className="muted">{training.data.reason}</p>}{runs.length>1&&<label className="aw-toolbar">Run<select aria-label="Agent 训练 Run" value={run} onChange={e=>update({run:e.target.value})}>{runs.map(id=><option key={id} value={id}>{training.data?.runs.find((r:Obj)=>r.run_id===id)?.run_name??id}</option>)}</select></label>}{run&&<TrainingRun key={run} runId={run} sha={a.sha256} season={season}/>}</>}
 </div></section></div></>
}

function TrainingRun({runId,sha,season}:{runId:string;sha?:string;season:string}){return <><div className="aw-training-open"><Link to={trainingUrl(season,[runId])}>打开训练工作区 / 添加 Run 对比</Link></div><RunInspector ids={[runId]} season={season} sha={sha}/></>}
