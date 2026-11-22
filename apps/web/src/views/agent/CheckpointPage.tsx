import {useEffect, useState} from 'react'
import {Link, useParams, useSearchParams} from 'react-router-dom'
import {trainingUrl} from '../TrainingWorkspace'
import {type Obj, evidence, enc, fmt, route} from './shared'
import {Matches} from './Matches'

export function CheckpointPage({season}:{season:string}){
 const {sha=''}=useParams();const [params,setParams]=useSearchParams();const [catalog,setCatalog]=useState<Obj[]>([]),[errors,setErrors]=useState<string[]>([]),[loading,setLoading]=useState(true),[retry,setRetry]=useState(0)
 useEffect(()=>{const c=new AbortController();setCatalog([]);setErrors([]);setLoading(true)
 evidence('/seasons',{signal:c.signal}).then(async d=>{const results=await Promise.allSettled(d.seasons.map((s:Obj)=>evidence('/seasons/'+enc(s.id),{signal:c.signal})));if(c.signal.aborted)return;setCatalog(results.flatMap(r=>r.status==='fulfilled'?[r.value]:[]));setErrors([...(d.warnings??[]),...results.flatMap((r,i)=>r.status==='rejected'?[d.seasons[i].id+': '+String(r.reason)]:[])])}).catch(e=>{if(!c.signal.aborted)setErrors([String(e)])}).finally(()=>{if(!c.signal.aborted)setLoading(false)});return()=>c.abort()},[sha,retry])
 const rows=catalog.flatMap(r=>r.agents.filter((a:Obj)=>a.sha256===sha).map((agent:Obj)=>({roster:r,agent})))
 const selected=params.get('agent')?rows.find(r=>r.roster.season===season&&r.agent.id===params.get('agent')):rows.find(r=>r.roster.season===season)??rows[0]
 const opponents=selected?Object.entries(selected.roster.matrix[selected.agent.id]??{}).filter(([,v])=>(v as Obj)?.games>0).map(([id])=>id):[]
 const opponent=opponents.includes(params.get('opponent')??'')?params.get('opponent')!:opponents[0]
 function choose(r:Obj){const p=new URLSearchParams(params);p.set('season',r.roster.season);p.set('agent',r.agent.id);p.delete('opponent');setParams(p)}
 return <><div className="aw-breadcrumb"><Link to={trainingUrl(season,params.get('run')?[params.get('run')!]:[])}>训练记录</Link> / Checkpoint</div><div className="aw-toolbar"><h1>Checkpoint</h1><span className="aw-hash">{sha}</span></div>
 {loading&&<p role="status">查找确切 SHA 评测…</p>}{errors.length>0&&<div className="error" role="alert">{errors.map(e=><div key={e}>{e}</div>)}<button onClick={()=>setRetry(v=>v+1)}>重试</button></div>}
 {!loading&&!rows.length&&<p>{errors.length?'部分评测来源不可用，尚未找到确切关联':'已完成评测中没有此 checkpoint；暂不可从评测 Agent 发起对战'}</p>}
 {rows.length>0&&<><h2>评测</h2><div className="aw-table"><table><thead><tr><th>批次 / 协议</th><th>Agent / 策略</th><th>胜 / 平 / 负</th><th>Elo</th><th></th></tr></thead><tbody>{rows.map(r=><tr key={r.roster.season+':'+r.agent.id} className={selected===r?'aw-selected':''}><td>{r.roster.season}<small>{r.roster.protocol}</small></td><td>{r.agent.name}<small>{r.agent.spec.policy.type} · k={r.agent.spec.policy.k??'—'}</small></td><td>{r.agent.ranking?`${r.agent.ranking.wins} / ${r.agent.ranking.draws} / ${r.agent.ranking.losses}`:'—'}</td><td>{fmt(r.agent.ranking?.elo,0)}</td><td><button onClick={()=>choose(r)}>交手记录</button></td></tr>)}</tbody></table></div></>}
 {!loading&&params.get('agent')&&!selected&&<p role="alert">所选 Agent 不属于此 checkpoint</p>}
 {selected&&<section className="aw-section"><div className="aw-toolbar"><h2>{selected.agent.name}</h2><Link to={route('/agents/'+enc(selected.agent.id),selected.roster.season)}>Agent 详情</Link>{selected.agent.playable&&<Link className="button-link" to={route('/play',selected.roster.season,{p1:selected.agent.id,p2:'human'})}>发起对战</Link>}<label>回放对手<select aria-label="Checkpoint 对手" value={opponent??''} onChange={e=>{const p=new URLSearchParams(params);p.set('season',selected.roster.season);p.set('agent',selected.agent.id);p.set('opponent',e.target.value);setParams(p)}}>{opponents.map(id=><option key={id} value={id}>{selected.roster.agents.find((a:Obj)=>a.id===id)?.name??id}</option>)}</select></label></div>
 {selected.roster.warning&&<details><summary>评测范围</summary>{selected.roster.warning}</details>}
 {opponent?<div className="aw-checkpoint-matches"><Matches key={selected.roster.season+':'+selected.agent.id+':'+opponent} season={selected.roster.season} agent={selected.agent} opponent={opponent} roster={selected.roster}/></div>:<p>没有已记录的交手对局</p>}
 </section>}</>
}
