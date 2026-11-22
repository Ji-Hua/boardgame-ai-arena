import {useEffect, useState} from 'react'
import {Link} from 'react-router-dom'
import {type Obj, request, enc, fmt, when, ErrorBox} from './shared'

const TOKEN_KEY='quoridor-platform-token'
const readToken=()=>{try{return localStorage.getItem(TOKEN_KEY)??''}catch{return ''}}
const wdl=(r:Obj)=>`${r.wins??'—'} / ${r.draws??'—'} / ${r.losses??'—'}`

/** Deployment decision for one checkpoint: gate evidence, the confirmed decision log entry, promote or withdraw. */
export function PromotionPanel({sha,onChange}:{sha:string;onChange:()=>void}){
 const [data,setData]=useState<Obj|null>(null),[error,setError]=useState(''),[tick,setTick]=useState(0)
 const [season,setSeason]=useState(''),[note,setNote]=useState(''),[token,setToken]=useState(readToken),[busy,setBusy]=useState(false)
 useEffect(()=>{const c=new AbortController();request('/api/v1/promotions/candidates/'+enc(sha),{signal:c.signal}).then(d=>{if(!c.signal.aborted){setData(d);setError('')}}).catch(e=>{if(!c.signal.aborted)setError(String(e))});return()=>c.abort()},[sha,tick])
 const gates:Obj[]=data?.gates??[],chosen=gates.some(g=>g.season===season)?season:gates[0]?.season??''
 async function send(path:string,body:Obj){setBusy(true);setError('');try{try{localStorage.setItem(TOKEN_KEY,token)}catch{/* optional */}
  await request('/api/v1/promotions'+path,{method:'POST',headers:{'Content-Type':'application/json','X-Platform-Token':token},body:JSON.stringify({sha256:sha,note,...body})})
  setNote('');setTick(t=>t+1);onChange()}catch(e){setError(String(e))}finally{setBusy(false)}}
 if(!data)return <>{error?<ErrorBox text={error} retry={()=>setTick(t=>t+1)}/>:<p role="status">读取门槛评测与上线记录…</p>}</>
 const d=data.deployed
 return <div className="pf-promotion">
  <p className="pf-note">上线是一次需要人确认的决定：只追加到审计日志 <code>promotions.jsonl</code>，不改动现有阵容、champion 别名或天梯批次。条件是该 checkpoint 作为候选完成过门槛评测。</p>
  {d&&<section className="aw-panel"><h2>已上线</h2><p>{when(d.time)} · 门槛评测批次 <Link to={'/ladder?season='+enc(d.season)}>{d.season}</Link>（{d.gate}）· 排名 {d.evidence.rank} / {d.evidence.roster_size} · Elo {fmt(d.evidence.elo,0)} · 胜 / 平 / 负 {wdl(d.evidence)}</p><blockquote>{d.note}</blockquote></section>}
  {gates.length?<div className="aw-table"><table><thead><tr><th/><th>门槛评测批次</th><th>排名</th><th>Elo</th><th>得分</th><th>胜 / 平 / 负</th><th>对各对手（胜 / 平 / 负）</th></tr></thead><tbody>
   {gates.map(g=><tr key={g.season}><td>{!d&&<input type="radio" name="gate" aria-label={'选择 '+g.season} checked={chosen===g.season} onChange={()=>setSeason(g.season)}/>}</td><td><Link to={'/ladder?season='+enc(g.season)}>{g.season}</Link><small>{g.gate} · {when(g.published)}</small></td><td>{g.rank} / {g.roster_size}</td><td>{fmt(g.elo,0)}</td><td>{fmt(g.score,3)}<small>95% {g.ci95?`${fmt(g.ci95[0],2)}–${fmt(g.ci95[1],2)}`:'—'}</small></td><td>{wdl(g)}</td>
    <td><ul className="pf-samples">{Object.entries(g.opponents as Record<string,Obj>).map(([o,r])=><li key={o}>{o} · {wdl(r)}</li>)}</ul></td></tr>)}
  </tbody></table></div>:<p>这个 checkpoint 还没有完成的门槛评测。<Link to={'/jobs/new?entrypoint=run_arena_matrix&sha='+enc(sha)}>新建门槛评测任务</Link></p>}
  {(d||gates.length>0)&&<div className="pf-form">
   <label>{d?'撤下理由':'上线理由'}<textarea aria-label={d?'撤下理由':'上线理由'} rows={3} value={note} onChange={e=>setNote(e.target.value)} placeholder={d?'为什么撤下':'依据哪些证据决定上线'}/></label>
   <label>写入令牌<input aria-label="写入令牌" type="password" value={token} onChange={e=>setToken(e.target.value)} autoComplete="off"/></label>
   <ErrorBox text={error}/>
   <div className="aw-toolbar">{d?<button disabled={busy||!note.trim()} onClick={()=>void send('/withdraw',{})}>确认撤下</button>
    :<button className="pf-primary" disabled={busy||!note.trim()||!chosen} onClick={()=>void send('',{season:chosen})}>确认上线</button>}</div>
  </div>}
 </div>
}
