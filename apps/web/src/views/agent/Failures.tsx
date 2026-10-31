import {useState} from 'react'
import {Link} from 'react-router-dom'
import {type Obj, useData, evidence, enc, route, notation, ErrorBox} from './shared'

const KINDS:[string,string][]=[['missed_win','错过立即获胜'],['loss','负'],['cycle','循环和棋'],['ply_limit','步数上限']]
const label=(k:string)=>KINDS.find(([id])=>id===k)?.[1]??k

/** Failure mining over one season's recorded games: recorded outcomes first, engine-located missed wins on demand. */
export function Failures({season,agent,roster}:{season:string;agent:Obj;roster:Obj}){
 const r=useData('/seasons/'+enc(season)+'/failures/'+enc(agent.id))
 const [kind,setKind]=useState('missed_win'),[opponent,setOpponent]=useState('all'),[page,setPage]=useState(0)
 const [located,setLocated]=useState<Record<string,Obj>>({}),[error,setError]=useState('')
 const name=(id:string)=>roster.agents.find((x:Obj)=>x.id===id)?.name??id
 const rows:Obj[]=(r.data?.failures??[]).filter((f:Obj)=>(kind==='all'||f.kinds.includes(kind))&&(opponent==='all'||f.opponent===opponent))
 async function locate(id:string){setError('');try{const d=await evidence('/seasons/'+enc(season)+'/replay/'+enc(id)+'/missed-wins?'+new URLSearchParams({agent:agent.id}));setLocated(v=>({...v,[id]:d}))}catch(e){setError(String(e))}}
 const replay=(f:Obj,ply=0)=>route('/games/'+enc(f.id),season,{agent:agent.id,opponent:f.opponent,ply})
 return <section className="aw-section"><ErrorBox text={r.error||error} retry={r.retry}/>{!r.data&&!r.error&&<p role="status">扫描已记录对局…</p>}
 {r.data&&<><div className="pf-chips" role="group" aria-label="失败类型"><button aria-pressed={kind==='all'} onClick={()=>{setKind('all');setPage(0)}}>全部 <b>{r.data.failures.length}</b></button>{KINDS.map(([k,l])=><button key={k} aria-pressed={kind===k} onClick={()=>{setKind(k);setPage(0)}}>{l} <b>{r.data!.totals[k]}</b></button>)}</div>
 <div className="aw-toolbar"><select aria-label="失败样本对手" value={opponent} onChange={e=>{setOpponent(e.target.value);setPage(0)}}><option value="all">全部对手</option>{[...new Set<string>(r.data.failures.map((f:Obj)=>f.opponent))].map(o=><option key={o} value={o}>{name(o)}</option>)}</select><small>{rows.length} 局 · 共 {r.data.games} 局已记录 · 分类来自记录的结果与每局决策诊断</small></div>
 <div className="aw-table"><table><thead><tr><th>对局 / 开局</th><th>对手 · 执子</th><th>类型</th><th>步数</th><th></th></tr></thead><tbody>{rows.slice(page*20,page*20+20).map(f=><tr key={f.id}><td>{f.id.slice(0,8)}:{f.id.split(':')[1]}<small>开局 {f.opening??'—'}{f.control?' · 标准开局对照':''}</small></td><td>{name(f.opponent)}<small>P{f.seat}</small></td><td>{f.kinds.map(label).join('、')}{f.missed_immediate_win>0&&<small>错过 {f.missed_immediate_win} 次</small>}</td><td>{f.steps}<small>{f.termination}</small></td>
 <td>{f.available?<><Link to={replay(f)}>回放</Link>{f.missed_immediate_win>0&&!located[f.id]&&<button onClick={()=>void locate(f.id)}>定位错失步</button>}{located[f.id]&&<div className="aw-moves">{located[f.id].plies.map((p:Obj)=><Link key={p.ply} to={replay(f,p.ply)} title={'实际 '+notation(p.actual)+' · 可直接获胜 '+p.winning.map(notation).join(' / ')}>第 {p.ply} 步 · 实际 {notation(p.actual)} · 可走 {p.winning.map(notation).join('/')}</Link>)}{!located[f.id].plies.length&&<small>复算未发现错失步</small>}</div>}</>:'无棋谱'}</td></tr>)}</tbody></table></div>
 {!rows.length&&<p>没有此类对局</p>}<div className="aw-toolbar"><button disabled={page===0} onClick={()=>setPage(p=>p-1)}>上一页</button><button disabled={(page+1)*20>=rows.length} onClick={()=>setPage(p=>p+1)}>下一页</button></div></>}</section>
}
