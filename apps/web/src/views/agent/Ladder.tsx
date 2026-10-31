import {Link, useSearchParams} from 'react-router-dom'
import {type Obj, enc, fmt, route, Tag, agentVersion} from './shared'

export function Ladder({season,roster,seasons}:{season:string;roster:Obj;seasons:Obj[]}){
 const [params,setParams]=useSearchParams();const type=params.get('type')??'all';const q=params.get('q')??''
 const rows=[...roster.agents].sort((a,b)=>(a.ranking?.rank??Infinity)-(b.ranking?.rank??Infinity)).filter(a=>(type==='all'||a.kind===type)&&[a.name,a.id,a.sha256??''].join(' ').toLowerCase().includes(q.toLowerCase()))
 function update(k:string,v:string){const p=new URLSearchParams(params);p.set(k,v);setParams(p)}
 return <><div className="aw-toolbar"><h1>天梯榜</h1><label>评测批次<select aria-label="评测批次" value={season} onChange={e=>setParams({season:e.target.value})}>{seasons.map(s=><option key={s.id} value={s.id}>{s.id}{s.gate?` · 门槛 ${s.gate}`:''}</option>)}</select></label><Link to={route('/training',season)}>训练记录</Link></div><div className="aw-toolbar"><label>类型<select aria-label="Agent 类型" value={type} onChange={e=>update('type',e.target.value)}><option value="all">全部</option><option value="neural">神经网络</option><option value="heuristic">启发式</option><option value="random">随机</option></select></label><input aria-label="搜索 Agent" placeholder="名称 / SHA" value={q} onChange={e=>update('q',e.target.value)}/><small>{roster.protocol} · {rows.length} Agent</small></div>
 {roster.warning&&<details className="aw-note"><summary>排名适用范围</summary>{roster.warning}<p>{roster.rating}</p></details>}
 <div className="aw-table aw-ladder-table"><table><thead><tr><th>#</th><th>Agent / 版本</th><th>类型</th><th>Elo</th><th>胜 / 平 / 负</th><th></th></tr></thead><tbody>{rows.map(a=><tr key={a.id}><td>{a.ranking?.rank??'—'}</td><td><Link to={route('/agents/'+enc(a.id),season)}>{a.name}</Link><small>{agentVersion(a)}</small></td><td><Tag agent={a}/></td><td>{fmt(a.ranking?.elo,0)}</td><td>{a.ranking?`${a.ranking.wins} / ${a.ranking.draws} / ${a.ranking.losses}`:'—'}</td><td><Link to={route('/play',season,{p1:a.id,p2:'human'})}>对战</Link></td></tr>)}</tbody></table></div>{rows.length===0&&<p>没有匹配的 Agent</p>}</>
}
