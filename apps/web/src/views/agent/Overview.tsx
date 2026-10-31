import {Link} from 'react-router-dom'
import {trainingUrl} from '../TrainingWorkspace'
import {type Obj, useV1, useRuns, enc, fmt, when, route, ErrorBox, STATES} from './shared'

export function Overview({season,roster}:{season:string;roster:Obj|null}){
 const models=useV1('/models'),experiments=useV1('/experiments'),runs=useRuns()
 const top=roster?[...roster.agents].filter((a:Obj)=>a.ranking).sort((a:Obj,b:Obj)=>a.ranking.rank-b.ranking.rank).slice(0,5):[]
 const recent=[...(runs.data?.runs??[])].sort((a:Obj,b:Obj)=>String(b.modified_at??b.created_at??'').localeCompare(String(a.modified_at??a.created_at??''))).slice(0,6)
 return <><div className="aw-toolbar"><h1>总览</h1><small>训练、诊断与对战的统一入口</small></div>
 <div className="pf-overview">
 <section className="aw-panel"><div className="aw-toolbar"><h2>模型</h2><Link to="/models">模型库 →</Link></div><ErrorBox text={models.error} retry={models.retry}/>{!models.data&&!models.error&&<p role="status">加载模型索引…</p>}
 {models.data&&<div className="pf-stats"><Link to="/models"><b>{models.data.summary.total}</b>已登记</Link>{STATES.map(([k,label])=><Link key={k} to={'/models?state='+k}><b>{models.data!.summary[k]}</b>{label}</Link>)}</div>}</section>
 <section className="aw-panel"><div className="aw-toolbar"><h2>天梯</h2>{season&&<Link to={route('/ladder',season)}>完整天梯 →</Link>}</div>{roster?<><small>{season} · {roster.protocol}</small><div className="aw-table"><table><tbody>{top.map((a:Obj)=><tr key={a.id}><td>{a.ranking.rank}</td><td><Link to={route('/agents/'+enc(a.id),season)}>{a.name}</Link></td><td>Elo {fmt(a.ranking.elo,0)}</td></tr>)}</tbody></table></div></>:<p role="status">加载评测批次…</p>}</section>
 <section className="aw-panel"><div className="aw-toolbar"><h2>最近训练</h2><Link to="/training">训练记录 →</Link></div><ErrorBox text={runs.error} retry={runs.retry}/>{!runs.data&&!runs.error&&<p role="status">加载训练记录…</p>}<ul className="pf-list">{recent.map((r:Obj)=><li key={r.run_id}><Link to={trainingUrl(season,[r.run_id])}>{r.run_name??r.run_id}</Link><small>{r.source} · {r.algorithm} · {r.status??'—'} · {r.modified_at??r.created_at??'时间未记录'}</small></li>)}</ul></section>
 <section className="aw-panel"><div className="aw-toolbar"><h2>最近实验</h2><Link to="/experiments">全部实验 →</Link></div><ErrorBox text={experiments.error} retry={experiments.retry}/>{!experiments.data&&!experiments.error&&<p role="status">加载实验…</p>}<ul className="pf-list">{(experiments.data?.experiments??[]).slice(0,5).map((e:Obj)=><li key={e.id}><Link to={'/experiments/'+enc(e.id)}>{e.id}</Link><small>{e.arms.length} 个 arm · {e.reports.length?'有报告':'无报告'} · {when(e.modified)}</small></li>)}</ul></section>
 </div>
 <div className="aw-toolbar pf-quick"><Link className="button-link" to="/lab">打开实验室</Link>{season&&<Link className="button-link" to={route('/play',season)}>开始对战</Link>}</div></>
}
