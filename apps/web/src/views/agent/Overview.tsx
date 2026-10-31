import React from 'react'
import {Link} from 'react-router-dom'
import {Boxes, Cpu, FlaskConical, ListChecks, Trophy, type LucideIcon} from 'lucide-react'
import {trainingUrl} from '../TrainingWorkspace'
import {JobStatus} from './Jobs'
import {type Obj, useV1, useRuns, enc, fmt, when, route, ErrorBox, STATES} from './shared'

const ACTIVE=['queued','preparing','running']

function Kpi({icon:Icon,label,value,hint,to}:{icon:LucideIcon;label:string;value:React.ReactNode;hint?:React.ReactNode;to:string}){
 return <Link className="pf-kpi" to={to}><span className="pf-kpi-label"><Icon aria-hidden="true"/>{label}</span><b>{value??'…'}</b><small>{hint??' '}</small></Link>
}

function Panel({title,to,more,children}:{title:string;to:string;more:string;children:React.ReactNode}){
 return <section className="aw-panel pf-card"><header><h2>{title}</h2><Link to={to}>{more} →</Link></header>{children}</section>
}

export function Overview({season,roster}:{season:string;roster:Obj|null}){
 const models=useV1('/models'),experiments=useV1('/experiments'),runs=useRuns(),jobs=useV1('/jobs'),system=useV1('/jobs/system')
 const top=roster?[...roster.agents].filter((a:Obj)=>a.ranking).sort((a:Obj,b:Obj)=>a.ranking.rank-b.ranking.rank).slice(0,6):[]
 const elos=top.map((a:Obj)=>a.ranking.elo),low=Math.min(...elos)-100,span=Math.max(1,Math.max(...elos)-low)
 const recent=[...(runs.data?.runs??[])].sort((a:Obj,b:Obj)=>String(b.modified_at??b.created_at??'').localeCompare(String(a.modified_at??a.created_at??''))).slice(0,5)
 const jobRows:Obj[]=jobs.data?.jobs??[],active=jobRows.filter(j=>ACTIVE.includes(j.status))
 const summary=models.data?.summary,gpu=system.data?.gpu?.gpus?.[0]
 return <>
 <div className="pf-hero"><div><h1>总览</h1><p>训练、诊断与对战的统一入口</p></div>
  <div className="pf-actions"><Link className="pf-primary" to="/jobs/new">新建任务</Link>{season&&<Link className="button-link" to={route('/play',season)}>开始对战</Link>}<Link className="button-link" to="/lab">打开实验室</Link></div></div>
 <div className="pf-kpis">
  <Kpi icon={Boxes} label="已登记模型" value={summary?.total} to="/models" hint={summary&&`${summary.analyzable} 可分析 · ${summary.playable} 可对战`}/>
  <Kpi icon={Trophy} label="已上天梯" value={summary?.ranked} to="/models?state=ranked" hint={top[0]&&`榜首 ${top[0].name} · ${fmt(top[0].ranking.elo,0)}`}/>
  <Kpi icon={FlaskConical} label="实验" value={experiments.data?.experiments.length} to="/experiments" hint={experiments.data?.experiments[0]?.id}/>
  <Kpi icon={ListChecks} label="进行中任务" value={jobs.error?'—':jobs.data&&active.length} to="/jobs" hint={jobs.error?'任务服务不可用':jobs.data&&`${jobs.data.counts.succeeded??0} 成功 · ${jobs.data.counts.failed??0} 失败`}/>
  <Kpi icon={Cpu} label="GPU" value={system.error?'—':system.data&&(gpu?`${gpu.utilization}%`:'不可用')} to="/jobs" hint={gpu&&`${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB`}/>
 </div>
 {summary&&<nav className="pf-capabilities" aria-label="模型能力">{STATES.map(([k,label])=><Link key={k} to={'/models?state='+k}><b>{summary[k]}</b>{label}</Link>)}</nav>}
 <ErrorBox text={models.error} retry={models.retry}/>
 <div className="pf-overview">
  <Panel title="天梯" to={season?route('/ladder',season):'/ladder'} more="完整天梯">{roster?<><small className="pf-sub">{season} · {roster.protocol}</small><ol className="pf-ranks">{top.map((a:Obj)=><li key={a.id}><span className="pf-rank">{a.ranking.rank}</span><Link to={route('/agents/'+enc(a.id),season)}>{a.name}</Link><span className="pf-bar" aria-hidden="true"><i style={{width:`${(100*(a.ranking.elo-low)/span).toFixed(1)}%`}}/></span><b>{fmt(a.ranking.elo,0)}</b></li>)}</ol></>:<p role="status">加载评测批次…</p>}</Panel>
  <Panel title="最近任务" to="/jobs" more="全部任务"><ErrorBox text={jobs.error}/>{!jobs.data&&!jobs.error&&<p role="status">加载任务…</p>}{jobs.data&&!jobRows.length&&<p className="pf-empty">还没有任务。<Link to="/jobs/new">新建一个</Link></p>}<ul className="pf-list">{jobRows.slice(0,5).map(j=><li key={j.id}><div><Link to={'/jobs/'+enc(j.id)}>{j.name??j.id}</Link><small>{j.entrypoint} · {when(j.created)}</small></div><JobStatus status={j.status}/></li>)}</ul></Panel>
  <Panel title="最近实验" to="/experiments" more="全部实验"><ErrorBox text={experiments.error} retry={experiments.retry}/>{!experiments.data&&!experiments.error&&<p role="status">加载实验…</p>}<ul className="pf-list">{(experiments.data?.experiments??[]).slice(0,5).map((e:Obj)=><li key={e.id}><div><Link to={'/experiments/'+enc(e.id)}>{e.id}</Link><small>{e.arms.length} 个 arm · {e.reports.length?'有报告':'无报告'}</small></div><small>{when(e.modified)}</small></li>)}</ul></Panel>
  <Panel title="最近训练" to="/training" more="训练记录"><ErrorBox text={runs.error} retry={runs.retry}/>{!runs.data&&!runs.error&&<p role="status">加载训练记录…</p>}<ul className="pf-list">{recent.map((r:Obj)=><li key={r.run_id}><div><Link to={trainingUrl(season,[r.run_id])}>{r.run_name??r.run_id}</Link><small>{[r.algorithm,r.source,r.status].filter(Boolean).join(' · ')}</small></div><small>{when(r.modified_at??r.created_at)}</small></li>)}</ul></Panel>
 </div></>
}
