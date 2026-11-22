import {Link, useParams, useSearchParams} from 'react-router-dom'
import {type Obj, useV1, enc, fmt, when, ErrorBox, Json} from './shared'
import {modelName, semantics} from './Models'
import {JobStatus} from './Jobs'

export function ExperimentList(){
 const r=useV1('/experiments'),jobs=useV1('/jobs')
 // Experiments created by Jobs need no directory; they are listed by their slug.
 const named=new Set((r.data?.experiments??[]).map((e:Obj)=>e.id))
 const fromJobs=[...new Set<string>((jobs.data?.jobs??[]).map((j:Obj)=>j.experiment).filter((x:string|null)=>x&&!named.has(x)))].map(id=>({id,modified:Math.max(...jobs.data!.jobs.filter((j:Obj)=>j.experiment===id).map((j:Obj)=>j.created)),arms:[],reports:[],has_assessment:false,jobs_only:true}))
 const rows:Obj[]=[...(r.data?.experiments??[]),...fromJobs].sort((a,b)=>b.modified-a.modified)
 const count=(id:string)=>(jobs.data?.jobs??[]).filter((j:Obj)=>j.experiment===id).length
 return <><div className="aw-toolbar"><h1>实验</h1><small>只读外部实验目录 · 报告与评估按原始记录展示</small></div><ErrorBox text={r.error} retry={r.retry}/>{!r.data&&!r.error&&<p role="status">加载实验…</p>}
 {r.data&&<div className="aw-table"><table><thead><tr><th>实验</th><th>更新时间</th><th>训练 arm</th><th>平台任务</th><th>报告</th><th>评估</th></tr></thead><tbody>{rows.map(e=><tr key={e.id}><td><Link to={'/experiments/'+enc(e.id)}>{e.id}</Link>{e.jobs_only&&<small>仅平台任务</small>}</td><td>{when(e.modified)}</td><td>{e.arms.length}</td><td>{count(e.id)||'—'}</td><td>{e.reports.length?e.reports.length+' 份':'—'}</td><td>{e.has_assessment?'有':'—'}</td></tr>)}</tbody></table></div>}
 {r.data&&!rows.length&&<p>暂无实验目录</p>}</>
}

export function ExperimentDetail({season}:{season:string}){
 const {slug=''}=useParams();const [params,setParams]=useSearchParams();const r=useV1('/experiments/'+enc(slug));const jobs=useV1('/jobs?experiment='+enc(slug))
 const runs:Obj[]=jobs.data?.jobs??[]
 // A Job-only experiment has no directory; show its Jobs instead of a missing-directory error.
 const missing=Boolean(r.error)&&runs.length>0
 const e=r.data??(missing?{id:slug,modified:Math.max(...runs.map(j=>j.created)),arms:[],reports:[],models:[],files:[],assessment:null,warnings:['此实验只有平台任务，没有外部实验目录']}:null)
 const tab=['reports','models','jobs','assessment','files'].includes(params.get('tab')??'')?params.get('tab')!:e&&!e.reports.length?(e.models.length?'models':'jobs'):'reports'
 const report=e?.reports.find((x:Obj)=>x.path===params.get('report'))??e?.reports[0]
 function set(values:Record<string,string>){const p=new URLSearchParams(params);Object.entries(values).forEach(([k,v])=>p.set(k,v));setParams(p,{replace:true})}
 const arms=e?[...new Set<string>(e.models.flatMap((m:Obj)=>m.arms))].sort():[]
 return <><div className="aw-breadcrumb"><Link to="/experiments">实验</Link> / {slug}</div>{!missing&&<ErrorBox text={r.error} retry={r.retry}/>}{!e&&!r.error&&<p role="status">加载实验记录…</p>}
 {e&&<><div className="aw-toolbar"><h1>{e.id}</h1><small>更新 {when(e.modified)}</small></div>
 <div className="aw-toolbar aw-metrics"><span>{e.arms.length} 个训练 arm</span><span>{e.models.length} 个已登记模型</span><span>{e.models.filter((m:Obj)=>m.ranked).length} 个已上天梯</span><span>{runs.length} 个平台任务</span><Link className="button-link" to={'/jobs/new?experiment='+enc(slug)}>为此实验新建任务</Link></div>
 {e.warnings.map((w:string)=><p key={w} className="warning">{w}</p>)}
 <div className="aw-tabs" role="tablist" aria-label="实验视图">{[['reports',`报告 · ${e.reports.length}`],['models',`模型 · ${e.models.length}`],['jobs',`任务 · ${runs.length}`],['assessment','评估'],['files',`文件 · ${e.files.length}`]].map(([id,label])=><button key={id} role="tab" aria-selected={tab===id} onClick={()=>set({tab:id})}>{label}</button>)}</div>
 <div className="pf-tab-body" role="tabpanel">
 {tab==='reports'&&(report?<>{e.reports.length>1&&<select aria-label="报告" value={report.path} onChange={x=>set({report:x.target.value})}>{e.reports.map((x:Obj)=><option key={x.path} value={x.path}>{x.path}</option>)}</select>}{report.truncated&&<p className="warning">报告超过显示上限，已截断</p>}<pre className="pf-report">{report.text}</pre></>:<p>此实验目录没有 Markdown 报告</p>)}
 {tab==='models'&&(e.models.length?arms.map(arm=><section key={arm} className="pf-arm"><h2>{arm.startsWith('job:')?<>平台任务 <Link to={'/jobs/'+enc(arm.slice(4))}>{arm.slice(4)}</Link></>:arm||'实验根目录'}</h2><div className="aw-table"><table><thead><tr><th>模型</th><th>架构 / 输出</th><th>天梯</th></tr></thead><tbody>{e.models.filter((m:Obj)=>m.arms.includes(arm)).map((m:Obj)=><tr key={m.sha256}><td><Link to={'/models/'+m.sha256+'?season='+enc(season)}>{modelName(m)}</Link><small>{m.sha256.slice(0,12)}</small></td><td>{m.architecture??'未记录'}<small>{semantics(m.output_semantics)}</small></td><td>{m.seasons.length?m.seasons.map((s:Obj)=>`${s.season} · Elo ${fmt(s.elo,0)}`).join('；'):'—'}</td></tr>)}</tbody></table></div></section>):<p>没有按 catalog 来源路径确切关联到此实验的模型</p>)}
 {tab==='jobs'&&(runs.length?<>{runs.length>=2&&<div className="aw-toolbar"><Link className="button-link" to={'/jobs/compare?ids='+runs.slice(0,4).map(j=>enc(j.id)).join(',')}>对比{runs.length>4?'最近 4 个':'这些'}任务的曲线</Link></div>}<div className="aw-table"><table><thead><tr><th>任务</th><th>入口</th><th>状态</th><th>创建</th><th>注册模型</th></tr></thead><tbody>{runs.map(j=><tr key={j.id}><td><Link to={'/jobs/'+enc(j.id)}>{j.name}</Link><small>{j.id}</small></td><td>{j.entrypoint}<small>{j.spec.source.commit.slice(0,8)}</small></td><td><JobStatus status={j.status}/></td><td>{when(j.created)}</td><td>{j.registered.length||'—'}</td></tr>)}</tbody></table></div></>:<p>还没有属于此实验的平台任务</p>)}
 {tab==='assessment'&&(e.assessment?<Json value={e.assessment}/>:<p>没有 assessment.json</p>)}
 {tab==='files'&&<div className="aw-table"><table><thead><tr><th>名称</th><th>大小</th></tr></thead><tbody>{e.files.map((f:Obj)=><tr key={f.name}><td>{f.name}{f.directory?'/':''}</td><td>{f.directory?'目录':f.size+' B'}</td></tr>)}</tbody></table></div>}
 </div></>}</>
}
