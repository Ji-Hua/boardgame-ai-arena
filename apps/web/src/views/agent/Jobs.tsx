import {useEffect, useRef, useState} from 'react'
import {Link, useNavigate, useParams, useSearchParams} from 'react-router-dom'
import {type Obj, request, useV1, enc, fmt, when, ErrorBox, Json} from './shared'
import {COLORS, EvalCurve, JobAlerts, JobCurves} from './JobCurves'
import {CheckpointTimeline} from './JobTimeline'

const ACTIVE=['queued','preparing','running']
const STATUS:Record<string,string>={queued:'排队中',preparing:'准备中',running:'运行中',succeeded:'已完成',failed:'失败',cancelled:'已取消'}
const TOKEN_KEY='quoridor-platform-token'
const readToken=()=>{try{return localStorage.getItem(TOKEN_KEY)??''}catch{return ''}}
const duration=(job:Obj)=>{if(!job.started)return '—';const s=Math.round((job.finished??Date.now()/1000)-job.started);return s<60?s+' 秒':s<3600?Math.round(s/60)+' 分钟':(s/3600).toFixed(1)+' 小时'}
/** Versioned probe suites as "name passed/count"; older diagnoses only recorded the unversioned tactical set. */
export function ProbeSummary({d}:{d:Obj}){
 return d.probe_suites?.length?<>{d.probe_suites.map((p:Obj)=><span key={p.name} className="pf-probe" title={`${p.name}/${p.version} · sha256 ${p.sha256}`}>{p.name} <b>{p.passed} / {p.count}</b></span>)}</>:<>{d.tactical_wins??'—'} / {d.tactical_count??'—'}<small>战术题（未版本化）</small></>
}
/** Per-checkpoint diagnosis rows from the CNN diagnosis (development games, probe suites). */
export function DiagnosisTable({rows,source=false}:{rows:Obj[];source?:boolean}){
 return <div className="aw-table"><table><thead><tr><th>{source?'来源':'Checkpoint'}</th><th>开发对局 胜 / 负</th><th>探针集</th><th>错过立即获胜</th><th>重复率</th><th>决策耗时</th></tr></thead><tbody>{rows.map((d:Obj,i:number)=>{const c=d.counts??d;return <tr key={(d.source??'')+(d.sha256??d.model_id)+i}><td>{source?<>{d.source}<small>{d.model_id}</small></>:d.sha256?<Link to={'/models/'+d.sha256}>{d.model_id}</Link>:d.model_id}<small>{d.output_semantics??'语义未记录'}{d.search?' · 搜索':''}</small></td><td>{c.wins??'—'} / {c.losses??'—'}<small>{d.games} 局 · {d.unique_openings??'—'} 个开局</small></td><td><ProbeSummary d={d}/></td><td>{c.missed_immediate_win??'—'} / {c.immediate_win_available??'—'}</td><td>{fmt(d.repetition_rate,3)}</td><td>{fmt(d.mean_decision_ms,2)} ms</td></tr>})}</tbody></table></div>
}
export function JobStatus({status}:{status:string}){return <span className={'pf-job-status '+status}>{STATUS[status]??status}</span>}

/** Polls while ``active`` and keeps the previous value visible between refreshes. */
function useLive<T=Obj>(url:string|null,active:boolean,ms=3000,text=false){
 const [data,setData]=useState<T|null>(null),[error,setError]=useState(''),[tick,setTick]=useState(0)
 useEffect(()=>{setData(null);setError('')},[url])
 useEffect(()=>{if(!url)return;const c=new AbortController()
  const load=text?fetch(url,{signal:c.signal}).then(r=>r.ok?r.text():Promise.reject(new Error(r.status+' '+r.statusText))):request(url,{signal:c.signal})
  load.then(d=>{if(!c.signal.aborted){setData(d as T);setError('')}}).catch(e=>{if(!c.signal.aborted)setError(String(e))});return()=>c.abort()},[url,tick])
 useEffect(()=>{if(!active||!url)return;const t=setInterval(()=>setTick(v=>v+1),ms);return()=>clearInterval(t)},[active,url,ms])
 return {data,error,refresh:()=>setTick(v=>v+1)}
}

/** Metrics over server-sent events while the Job runs; polling when EventSource is missing or the stream closes. */
function useJobMetrics(id:string,active:boolean){
 const [failed,setFailed]=useState(false),[live,setLive]=useState<Obj|null>(null)
 const streaming=active&&!failed&&typeof EventSource!=='undefined'
 const polled=useLive(streaming?null:'/api/v1/jobs/'+enc(id)+'/metrics',active,5000)
 useEffect(()=>{if(!streaming)return;const source=new EventSource('/api/v1/jobs/'+enc(id)+'/stream')
  // Append when the update continues the rows we hold; a reconnect starts over and replaces them.
  source.addEventListener('update',e=>{const d=JSON.parse((e as MessageEvent).data);setLive(prev=>prev&&d.offset===prev.offset+prev.metrics.length?{...d,offset:prev.offset,metrics:[...prev.metrics,...d.metrics]}:d)})
  source.addEventListener('end',()=>source.close())
  source.onerror=()=>{if(source.readyState===EventSource.CLOSED)setFailed(true)}
  return()=>source.close()},[id,streaming])
 return streaming?{data:live,error:'',live:true}:{data:polled.data??live,error:polled.error,live:false}
}

export function JobList(){
 const [projectParams]=useSearchParams();const projectId=projectParams.get('project')
 const jobs=useLive('/api/v1/jobs'+(projectId?'?project_id='+enc(projectId):''),true,5000),system=useV1('/jobs/system')
 const rows:Obj[]=jobs.data?.jobs??[];const gpu=system.data?.gpu?.gpus?.[0]
 const [picked,setPicked]=useState<string[]>([])
 const toggle=(id:string)=>setPicked(p=>p.includes(id)?p.filter(x=>x!==id):p.length<COLORS.length?[...p,id]:p)
 return <><div className="aw-toolbar"><h1>任务</h1><small>训练、评测在独立容器中运行，固定到已提交的源码版本</small>
  {picked.length>=2?<Link className="button-link" to={'/jobs/compare?ids='+picked.map(enc).join(',')}>对比所选（{picked.length}）</Link>:<small>勾选 2–{COLORS.length} 个任务可对比曲线</small>}
  <Link to="/projects">研究项目</Link>{projectId&&<small>项目：{projectId}</small>}<Link className="pf-primary" to={projectId?"/jobs/new?project="+enc(projectId):"/jobs/new"}>新建任务</Link></div>
 <div className="aw-toolbar aw-metrics"><span>{gpu?`${gpu.name} · 显存 ${gpu.memory_used_mib}/${gpu.memory_total_mib} MiB · 利用率 ${gpu.utilization}%`:system.data?'GPU 不可用':'读取 GPU…'}</span>{jobs.data&&Object.entries(jobs.data.counts as Record<string,number>).filter(([,n])=>n>0).map(([s,n])=><span key={s}>{STATUS[s]} {n}</span>)}{system.data&&!system.data.writes_enabled&&<span>未配置写入令牌</span>}</div>
 <ErrorBox text={jobs.error||system.error}/>{!jobs.data&&!jobs.error&&<p role="status">加载任务…</p>}
 {jobs.data&&<div className="aw-table"><table><thead><tr><th aria-label="选择"/><th>任务</th><th>入口</th><th>状态</th><th>创建</th><th>用时</th></tr></thead><tbody>{rows.map(j=><tr key={j.id}><td><input type="checkbox" aria-label={'选择 '+j.name} checked={picked.includes(j.id)} disabled={!picked.includes(j.id)&&picked.length>=COLORS.length} onChange={()=>toggle(j.id)}/></td><td><Link to={'/jobs/'+enc(j.id)}>{j.name}</Link><small>{j.id}{j.experiment?' · '+j.experiment:''}</small></td><td>{j.entrypoint}<small>{j.spec.resources.gpu?'GPU':'CPU'} · {j.spec.source.commit.slice(0,8)}</small></td><td><JobStatus status={j.status}/>{j.error&&<small>{j.error}</small>}</td><td>{when(j.created)}</td><td>{duration(j)}</td></tr>)}</tbody></table></div>}
 {jobs.data&&!rows.length&&<p>还没有任务</p>}</>
}

export function JobNew(){
 const [params]=useSearchParams();const navigate=useNavigate();const meta=useV1('/jobs/entrypoints'+(params.get('project')?'?project_id='+enc(params.get('project')!):''))
 const projectId=params.get('project')??'quoridor-legacy'
 const [entrypoint,setEntrypoint]=useState(params.get('entrypoint')??'train_dqn'),[template,setTemplate]=useState(''),[config,setConfig]=useState('')
 const [name,setName]=useState(''),[experiment,setExperiment]=useState(params.get('experiment')??''),[resources,setResources]=useState<Record<string,string>>({}),[source,setSource]=useState('')
 const [token,setToken]=useState(readToken),[error,setError]=useState(''),[busy,setBusy]=useState(false),[gate,setGate]=useState(false)
 const entry=meta.data?.entrypoints?.[entrypoint]
 useEffect(()=>{if(meta.data&&!meta.data.entrypoints?.[entrypoint])setEntrypoint(Object.keys(meta.data.entrypoints??{})[0]??'')},[meta.data,entrypoint])
 useEffect(()=>{if(!entry)return;setTemplate('');if(entry.format==='json'&&entry.template){const value={...entry.template};if(entrypoint==='evaluate_research_policy'&&params.get('policy'))value.candidate_sha=params.get('policy');if(params.get('sha')&&Array.isArray(value[entrypoint==='run_arena_matrix'?'candidates':'checkpoints']))value[entrypoint==='run_arena_matrix'?'candidates':'checkpoints']=[params.get('sha')];setConfig(JSON.stringify(value,null,2))}else setConfig('')},[entrypoint,meta.data])
 useEffect(()=>{if(!template)return;const c=new AbortController();fetch('/api/v1/jobs/templates/'+enc(entrypoint)+'/'+enc(template),{signal:c.signal}).then(r=>r.ok?r.text():Promise.reject(new Error('模板读取失败'))).then(setConfig).catch(e=>{if(!c.signal.aborted)setError(String(e))});return()=>c.abort()},[template])
 async function submit(){setBusy(true);setError('');try{try{localStorage.setItem(TOKEN_KEY,token)}catch{/* optional */}
  const body={project_id:projectId,entrypoint,config,name:name||undefined,experiment:experiment||undefined,source:source||undefined,follow_up:gate&&entry?.kind==='train'?{gate:'gate-v1'}:undefined,resources:Object.fromEntries(Object.entries(resources).filter(([,v])=>v!=='').map(([k,v])=>[k,Number(v)]))}
  const job=await request('/api/v1/jobs',{method:'POST',headers:{'Content-Type':'application/json','X-Platform-Token':token},body:JSON.stringify(body)});navigate('/jobs/'+enc(job.id))}catch(e){setError(String(e))}finally{setBusy(false)}}
 return <><div className="aw-breadcrumb"><Link to="/jobs">任务</Link> / 新建</div><div className="aw-toolbar"><h1>新建任务</h1></div><ErrorBox text={meta.error} retry={meta.retry}/>{!meta.data&&!meta.error&&<p role="status">读取入口与源码版本…</p>}
 {meta.data&&<div className="pf-form"><p>所属项目：<Link to="/projects">{projectId}</Link></p>
  <label>入口<select aria-label="任务入口" value={entrypoint} onChange={e=>setEntrypoint(e.target.value)}>{Object.entries(meta.data.entrypoints as Record<string,Obj>).map(([k,v])=><option key={k} value={k}>{v.title} · {k}{v.enabled===false?'（未配置执行环境）':''}</option>)}</select></label>
  {entry&&<small>{entry.description} · {entry.gpu?'GPU':'CPU'}</small>}{entry?.enabled===false&&<p role="alert">{entry.unavailable_reason}</p>}
  {entry?.templates&&<label>配置模板<select aria-label="配置模板" value={template} onChange={e=>setTemplate(e.target.value)}><option value="">选择模板</option>{entry.templates.map((t:string)=><option key={t} value={t}>{t}</option>)}</select></label>}
  <label>配置（{entry?.format==='json'?'JSON':'YAML'}）<textarea aria-label="任务配置" rows={18} value={config} onChange={e=>setConfig(e.target.value)} spellCheck={false}/></label>
  <div className="pf-form-row"><label>名称<input aria-label="任务名称" value={name} placeholder={entrypoint} onChange={e=>setName(e.target.value)}/></label><label>实验<input aria-label="所属实验" value={experiment} placeholder="可选" onChange={e=>setExperiment(e.target.value)}/></label><label>源码<input aria-label="源码版本" value={source} placeholder={meta.data.source_ref+' @ '+meta.data.source_commit.slice(0,8)} onChange={e=>setSource(e.target.value)}/></label></div>
  <div className="pf-form-row">{[['cpus','CPU'],['memory_gb','内存 GB'],['timeout_h','超时 小时']].map(([k,label])=><label key={k}>{label}<input aria-label={label} type="number" min={0} step="any" value={resources[k]??''} placeholder="默认" onChange={e=>setResources(r=>({...r,[k]:e.target.value}))}/></label>)}</div>
  {entry?.kind==='train'&&<label className="pf-check"><input type="checkbox" checked={gate} onChange={e=>setGate(e.target.checked)}/>成功后对最后一个 checkpoint 自动门槛评测（gate-v1，结果发布为天梯评测批次）</label>}
  <label>写入令牌<input aria-label="写入令牌" type="password" value={token} onChange={e=>setToken(e.target.value)} autoComplete="off"/></label>
  <small>执行镜像 {entry?.image??meta.data.image}；提交后源码固定为当时的完整 commit</small>
  <ErrorBox text={error}/><div className="aw-toolbar"><button disabled={busy||!config.trim()||!entry||entry.enabled===false} onClick={()=>void submit()}>{busy?'提交中…':'提交任务'}</button></div>
 </div>}</>
}

export function JobDetail(){
 const {id=''}=useParams();const [params,setParams]=useSearchParams()
 const [active,setActive]=useState(true)
 const job=useLive('/api/v1/jobs/'+enc(id),active,3000)
 const metrics=useJobMetrics(id,active)
 const tab=['curves','checkpoints','evaluation','diagnosis','log','events','spec'].includes(params.get('tab')??'')?params.get('tab')!:'curves'
 const log=useLive<string>(tab==='log'?'/api/v1/jobs/'+enc(id)+'/log?tail=500':null,active,3000,true)
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),logRef=useRef<HTMLPreElement>(null)
 const j=job.data
 useEffect(()=>{if(j)setActive(ACTIVE.includes(j.status))},[j?.status])
 useEffect(()=>{logRef.current?.scrollTo(0,logRef.current.scrollHeight)},[log.data])
 useEffect(()=>{if(j&&metrics.data?.status&&metrics.data.status!==j.status)job.refresh()},[metrics.data?.status])
 const rows:Obj[]=(metrics.data?.metrics??[]).filter((r:Obj)=>!r.type||r.type==='train')
 async function cancel(){setBusy(true);setError('');try{await request('/api/v1/jobs/'+enc(id)+'/cancel',{method:'POST',headers:{'X-Platform-Token':readToken()}});job.refresh()}catch(e){setError(String(e)+'（令牌在新建任务页设置）')}finally{setBusy(false)}}
 const setTab=(t:string)=>{const p=new URLSearchParams(params);p.set('tab',t);setParams(p,{replace:true})}
 return <><div className="aw-breadcrumb"><Link to="/jobs">任务</Link> / {id}</div><ErrorBox text={job.error}/>{!j&&!job.error&&<p role="status">加载任务…</p>}
 {j&&<><div className="aw-toolbar"><h1>{j.name}</h1><JobStatus status={j.status}/>{ACTIVE.includes(j.status)&&<button disabled={busy||j.cancel_requested} onClick={()=>void cancel()}>{j.cancel_requested?'正在取消…':'取消任务'}</button>}</div>
 <div className="aw-toolbar aw-metrics"><span>{j.entrypoint}</span><span>{j.spec.resources.gpu?'GPU':'CPU'} · {j.spec.resources.cpus} CPU · {j.spec.resources.memory_gb} GB</span><span>源码 {j.spec.source.commit.slice(0,10)}</span><span>用时 {duration(j)}</span>{j.exit_code!=null&&<span>退出码 {j.exit_code}</span>}</div>
 {j.error&&<p className="warning">{j.error}</p>}<ErrorBox text={error}/><JobAlerts alerts={metrics.data?.alerts??[]}/>
 <div className="aw-toolbar">{j.run_files.includes('metrics.jsonl')&&<Link to={'/training?run='+enc(j.run_id)}>在训练工作区查看</Link>}{j.registered.map((sha:string)=><Link key={sha} to={'/models/'+sha}>模型 {sha.slice(0,12)}</Link>)}{j.experiment&&<Link to={'/experiments/'+enc(j.experiment)}>实验 {j.experiment}</Link>}{j.mlflow&&<a href={`/mlflow/#/experiments/${enc(j.mlflow.experiment_id)}/runs/${enc(j.mlflow.run_id)}`}>MLflow 运行</a>}{j.season&&<Link to={'/ladder?season='+enc(j.season)}>天梯评测批次 {j.season}</Link>}{j.spec.follow_up?.gate&&<small>成功后门槛评测 {j.spec.follow_up.gate}</small>}</div>
 {j.research_artifacts?.length>0&&<section aria-label="研究策略产物"><h2>研究策略产物</h2><Link to={'/projects/'+enc(j.project_id)+'/policies'}>项目研究策略</Link>{j.research_artifacts.map((a:Obj)=><p key={a.sha256}>{a.file} · {a.output_semantics}<small>SHA-256 {a.sha256}</small></p>)}<small>成功后登记至项目研究策略目录。</small></section>}
 {j.research_evaluation&&<section aria-label="研究策略评测结果"><h2>研究策略评测</h2><p>{j.research_evaluation.protocol} · {j.research_evaluation.games} 局</p><p>配对平均收益 {fmt(j.research_evaluation.mean_return,4)} · 标准误 {fmt(j.research_evaluation.standard_error,4)} · exploitability {fmt(j.research_evaluation.exploitability,6)}</p><Json value={j.research_evaluation}/></section>}
 <div className="aw-tabs" role="tablist" aria-label="任务视图">{[['curves','训练曲线'],['checkpoints',`Checkpoint · ${j.checkpoints?.length??0}`],['evaluation',`评测 · ${metrics.data?.evaluations?.length??0}`],['diagnosis',`诊断 · ${j.diagnoses?.length??0}`],['log','日志'],['events',`事件 · ${j.events.length}`],['spec','规格']].map(([k,label])=><button key={k} role="tab" aria-selected={tab===k} onClick={()=>setTab(k)}>{label}</button>)}</div>
 <div className="pf-tab-body" role="tabpanel">
 {tab==='curves'&&(rows.length?<JobCurves rows={rows} source={metrics.data?.metric_source} live={metrics.live}/>:<><ErrorBox text={metrics.error}/><p>{active?'等待第一批训练指标…':'没有训练指标'}</p></>)}
 {tab==='checkpoints'&&<CheckpointTimeline job={j} evaluations={metrics.data?.evaluations??[]} replays={metrics.data?.replays??[]}/>}
 {tab==='evaluation'&&((metrics.data?.evaluations?.length??0)?<><EvalCurve rows={metrics.data!.evaluations}/><div className="aw-table"><table><thead><tr><th>对手</th><th>进度</th><th>局数</th><th>胜 / 平 / 负</th><th>胜率</th></tr></thead><tbody>{metrics.data!.evaluations.map((r:Obj,i:number)=><tr key={i}><td>{r.opponent??r.opponent_name??r.opponent_id??'—'}{r.checkpoint_sha&&<small>{String(r.checkpoint_sha).slice(0,12)}</small>}</td><td>{r.episode??'最终'}</td><td>{r.games??r.num_games??r.eval_games??'—'}</td><td>{r.wins??'—'} / {r.draws??'—'} / {r.losses??'—'}</td><td>{fmt(r.win_rate,3)}</td></tr>)}</tbody></table></div></>:<p>暂无评测记录</p>)}
 {tab==='diagnosis'&&((j.diagnoses?.length??0)?<DiagnosisTable rows={j.diagnoses}/>:<p>{active?'每个 checkpoint 保存后自动诊断…':'此任务没有 checkpoint 诊断'}</p>)}
 {tab==='log'&&<><ErrorBox text={log.error}/><pre ref={logRef} className="pf-log">{log.data??'读取日志…'}</pre></>}
 {tab==='events'&&<ul className="pf-list">{j.events.map((e:Obj,i:number)=><li key={i}><strong>{e.event}</strong><small>{when(e.time)} · {Object.entries(e).filter(([k])=>!['event','time'].includes(k)).map(([k,v])=>k+'='+(typeof v==='object'?JSON.stringify(v):v)).join(' · ')}</small></li>)}</ul>}
 {tab==='spec'&&<Json value={j.spec}/>}
 </div></>}</>
}
