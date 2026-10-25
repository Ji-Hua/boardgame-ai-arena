import React,{useEffect,useMemo,useState} from 'react'
import {Link,useSearchParams} from 'react-router-dom'
import {api} from '../api'
import type {RunDetail,RunSummary,MetricSeries,AxisKey} from '../types'
import {availableAxes,chartData,comparisonGroups,display,COLORS} from '../runData'
import {LineChart} from '../components/LineChart'
import {EvidenceViewer} from './EvidenceViewer'
import './TrainingWorkspace.css'

const enc=encodeURIComponent
export function trainingUrl(season:string,ids:string[]){const p=new URLSearchParams({season});ids.forEach(id=>p.append('run',id));return '/training?'+p}
export function flattenConfig(value:unknown,prefix=''):Record<string,string>{
 if(value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).length)return Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b)).flatMap(([k,v])=>Object.entries(flattenConfig(v,prefix?prefix+'.'+k:k))))
 return {[prefix]:value===undefined?'未记录':JSON.stringify(value)}
}
type Loaded={detail?:RunDetail;metrics?:MetricSeries;errors:string[]}
export function RunInspector({ids,season,sha}:{ids:string[];season:string;sha?:string}){
 const key=ids.join('\0');const [loaded,setLoaded]=useState<Record<string,Loaded>>({}),[loading,setLoading]=useState(false),[version,setVersion]=useState(0)
 const [tab,setTab]=useState('curves'),[curve,setCurve]=useState(''),[axis,setAxis]=useState(''),[diffOnly,setDiffOnly]=useState(true),[filter,setFilter]=useState('')
 useEffect(()=>{let active=true;setLoaded({});setLoading(true);const selected=key?key.split('\0'):[]
 void Promise.all(selected.map(async id=>{const [d,m]=await Promise.allSettled([api.getRun(id),api.getMetrics(id)]);if(!active)return;setLoaded(prev=>({...prev,[id]:{detail:d.status==='fulfilled'?d.value:undefined,metrics:m.status==='fulfilled'?m.value:undefined,errors:[d.status==='rejected'?'记录: '+String(d.reason):'',m.status==='rejected'?'曲线: '+String(m.reason):''].filter(Boolean)}}))})).then(()=>{if(active)setLoading(false)})
 return()=>{active=false}},[key,version])
 const name=(id:string)=>loaded[id]?.detail?.run_name??id
 const entries=ids.flatMap(id=>loaded[id]?.metrics?[{runId:id,name:name(id),series:loaded[id].metrics!}]:[])
 const groups=useMemo(()=>comparisonGroups(entries),[loaded,key])
 const axes=ids.length>1?[...new Set(groups.map(([,g])=>g.axisType))]:entries[0]?availableAxes(entries[0].series).map(a=>a.key):[]
 const chosenAxis=axes.includes(axis)?axis:axes[0]??''
 const choices=ids.length>1?groups.filter(([,g])=>g.axisType===chosenAxis).map(([key,g])=>({key,label:g.label,unit:g.unit,scope:g.scope,curves:g.curves})):entries[0]?entries[0].series.columns.filter(k=>chartData(entries[0].series.points,chosenAxis as AxisKey,k).length).map(k=>({key:k,label:entries[0].series.metadata?.[k]?.label??k,unit:entries[0].series.metadata?.[k]?.unit??'单位未记录',scope:entries[0].series.metadata?.[k]?.scope??'范围未记录',curves:[{runId:ids[0],label:name(ids[0]),points:entries[0].series.points,yKey:k,xKey:chosenAxis as AxisKey}]})):[]
 const selected=choices.find(c=>c.key===curve)??choices.find(c=>c.curves.some(v=>/cross_entropy|loss/.test(v.yKey)))??choices[0]
 const configs=Object.fromEntries(ids.map(id=>{const d=loaded[id]?.detail;return [id,d?flattenConfig({source:d.source,algorithm:d.algorithm,model_arch:d.model_arch,reward_mode:d.reward_mode,config:d.config}):{}]}))
 const fields=[...new Set(ids.flatMap(id=>Object.keys(configs[id])))].sort().filter(k=>k.toLowerCase().includes(filter.toLowerCase())).filter(k=>!diffOnly||ids.length===1||new Set(ids.map(id=>configs[id][k]??'未记录')).size>1)
 if(!ids.length)return <div className="aw-panel aw-training-empty">选择训练 Run</div>
 return <section className="aw-panel aw-run-inspector"><div className="aw-toolbar"><h2>{ids.length===1?name(ids[0]):`${ids.length} Runs 对比`}</h2><button onClick={()=>setVersion(v=>v+1)} disabled={loading}>{loading?'加载中…':'刷新'}</button></div>
 <div className="aw-run-legend">{ids.map((id,i)=><div key={id}><span style={{color:COLORS[i]}}>●</span><strong>{name(id)}</strong><small>{loaded[id]?.detail?.source} · {loaded[id]?.detail?.status}</small><small className="aw-hash">Run {id}</small>{loaded[id]?.detail?.source==='mlflow'&&loaded[id]?.detail?.experiment_id&&<a target="_blank" rel="noreferrer" href={'http://192.168.68.88:8742/#/experiments/'+enc(loaded[id].detail!.experiment_id!)+'/runs/'+enc(id)}>MLflow ↗</a>}{loaded[id]?.errors.map(e=><span role="alert" className="error" key={e}>{e}</span>)}</div>)}</div>
 <div className="aw-tabs" role="tablist" aria-label="训练视图">{[['curves','训练指标'],['config','配置对比'],['checkpoints','Checkpoints'],['evidence','证据']].map(([id,label])=><button key={id} role="tab" aria-selected={tab===id} onClick={()=>setTab(id)}>{label}</button>)}</div>
 <div className={'aw-run-body '+(tab==='evidence'?'aw-evidence-body':'')}>
 {tab==='curves'&&<>{selected?<><div className="aw-toolbar"><label>指标<select aria-label="对比指标" value={selected.key} onChange={e=>setCurve(e.target.value)}>{choices.map(c=><option key={c.key} value={c.key}>{c.label} · {c.scope}</option>)}</select></label><label>横轴<select aria-label="对比横轴" value={chosenAxis} onChange={e=>setAxis(e.target.value)}>{axes.map(a=><option key={a} value={a}>{a}</option>)}</select></label><small>{selected.unit} · {selected.scope}</small></div>
 <div className="chart-card aw-training-chart"><LineChart series={selected.curves.map(c=>({...c,color:COLORS[ids.indexOf(c.runId)]}))} xKey={selected.curves[0].xKey} xLabel={chosenAxis} yLabel={selected.label+' ('+selected.unit+')'} width={900} height={340}/><div className="aw-toolbar">{selected.curves.map(c=><span key={c.runId+c.yKey} style={{color:COLORS[ids.indexOf(c.runId)]}}>● {c.label} <small>{c.points.length} 点{(loaded[c.runId]?.metrics?.metadata?.[c.yKey] as any)?.origin?.source==='local_training_files'?' · 历史训练文件':''}</small></span>)}</div></div>
 {ids.some(id=>!selected.curves.some(c=>c.runId===id))&&<small role="status">未叠加：{ids.filter(id=>!selected.curves.some(c=>c.runId===id)).map(name).join('、')}（指标、范围或横轴不兼容）</small>}
 </>:<p role="status">{loading?'加载曲线…':ids.length>1?'暂无语义、单位、范围和横轴一致的曲线':'暂无已记录曲线'}</p>}
 {entries.some(e=>e.series.warnings?.length)&&<details><summary>指标来源提示</summary>{entries.flatMap(e=>e.series.warnings.map((w,i)=><p key={e.runId+i}>{e.name} · {w}</p>))}</details>}
 {entries.some(e=>e.series.snapshots?.length)&&<details><summary>验证 / checkpoint 快照</summary>{entries.map(e=><details key={e.runId}><summary>{e.name} · {e.series.snapshots?.length??0}</summary><pre className="aw-json">{display(e.series.snapshots)}</pre></details>)}</details>}
 </>}
 {tab==='config'&&<><div className="aw-toolbar"><input aria-label="筛选配置字段" placeholder="字段" value={filter} onChange={e=>setFilter(e.target.value)}/>{ids.length>1&&<label><input type="checkbox" checked={diffOnly} onChange={e=>setDiffOnly(e.target.checked)}/>仅差异</label>}<small>{fields.length} 个字段</small></div><div className="aw-table aw-config-table"><table><thead><tr><th>字段</th>{ids.map(id=><th key={id}>{name(id)}</th>)}</tr></thead><tbody>{fields.map(field=>{const different=new Set(ids.map(id=>configs[id][field]??'未记录')).size>1;return <tr key={field} className={different?'aw-selected':''}><th>{field}</th>{ids.map(id=><td key={id}><pre>{loaded[id]?.detail?(configs[id][field]??'未记录'):loading?'加载中…':'不可用'}</pre></td>)}</tr>})}</tbody></table></div>{!fields.length&&!loading&&<p>没有匹配的配置差异</p>}</>}
 {tab==='checkpoints'&&<div className="aw-table"><table><thead><tr><th>Run / Checkpoint</th><th>进度</th><th>SHA</th><th></th></tr></thead><tbody>{ids.flatMap(id=>(loaded[id]?.detail?.checkpoints??[]).map((c,i)=><tr key={id+':'+i} className={sha===c.sha256?'aw-selected':''}><td>{name(id)}<small>{c.filename}{sha===c.sha256?' · 当前 Agent':''}</small></td><td>{c.progress_type??'进度类型未记录'} {c.step??c.episode??'—'}</td><td><span className="aw-hash" title={c.sha256??''}>{c.sha256?.slice(0,12)??'SHA 未记录'}</span></td><td>{c.sha256?<Link to={'/checkpoints/'+enc(c.sha256)+'?'+new URLSearchParams({season,run:id})}>评测 / 回放 / 对战</Link>:<span>无法确切关联</span>}</td></tr>))}</tbody></table>{!loading&&ids.every(id=>!loaded[id]?.detail?.checkpoints.length)&&<p>暂无 checkpoint 记录</p>}</div>}
 {tab==='evidence'&&<EvidenceViewer value={Object.fromEntries(ids.map(id=>[id,loaded[id]??{status:'加载中'}]))} filename="training-evidence.json"/>}
 </div></section>
}

export function TrainingWorkspace({season}:{season:string}){
 const [params,setParams]=useSearchParams();const ids=[...new Set(params.getAll('run').filter(Boolean))].slice(0,5)
 const [runs,setRuns]=useState<RunSummary[]>([]),[error,setError]=useState(''),[warnings,setWarnings]=useState<string[]>([]),[q,setQ]=useState(''),[source,setSource]=useState('all'),[version,setVersion]=useState(0),[loading,setLoading]=useState(true)
 useEffect(()=>{let active=true;setLoading(true);setError('');api.listRuns().then(d=>{if(active){setRuns(d.runs);setWarnings(d.warnings??[])}}).catch(e=>{if(active)setError(String(e))}).finally(()=>{if(active)setLoading(false)});return()=>{active=false}},[version])
 function select(next:string[]){const p=new URLSearchParams(params);p.delete('run');next.forEach(id=>p.append('run',id));setParams(p)}
 const filtered=runs.filter(r=>(source==='all'||r.source===source)&&[r.run_id,r.run_name,r.algorithm].join(' ').toLowerCase().includes(q.toLowerCase()))
 return <><div className="aw-breadcrumb"><Link to={'/ladder?season='+enc(season)}>天梯</Link> / 训练记录</div><div className="aw-toolbar"><h1>训练记录</h1><span>{ids.length} / 5 Runs</span><button disabled={!ids.length} onClick={()=>select([])}>清空选择</button></div>
 <div className="aw-training-workspace"><aside className="aw-panel aw-run-selector"><div className="aw-toolbar"><input aria-label="搜索训练" placeholder="Run / 名称 / 算法" value={q} onChange={e=>setQ(e.target.value)}/><select aria-label="训练来源" value={source} onChange={e=>setSource(e.target.value)}><option value="all">全部来源</option>{[...new Set(runs.map(r=>r.source))].map(s=><option key={s} value={s}>{s}</option>)}</select><button onClick={()=>setVersion(v=>v+1)} disabled={loading}>刷新列表</button></div>{error&&<div role="alert" className="error">{error}</div>}{loading&&<small role="status">加载 Run…</small>}
 <div className="aw-run-list">{filtered.map(r=><div className={'aw-run-choice '+(ids.includes(r.run_id)?'aw-selected':'')} key={r.run_id}><input type="checkbox" aria-label={'选择 '+(r.run_name??r.run_id)} checked={ids.includes(r.run_id)} disabled={!ids.includes(r.run_id)&&ids.length>=5} onChange={()=>select(ids.includes(r.run_id)?ids.filter(id=>id!==r.run_id):[...ids,r.run_id])}/><button title={r.run_id} onClick={()=>select([r.run_id])}><strong>{r.run_name??r.run_id}</strong><small>{r.source} · {r.algorithm} · {r.status??'—'}</small></button></div>)}</div>{!loading&&!filtered.length&&<small>没有匹配的 Run</small>}{warnings.length>0&&<details><summary>来源提示 ({warnings.length})</summary>{warnings.map(w=><p key={w}>{w}</p>)}</details>}</aside>
 <RunInspector ids={ids} season={season}/></div></>
}
