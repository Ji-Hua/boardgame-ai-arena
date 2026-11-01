import {useEffect, useState} from 'react'
import {Link, useSearchParams} from 'react-router-dom'
import {LineChart} from '../../components/LineChart'
import type {AxisKey, MetricPoint} from '../../types'
import {type Obj, request, enc, fmt, ErrorBox} from './shared'
import {COLORS, curveKeys, numeric, series} from './JobCurves'
import {JobStatus} from './Jobs'

export const COMPARE_LIMIT=COLORS.length
const MAIN:[string,string][]=[['win_rate','胜率（最近窗口）'],['truncation_rate','截断率'],['repeat_rate','重复局面率'],['avg_episode_length','平均对局长度'],['loss','loss'],['avg_q_max','|Q| 最大']]
const opponentOf=(r:Obj)=>String(r.opponent_name??r.opponent??r.opponent_id??'—')
const train=(m:Obj)=>(m?.metrics??[]).filter((r:Obj)=>!r.type||r.type==='train')

/** Overlay the curves of two to four Jobs; each Job keeps one color across every chart and the summary table. */
export function JobCompare(){
 const [params,setParams]=useSearchParams()
 const ids=(params.get('ids')??'').split(',').filter(Boolean).slice(0,COMPARE_LIMIT)
 const [data,setData]=useState<Obj[]|null>(null),[error,setError]=useState('')
 const [smooth,setSmooth]=useState(0.6),[axis,setAxis]=useState<AxisKey>('episode'),[extra,setExtra]=useState(''),[opponent,setOpponent]=useState('')
 useEffect(()=>{const c=new AbortController();setData(null);setError('')
  Promise.all(ids.map(id=>Promise.all([request('/api/v1/jobs/'+enc(id),{signal:c.signal}),request('/api/v1/jobs/'+enc(id)+'/metrics',{signal:c.signal})]).then(([job,metrics])=>({job,metrics}))))
   .then(rows=>{if(!c.signal.aborted)setData(rows)}).catch(e=>{if(!c.signal.aborted)setError(String(e))})
  return()=>c.abort()},[ids.join(',')])
 const remove=(id:string)=>{const p=new URLSearchParams(params);p.set('ids',ids.filter(x=>x!==id).join(','));setParams(p,{replace:true})}
 const rows=(data??[]).map(d=>train(d.metrics))
 const chart=(key:string,title:string)=>{const present=(data??[]).map((d,i)=>({d,i})).filter(({i})=>rows[i].some((r:Obj)=>numeric(r[key])))
  if(!present.length)return null
  return <section key={key} className="pf-chart-panel" aria-label={title}><header><h3>{title}</h3></header>
   <LineChart series={present.map(({d,i})=>({points:series(rows[i],[key],smooth,false),yKey:key,label:d.job.name,color:COLORS[i]}))} xKey={axis} xLabel={axis} yLabel={title} width={560} height={200}/></section>}
 const others=[...new Set(rows.flatMap(curveKeys))].filter(k=>!MAIN.some(([m])=>m===k)).sort()
 const opponents=[...new Set((data??[]).flatMap(d=>(d.metrics.evaluations??[]).filter((r:Obj)=>numeric(r.episode)&&numeric(r.win_rate)).map(opponentOf)))]
 const chosenOpponent=opponents.includes(opponent)?opponent:opponents[0]
 const evalPoints=(d:Obj):MetricPoint[]=>(d.metrics.evaluations??[]).filter((r:Obj)=>opponentOf(r)===chosenOpponent&&numeric(r.episode)).map((r:Obj)=>({episode:r.episode,step:r.step??null,values:{win_rate:numeric(r.win_rate)?r.win_rate:null}}))
 const best=(d:Obj)=>(d.metrics.evaluations??[]).filter((r:Obj)=>numeric(r.win_rate)).reduce((a:Obj|null,r:Obj)=>!a||r.win_rate>a.win_rate?r:a,null)
 return <><div className="aw-breadcrumb"><Link to="/jobs">任务</Link> / 对比</div><div className="aw-toolbar"><h1>任务对比</h1><small>平滑与横轴对所有曲线生效；胜率、截断率等是各任务自己记录窗口内的比例</small></div>
  {ids.length<2&&<p>请在<Link to="/jobs">任务列表</Link>里选择 2–{COMPARE_LIMIT} 个任务</p>}
  <ErrorBox text={error}/>{ids.length>=2&&!data&&!error&&<p role="status">加载任务指标…</p>}
  {data&&ids.length>=2&&<>
   <div className="aw-table"><table><thead><tr><th>任务</th><th>入口</th><th>状态</th><th>episode</th><th>胜率</th><th>截断率</th><th>重复局面</th><th>loss</th><th>最佳评测胜率</th><th/></tr></thead><tbody>
    {data.map((d,i)=>{const last=rows[i][rows[i].length-1]??{},b=best(d);return <tr key={d.job.id}><td><span className="pf-swatch" style={{background:COLORS[i]}}/><Link to={'/jobs/'+enc(d.job.id)}>{d.job.name}</Link><small>{d.job.id}{d.job.experiment?' · '+d.job.experiment:''}</small></td><td>{d.job.entrypoint}<small>{d.job.spec.source.commit.slice(0,8)}</small></td><td><JobStatus status={d.job.status}/></td><td>{last.episode??'—'}</td>
     <td>{numeric(last.win_rate)?fmt(100*last.win_rate,0)+'%':'—'}</td><td>{numeric(last.truncation_rate)?fmt(100*last.truncation_rate,0)+'%':'—'}</td><td>{numeric(last.repeat_rate)?fmt(100*last.repeat_rate,0)+'%':'—'}</td><td>{fmt(last.loss,4)}</td>
     <td>{b?<>{fmt(100*b.win_rate,0)}%<small>对 {opponentOf(b)} · episode {b.episode??'最终'}</small></>:'—'}</td><td><button onClick={()=>remove(d.job.id)} aria-label={'移除 '+d.job.name}>移除</button></td></tr>})}
   </tbody></table></div>
   <div className="aw-toolbar pf-curve-controls"><label>横轴<select aria-label="横轴" value={axis} onChange={e=>setAxis(e.target.value as AxisKey)}><option value="episode">episode</option><option value="step">环境步数</option></select></label>
    <label>平滑<select aria-label="平滑" value={smooth} onChange={e=>setSmooth(Number(e.target.value))}>{[0,0.6,0.9].map(v=><option key={v} value={v}>{v?v:'无'}</option>)}</select></label>
    <ul className="pf-legend">{data.map((d,i)=><li key={d.job.id}><i style={{background:COLORS[i]}}/>{d.job.name}</li>)}</ul></div>
   <div className="pf-chart-grid">{MAIN.map(([k,title])=>chart(k,title))}</div>
   {opponents.length>0&&<section className="pf-chart-panel pf-chart-wide" aria-label="评测胜率对比"><header><h3>评测胜率</h3><label>对手<select aria-label="评测对手" value={chosenOpponent} onChange={e=>setOpponent(e.target.value)}>{opponents.map(o=><option key={o} value={o}>{o}</option>)}</select></label></header>
    <LineChart series={data.map((d,i)=>({points:evalPoints(d),yKey:'win_rate',label:d.job.name,color:COLORS[i]})).filter(s=>s.points.length)} xKey="episode" xLabel="episode" yLabel="胜率" width={1200} height={240}/></section>}
   {others.length>0&&<section className="pf-chart-panel pf-chart-wide" aria-label="其他指标对比"><header><h3>其他指标</h3><label><select aria-label="对比指标" value={others.includes(extra)?extra:others[0]} onChange={e=>setExtra(e.target.value)}>{others.map(k=><option key={k} value={k}>{k}</option>)}</select></label></header>
    {(()=>{const key=others.includes(extra)?extra:others[0];return <LineChart series={data.map((d,i)=>({points:series(rows[i],[key],smooth,false),yKey:key,label:d.job.name,color:COLORS[i]})).filter((_,i)=>rows[i].some((r:Obj)=>numeric(r[key])))} xKey={axis} xLabel={axis} yLabel={key} width={1200} height={240}/>})()}</section>}
  </>}</>
}
