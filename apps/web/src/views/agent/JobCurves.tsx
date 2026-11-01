import {useState} from 'react'
import {LineChart} from '../../components/LineChart'
import type {AxisKey, MetricPoint} from '../../types'
import {type Obj, fmt} from './shared'

export const COLORS=['#4f9eff','#f5a524','#3ecf8e','#e5484d']
/** Fixed panels for the training contract; any other numeric field stays reachable under "其他指标". */
const PANELS:{title:string;keys:[string,string][];log?:boolean}[]=[
 {title:'对局结果（最近窗口）',keys:[['win_rate','胜'],['loss_rate','负'],['truncation_rate','截断']]},
 {title:'行为',keys:[['repeat_rate','重复局面'],['wall_ratio','放墙比例']]},
 {title:'对局长度',keys:[['avg_episode_length','平均'],['max_episode_length','最长']]},
 {title:'损失',keys:[['loss','loss'],['demonstration_loss','示范 loss']],log:true},
 {title:'价值估计',keys:[['avg_q_max','|Q| 最大'],['q_mean','Q 均值'],['target_mean','目标均值']]},
 {title:'优化',keys:[['grad_norm','梯度范数'],['td_error_max','最大 TD 误差']],log:true},
]
const PANEL_KEYS=new Set(PANELS.flatMap(p=>p.keys.map(([k])=>k)))
const SUMMARY:[string,string,(v:number)=>string][]=[['win_rate','胜率',v=>fmt(100*v,0)+'%'],['truncation_rate','截断率',v=>fmt(100*v,0)+'%'],['repeat_rate','重复局面',v=>fmt(100*v,0)+'%'],['loss','loss',v=>fmt(v,4)],['episodes_per_s','局 / 秒',v=>fmt(v,2)]]

export const numeric=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)
export const curveKeys=(rows:Obj[])=>[...new Set(rows.flatMap(r=>Object.entries(r).filter(([k,v])=>typeof v==='number'&&!['episode','step'].includes(k)).map(([k])=>k)))]

/** Exponential moving average per key (TensorBoard style); log scale drops non-positive values. */
export function series(rows:Obj[],keys:string[],smooth:number,log:boolean):MetricPoint[]{
 const state:Record<string,number|undefined>={}
 return rows.map(r=>({episode:r.episode??null,step:r.step??null,values:Object.fromEntries(keys.map(k=>{
  let v=numeric(r[k])?r[k]:null
  if(v!==null&&log)v=v>0?Math.log10(v):null
  if(v!==null&&smooth>0){const prev=state[k];v=prev===undefined?v:smooth*prev+(1-smooth)*v;state[k]=v}
  return [k,v]}))}))
}

function Panel({title,keys,rows,axis,smooth,log}:{title:string;keys:[string,string][];rows:Obj[];axis:AxisKey;smooth:number;log:boolean}){
 const present=keys.filter(([k])=>rows.some(r=>numeric(r[k])))
 if(!present.length)return null
 const points=series(rows,present.map(([k])=>k),smooth,log)
 return <section className="pf-chart-panel" aria-label={title}><header><h3>{title}</h3><ul className="pf-legend">{present.map(([k,label],i)=><li key={k}><i style={{background:COLORS[i]}}/>{label}</li>)}{log&&<li><small>log₁₀</small></li>}</ul></header>
  <LineChart series={present.map(([k,label],i)=>({points,yKey:k,label,color:COLORS[i]}))} xKey={axis} xLabel={axis} yLabel={title} width={560} height={200}/></section>
}

export function JobCurves({rows,source,live}:{rows:Obj[];source?:string|null;live:boolean}){
 const [smooth,setSmooth]=useState(0),[axis,setAxis]=useState<AxisKey>('episode'),[log,setLog]=useState(false),[key,setKey]=useState('')
 const keys=curveKeys(rows),others=keys.filter(k=>!PANEL_KEYS.has(k))
 const chosen=others.includes(key)?key:others[0]
 const last=rows[rows.length-1]??{}
 return <>
  <div className="aw-toolbar pf-curve-controls"><label>横轴<select aria-label="横轴" value={axis} onChange={e=>setAxis(e.target.value as AxisKey)}><option value="episode">episode</option><option value="step">环境步数</option></select></label>
   <label>平滑<select aria-label="平滑" value={smooth} onChange={e=>setSmooth(Number(e.target.value))}>{[0,0.6,0.9].map(v=><option key={v} value={v}>{v?v:'无'}</option>)}</select></label>
   <label className="pf-check"><input type="checkbox" checked={log} onChange={e=>setLog(e.target.checked)}/>损失类用对数坐标</label>
   <small>{rows.length} 个记录点 · {source??'metrics.jsonl'}{live?' · 实时':''}</small></div>
  <div className="pf-curve-summary" aria-label="最新指标">{last.episode!=null&&<div><span>episode</span><b>{last.episode}</b></div>}{SUMMARY.filter(([k])=>numeric(last[k])).map(([k,label,show])=><div key={k}><span>{label}</span><b>{show(last[k])}</b></div>)}</div>
  <div className="pf-chart-grid">{PANELS.map(p=><Panel key={p.title} title={p.title} keys={p.keys} rows={rows} axis={axis} smooth={smooth} log={log&&!!p.log}/>)}</div>
  {chosen&&<section className="pf-chart-panel pf-chart-wide" aria-label="其他指标"><header><h3>其他指标</h3><label><select aria-label="任务指标" value={chosen} onChange={e=>setKey(e.target.value)}>{others.map(k=><option key={k} value={k}>{k}</option>)}</select></label></header>
   <LineChart points={series(rows,[chosen],smooth,false)} xKey={axis} xLabel={axis} yKey={chosen} yLabel={chosen} width={1200} height={240}/></section>}
 </>
}

export function JobAlerts({alerts}:{alerts:Obj[]}){
 if(!alerts.length)return null
 return <ul className="pf-alerts" aria-label="训练健康提示">{alerts.map(a=><li key={a.code+(a.episode??'')} className={a.level}><strong>{a.level==='error'?'错误':'注意'}</strong>{a.message}{a.episode!=null&&<small>episode {a.episode}</small>}</li>)}</ul>
}

/** Win rate per evaluation opponent over training; rows without an episode (final-only evaluations) are left to the table. */
export function EvalCurve({rows}:{rows:Obj[]}){
 const name=(r:Obj)=>String(r.opponent_name??r.opponent??r.opponent_id??'—')
 const opponents=[...new Set(rows.filter(r=>numeric(r.episode)&&numeric(r.win_rate)).map(name))].slice(0,COLORS.length)
 if(!opponents.length||rows.filter(r=>numeric(r.episode)).length<2)return null
 const points:MetricPoint[]=rows.filter(r=>numeric(r.episode)).map(r=>({episode:r.episode,step:r.step??null,values:{[name(r)]:numeric(r.win_rate)?r.win_rate:null}}))
 return <section className="pf-chart-panel pf-chart-wide" aria-label="评测胜率"><header><h3>评测胜率</h3><ul className="pf-legend">{opponents.map((o,i)=><li key={o}><i style={{background:COLORS[i]}}/>{o}</li>)}</ul></header>
  <LineChart series={opponents.map((o,i)=>({points,yKey:o,label:o,color:COLORS[i]}))} xKey="episode" xLabel="episode" yLabel="胜率" width={1200} height={240}/></section>
}
