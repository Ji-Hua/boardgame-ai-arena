import React,{useEffect,useRef,useState} from 'react'
import {labRequest,type Position} from '../lab'
import {Tensor} from './LabView'
import {actionKey,briefNumber,orderedCandidates} from './DecisionSummary'
type Obj=Record<string,any>
export function differenceViews(a:Obj,b:Obj){
 for(const field of ['position_hash','checkpoint_sha256','algorithm','depth'])if(a[field]!==b[field])throw new Error('A/B 证据来源不一致：'+field)
 const av=a.neural?.sensitivities?.find((s:Obj)=>s.objective==='q')?.views
 const bv=b.neural?.sensitivities?.find((s:Obj)=>s.objective==='q')?.views
 if(!av||!bv||av.length!==bv.length)throw new Error('缺少匹配的原始输出梯度')
 return av.map((v:Obj,i:number)=>{
  const w=bv[i];if(v.name!==w.name||v.layout!==w.layout||v.values.length!==w.values.length)throw new Error('梯度输入不一致')
  const values=v.values.map((row:number[],x:number)=>{if(row.length!==w.values[x].length)throw new Error('梯度形状不一致');return row.map((n,y)=>{const d=n-w.values[x][y];if(!Number.isFinite(d))throw new Error('梯度包含非有限值');return d})})
  // Current CNN encoder broadcasts stock features; perturbing one stock scalar
  // changes all its cells together. Sum partial derivatives for that scalar.
  if((i===4||i===5)&&av.length===7)return {...v,layout:'scalar',values:[[values.flat().reduce((s:number,n:number)=>s+n,0)]],name:v.name+' · 广播标量合计'}
  return {...v,values}
 })
}
export function PairComparison({position,actor,result,actual,bId,onB,onBranch,onView}: {position:Position;actor:Obj;result:Obj;actual?:Obj;bId:string;onB:(id:string)=>void;onBranch:(p:Position|null)=>void;onView:(label:string)=>void}){
 const rows=orderedCandidates(result),a=rows.find(r=>r.id===actionKey(actual))??(!actual?rows.find(r=>r.id===result.selected_id):undefined)
 const b=rows.find(r=>r.id===bId&&r.id!==a?.id)??rows.find(r=>r.id!==a?.id)
 const [mode,setMode]=useState('scores'),[pair,setPair]=useState<Obj|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[plane,setPlane]=useState(0),[retry,setRetry]=useState(0)
 const cache=useRef(new Map<string,Obj>())
 const key=JSON.stringify([position,actor,a?.id,b?.id,result.created_at])
 useEffect(()=>{setPair(null);setError('');setBusy(false);onBranch(null);onView('原局面 · 落子前')},[key])
 useEffect(()=>{
  if(mode==='scores'||!a||!b){setBusy(false);return}
  const cached=cache.current.get(key);if(cached){setPair(cached);setBusy(false);return}
  const c=new AbortController();setBusy(true);setError('')
  const timer=setTimeout(()=>{void(async()=>{try{
   const payload={position,algorithm:actor.spec.algo.type,depth:actor.spec.algo.params?.depth??1,...(actor.spec.algo.type==='dqn'?{model_id:actor.actual_sha256??actor.sha256}:{})}
   const left=await labRequest('branch',{...payload,action:a.action},c.signal)
   if(c.signal.aborted)return
   const right=await labRequest('branch',{...payload,action:b.action},c.signal)
   if(c.signal.aborted)return
   for(const field of ['position_hash','checkpoint_sha256','algorithm','depth'])if(left[field]!==right[field])throw new Error('A/B 来源不一致')
   if(result.position_hash&&left.position_hash!==result.position_hash)throw new Error('局面证据不一致')
   const data={a:left,b:right,views:result.neural?differenceViews(left,right):[]}
   cache.current.set(key,data);while(cache.current.size>12)cache.current.delete(cache.current.keys().next().value!)
   setPair(data)
  }catch(e:any){if(!c.signal.aborted)setError(String(e))}finally{if(!c.signal.aborted)setBusy(false)}})()},250)
  return()=>{clearTimeout(timer);c.abort()}
 },[key,mode,retry])
 function view(next:string){onBranch(null);onView('原局面 · 落子前');setMode(next)}
 const numeric=(v:Obj|undefined)=>v?.bound==='exact'&&typeof v.value==='number'&&Number.isFinite(v.value)
 const left=pair?.a??a,right=pair?.b??b,gap=numeric(left)&&numeric(right)?left.value-right.value:null
 const fmt=(r:Obj|undefined)=>r?(r.bound==='upper'?'≤ ':r.bound==='lower'?'≥ ':'')+briefNumber(r.value):'—'
 return <section className="aw-pair" aria-label="动作 A B 比较"><div className="aw-toolbar"><strong>A · {actual?'实际动作':'复算首选'}：{a?.label??actionKey(actual)??'—'}</strong><label>B<select aria-label="对照动作 B" value={b?.id??''} onChange={e=>{onBranch(null);onView('原局面 · 落子前');onB(e.target.value)}}>{rows.filter(r=>r.id!==a?.id).map(r=><option key={r.id} value={r.id}>{r.label??r.id}{r.route_eligible===false?' · 类型未选':''}</option>)}</select></label></div>
 <div className="aw-pair-scores"><span>A <b>{fmt(left)}</b></span><span>B <b>{fmt(right)}</b></span><span>A−B <b>{gap===null?'—':briefNumber(gap)}</b></span><small>{result.neural?.model?.output_semantics??'启发式分'} · 复算</small></div>
 {(a?.route_eligible===false||b?.route_eligible===false)&&<small className="aw-proof-note">包含未被类型路由选中的动作；动作头分差不能单独解释最终选择。</small>}
 <div className="aw-tabs" role="tablist" aria-label="A B 证据">{[['scores','评分'],['input','输入证据'],['branches','分支比较']].map(([id,label])=><button key={id} role="tab" aria-selected={mode===id} onClick={()=>view(id)}>{label}</button>)}</div>
 {!a&&<p>实际动作不在当前合法候选中，无法建立 A/B 比较。</p>}{!b&&<p>没有其他合法候选。</p>}
 {busy&&<small role="status">计算 A/B 证据…</small>}{error&&<div className="error">{error}<button onClick={()=>setRetry(v=>v+1)}>重试比较</button></div>}
 {mode==='input'&&<><small className="aw-proof-note">原局面 · 落子前输入。局部敏感度 ∂(A−B)/∂输入；正值表示增大该输入更偏向 A，不代表因果贡献。</small>{result.neural?<><select aria-label="A B 输入平面" value={plane} onChange={e=>setPlane(+e.target.value)}>{result.neural.input.views.map((v:Obj,i:number)=><option key={i} value={i}>{v.name}</option>)}</select><div className="aw-pair-input"><div><h3>实际输入</h3><Tensor data={result.neural.input.views[plane]}/></div>{pair?.views[plane]&&<div><h3>A−B 敏感度</h3><Tensor data={pair.views[plane]}/></div>}</div></>:<p className="muted">该 Agent 无神经网络输入；在分支比较中查看路径与搜索依据。</p>}</>}
 {mode==='branches'&&pair&&<><small className="aw-proof-note">{result.neural?'引擎单步推进，不是网络搜索；分数来自落子前。':`相同算法与深度 ${actor.spec.algo.params?.depth??1}；有限深度结果，不是已求解胜负。`} 路径长度忽略对方棋子。</small><div className="aw-pair-branches">{[['a',a],['b',b]].map(([id,row]:any)=>{const d=pair[id],self=position.turn-1;return <div key={id}><h3>{id.toUpperCase()} · {row?.label}</h3><small>自身路径 {row?.before_self_opp?.[0]??result.distances?.[self]??'—'} → {d.after?.distances?.[self]??'—'} · 对手 {row?.before_self_opp?.[1]??result.distances?.[1-self]??'—'} → {d.after?.distances?.[1-self]??'—'}</small><button onClick={()=>{onBranch(d.after.position);onView(`假设分支 ${id.toUpperCase()} · ${row?.label} 后`)}}>查看 {id.toUpperCase()} 分支</button>{d.principal_variation?.map((p:Obj,i:number)=><button key={i} disabled={!p.position&&!p.after?.position} onClick={()=>{onBranch(p.position??p.after.position);onView(`假设分支 ${id.toUpperCase()} · 第 ${i+1} 手后`)}}>{i+1}. {p.label??p.action_id??'局面'}</button>)}</div>})}</div><button onClick={()=>{onBranch(null);onView('原局面 · 落子前')}}>返回比较原局面</button></>}
 </section>
}

