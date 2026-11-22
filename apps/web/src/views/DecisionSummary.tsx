import React from 'react'
type Obj=Record<string,any>
export const actionKey=(a?:Obj)=>a?.type&&a?.target?`${a.type}:${a.target[0]}:${a.target[1]}`:null
export const briefNumber=(n:any)=>typeof n==='number'&&Number.isFinite(n)?(n===0?'0':Math.abs(n)<.001||Math.abs(n)>=10000?n.toExponential(2):Number(n.toPrecision(3)).toString()):String(n??'—')
export function orderedCandidates(result:Obj){const val=(v:any)=>v==='+inf'?Infinity:v==='-inf'?-Infinity:Number(v??-Infinity);return [...(result?.candidates??[])].sort((a,b)=>(val(b.value)-val(a.value))||((a.order??0)-(b.order??0)))}
export function DecisionSummary({actor,result,actual,comparison}:{actor:Obj;result:Obj;actual?:Obj;comparison:boolean}){
 const rows=orderedCandidates(result),actualId=actionKey(actual),played=rows.find(r=>r.id===actualId),best=rows.find(r=>r.id===result.selected_id)
 const policy=actor?.spec?.policy,k=policy?.k??1,algo=actor?.spec?.algo?.type
 const eligible=rows.filter(r=>r.route_eligible!==false),rank=played?eligible.findIndex(r=>r.id===played.id)+1:0
 const gap=played&&best&&played.route_eligible!==false&&played.bound==='exact'&&best.bound==='exact'&&typeof played.value==='number'&&typeof best.value==='number'?best.value-played.value:null
 const stochastic=policy?.type==='top_k'&&k>1
 const label=(c?:Obj)=>c?.label??c?.id??'—'
 return <section className="aw-decision-summary" aria-label="决策依据">
 <div className="aw-proof-label"><span className="aw-tag">{comparison?'对照复算':'当前 Agent · 重新计算'}</span><strong>{actor?.name??actor?.id}</strong></div>
 <div className="aw-decision-facts"><div><small>历史实际动作</small><strong>{actualId?label(played)=== '—'?actualId:label(played):'尚未落子'}</strong></div><div><small>复算 Top1</small><strong>{label(best)}</strong></div><div><small>实际动作排序</small><strong>{played?(rank?`${rank} / ${eligible.length}`:'类型未选'):'—'}</strong></div><div><small>与复算 Top1 分差</small><strong>{gap===null?'—':briefNumber(gap)}</strong></div></div>
 <div className="aw-basis"><span>选择策略</span><strong>{policy?.type==='top_k'?(k===1?'Top1 · 最高分，同分按动作顺序':`Top-${k} · 前 ${k} 个均匀抽样`):policy?.type??(actor?.kind==='human'?'玩家选择':'未记录')}</strong>
 {algo==='minimax'&&<><span>搜索</span><strong>深度 {actor.spec.algo.params?.depth} · 节点 {result.stats?.nodes??'—'} · 剪枝 {result.stats?.cutoffs??'—'}</strong></>}
 {algo==='greedy'&&<><span>评分依据</span><strong>对手路径增量 − 自身路径增量</strong></>}
 {result.neural&&<><span>网络输出</span><strong>{result.neural.model?.output_semantics} · {rows.length} 个合法动作</strong><span>合法掩码</span><strong>原始最大值 {result.neural.raw_best?.label??'—'} · {result.neural.raw_best?.legal===true?'合法':result.neural.raw_best?.legal===false?'不合法，已排除':'未记录'}</strong></>}
 {result.neural?.hierarchy&&<><span>动作类型</span><strong>{result.neural.hierarchy.selected_type} · 先选类型，再比较类型内动作</strong></>}
 {played&&<><span>实际动作的路径变化</span><strong>自身 {briefNumber(played.delta_self)} · 对手 {briefNumber(played.delta_opp)}{algo==='dqn'?'（规则诊断，不是网络输入贡献）':''}</strong></>}
 </div>
 {actualId&&<p className={'aw-verdict '+(actualId===result.selected_id?'match':'different')}>{comparison?(actualId===result.selected_id?'对照 Top1 与历史落子相同':'对照 Top1 与历史落子不同'):stochastic?(played&&rank>0&&rank<=k?'实际动作在本次评分的 Top-k 内；历史抽样状态未记录':'实际动作不在本次评分的 Top-k 内；不能还原历史抽样'):policy?.type==='top_k'?(actualId===result.selected_id?'复算 Top1 与历史落子一致':'复算 Top1 与历史落子不一致'): '策略未完整记录，不能据此还原历史选择'}</p>}
 <small className="aw-proof-note">候选与依据为当前代码复算；历史候选、搜索树未记录。{rows.some(r=>r.bound!=='exact')?'排序包含搜索界，非精确名次；无精确值不显示分差。':''}{stochastic?'诊断接口只返回 Top1 评分，不重放抽样。':''}</small>
 </section>
}
