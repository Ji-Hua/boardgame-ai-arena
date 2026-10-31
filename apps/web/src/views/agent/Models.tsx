import {Link, useParams, useSearchParams} from 'react-router-dom'
import {contextQuery, runRoute} from '../../platform'
import {trainingUrl} from '../TrainingWorkspace'
import {EvidenceViewer} from '../EvidenceViewer'
import {type Obj, useV1, enc, fmt, route, ErrorBox, STATES, States} from './shared'

export const semantics=(s?:string|null)=>({q_values:'Q 值',policy_logits:'策略 logits'}[s as string]??s??'未记录')
export const modelName=(m:Obj)=>m.aliases?.[0]??m.sha256.slice(0,12)
const best=(m:Obj)=>[...(m.seasons??[])].sort((a:Obj,b:Obj)=>(b.elo??-Infinity)-(a.elo??-Infinity))[0]
export function experimentOf(record:Obj){const raw=String(record?.training_experiment??record?.source_path??'');const m=raw.match(/\/experiments\/([A-Za-z0-9][A-Za-z0-9._-]*)(?:\/(.*))?$/);return m?{slug:m[1],arm:(m[2]??'').replace(/\/checkpoints\/[^/]+$/,'')}:null}

export function ModelList({season}:{season:string}){
 const [params,setParams]=useSearchParams();const q=params.get('q')??'',state=params.get('state')??'all',arch=params.get('arch')??'all'
 const r=useV1('/models');const models:Obj[]=r.data?.models??[]
 function update(k:string,v:string){const p=new URLSearchParams(params);if(v&&v!=='all')p.set(k,v);else p.delete(k);setParams(p,{replace:true})}
 const rows=models.filter(m=>(state==='all'||m[state])&&(arch==='all'||m.architecture===arch)&&[m.sha256,...m.aliases,m.architecture??''].join(' ').toLowerCase().includes(q.toLowerCase()))
 return <><div className="aw-toolbar"><h1>模型库</h1><small>按 checkpoint SHA 去重 · 能力状态互不推导 · 静态资格不等于已成功加载</small></div>
 <ErrorBox text={r.error} retry={r.retry}/>{!r.data&&!r.error&&<p role="status">加载模型索引…</p>}
 {r.data&&<><div className="pf-chips" role="group" aria-label="能力状态"><button aria-pressed={state==='all'} onClick={()=>update('state','all')}>全部 <b>{r.data.summary.total}</b></button>{STATES.map(([k,label])=><button key={k} aria-pressed={state===k} onClick={()=>update('state',k)}>{label} <b>{r.data!.summary[k]}</b></button>)}</div>
 <div className="aw-toolbar"><input aria-label="搜索模型" placeholder="别名 / SHA" value={q} onChange={e=>update('q',e.target.value)}/><select aria-label="架构" value={arch} onChange={e=>update('arch',e.target.value)}><option value="all">全部架构</option>{[...new Set(models.map(m=>m.architecture).filter(Boolean))].map(a=><option key={a} value={a}>{a}</option>)}</select><small>{rows.length} / {models.length} 个模型</small></div>
 <div className="aw-table"><table><thead><tr><th>模型</th><th>架构 / 输出</th><th>状态</th><th>最佳天梯</th><th>训练来源</th></tr></thead><tbody>{rows.map(m=>{const b=best(m);return <tr key={m.sha256}><td><Link to={'/models/'+m.sha256+'?season='+enc(season)}>{modelName(m)}</Link><small>{m.aliases.length>1?`+${m.aliases.length-1} 别名 · `:''}{m.sha256.slice(0,12)}</small></td><td>{m.architecture??'未记录'}<small>{semantics(m.output_semantics)}</small></td><td><States model={m}/></td><td>{b?<>{fmt(b.elo,0)}<small>{b.season}</small></>:'—'}</td><td>{m.run_refs.length?`${m.run_refs.length} 个 Run`:'—'}</td></tr>})}</tbody></table></div>{!rows.length&&<p>没有匹配的模型</p>}
 {r.data.warnings?.length>0&&<details className="aw-note"><summary>索引提示 ({r.data.warnings.length})</summary>{r.data.warnings.map((w:string)=><p key={w}>{w}</p>)}</details>}</>}</>
}

export function ModelDetail({season}:{season:string}){
 const {sha=''}=useParams();const [params,setParams]=useSearchParams();const r=useV1(/^[a-f0-9]{64}$/.test(sha)?'/models/'+sha:null);const m=r.data
 const tab=['evaluation','training','files','reasons','evidence'].includes(params.get('tab')??'')?params.get('tab')!:'evaluation'
 const experiments=m?[...new Map((m.evidence??[]).map((e:Obj)=>experimentOf(e.record)).filter(Boolean).map((x:Obj)=>[x.slug+':'+x.arm,x])).values()] as Obj[]:[]
 const playable=m?.seasons?.find((s:Obj)=>s.season===season)??m?.seasons?.[0]
 if(!/^[a-f0-9]{64}$/.test(sha))return <p role="alert">模型路由需要完整的 64 位小写 SHA-256，不接受别名或缩写</p>
 return <><div className="aw-breadcrumb"><Link to="/models">模型库</Link> / {m?modelName(m):sha.slice(0,12)}</div><ErrorBox text={r.error} retry={r.retry}/>{!m&&!r.error&&<p role="status">加载模型并校验 SHA…</p>}
 {m&&<><div className="aw-toolbar"><h1>{modelName(m)}</h1><States model={m}/></div><span className="aw-hash">{m.sha256}</span>
 <div className="aw-toolbar aw-metrics"><span>{m.architecture??'架构未记录'}</span><span>{semantics(m.output_semantics)}</span><span>{m.sha_verified?'SHA 已校验':'SHA 未校验'}</span>{m.aliases.length>1&&<span>别名 {m.aliases.join(' / ')}</span>}</div>
 <div className="aw-toolbar">{m.analyzable?<Link className="button-link" to={'/lab?'+contextQuery(undefined,undefined,undefined,m.sha256)}>在实验室分析</Link>:<button disabled title={(m.reasons.analyzable??[]).join(', ')}>不可分析</button>}{playable?<Link className="button-link" to={route('/play',playable.season,{p1:playable.agent,p2:'human'})}>发起对战</Link>:<button disabled title="此 checkpoint 不在任何已完成评测批次中">暂不可对战</button>}<Link to={'/checkpoints/'+m.sha256+'?season='+enc(season)}>交手记录与回放 →</Link></div>
 {m.conflicts.length>0&&<div className="error" role="alert">证据冲突：{m.conflicts.join(' · ')}</div>}
 <div className="aw-tabs" role="tablist" aria-label="模型详情">{[['evaluation',`天梯评测 · ${m.seasons.length}`],['training',`训练来源 · ${m.run_refs.length+experiments.length}`],['files','Checkpoint 文件'],['reasons','能力说明'],['evidence','原始证据']].map(([id,label])=><button key={id} role="tab" aria-selected={tab===id} onClick={()=>{const p=new URLSearchParams(params);p.set('tab',id);setParams(p,{replace:true})}}>{label}</button>)}</div>
 <div className="pf-tab-body" role="tabpanel">
 {tab==='evaluation'&&(m.seasons.length?<div className="aw-table"><table><thead><tr><th>评测批次</th><th>Agent / 策略</th><th>排名</th><th>Elo</th><th>胜 / 平 / 负</th></tr></thead><tbody>{m.seasons.map((s:Obj)=><tr key={s.season+s.agent}><td>{s.season}<small>{s.protocol}</small></td><td><Link to={route('/agents/'+enc(s.agent),s.season)}>{s.name}</Link><small>{s.policy?.type??'—'}{s.policy?.k!=null?' · k='+s.policy.k:''}</small></td><td>{s.rank??'—'}</td><td>{fmt(s.elo,0)}</td><td>{s.wins??'—'} / {s.draws??'—'} / {s.losses??'—'}</td></tr>)}</tbody></table></div>:<p>此 checkpoint 尚未出现在已完成的评测批次中</p>)}
 {tab==='training'&&<>{experiments.length>0&&<><h2>实验</h2><ul className="pf-list">{experiments.map(x=><li key={x.slug+x.arm}><Link to={'/experiments/'+enc(x.slug)}>{x.slug}</Link> <small>{x.arm||'—'}</small></li>)}</ul></>}<h2>训练 Run</h2>{m.run_refs.length?<div className="aw-table"><table><thead><tr><th>Run</th><th>来源 / 证据</th><th>Checkpoint</th><th></th></tr></thead><tbody>{m.run_refs.map((ref:Obj,i:number)=><tr key={i}><td><Link to={trainingUrl(season,[ref.run_id])}>{ref.run_id}</Link></td><td>{ref.source}<small>{ref.evidence}</small></td><td>{ref.checkpoint??'—'}</td><td><Link to={runRoute(ref.run_id)}>完整详情</Link></td></tr>)}</tbody></table></div>:<p className="muted">没有按 SHA 确切关联的训练 Run；不按名称猜测</p>}</>}
 {tab==='files'&&<div className="aw-table"><table><thead><tr><th>路径</th><th>状态</th></tr></thead><tbody>{m.checkpoints.map((c:Obj)=><tr key={c.path}><td><span className="aw-hash">{c.path}</span></td><td>{c.available?'可用':'不可用'} · {c.sha_verified?'SHA 已校验':'SHA 未校验'}<small>{c.reason}</small></td></tr>)}</tbody></table></div>}
 {tab==='reasons'&&<dl className="pf-reasons">{[['available','文件可用'],['sha_verified','SHA 校验'],...STATES].map(([k,label])=><div key={k}><dt>{label} · {m[k]?'是':'否'}</dt><dd>{(m.reasons?.[k]??[]).join(' · ')||(k==='ranked'?(m.ranked?'出现在已完成评测批次':'未出现在已完成评测批次'):'无阻塞项')}</dd></div>)}</dl>}
 {tab==='evidence'&&<EvidenceViewer filename={'model-'+m.sha256.slice(0,12)+'.json'} value={m}/>}
 </div></>}</>
}
