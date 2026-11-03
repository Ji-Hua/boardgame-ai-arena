import {useState} from 'react'
import {Link,useParams} from 'react-router-dom'
import {type Obj,useV1,enc,ErrorBox} from './shared'

import {ResearchEligibility} from './ResearchEligibility'

export function ResearchPolicies(){
 const {id=''}=useParams()
 const data=useV1('/jobs/projects/'+enc(id)+'/policies')
 const [token,setToken]=useState(''),[error,setError]=useState('')
 async function download(sha:string){
  setError('')
  try{
   const response=await fetch('/api/v1/jobs/projects/'+enc(id)+'/policies/'+enc(sha)+'/bundle',{headers:{'X-Platform-Token':token}})
   if(!response.ok)throw new Error('下载失败：'+response.status)
   const blob=await response.blob(),url=URL.createObjectURL(blob)
   const link=document.createElement('a');link.href=url;link.download=sha+'.zip';link.click()
   setTimeout(()=>URL.revokeObjectURL(url),1000)
  }catch(e){setError(String(e))}
 }
 return <><Link to="/projects">研究项目</Link><h1>研究策略 · {id}</h1>
 <p>策略以完整 SHA 固定身份。登记与评测不会自动晋升到现有天梯。</p>
 <ErrorBox text={data.error} retry={data.retry}/><ErrorBox text={error}/>
 <label>下载令牌<input aria-label="策略下载令牌" type="password" autoComplete="off" value={token} onChange={e=>setToken(e.target.value)}/></label>
 {!data.data&&!data.error&&<p role="status">加载策略…</p>}
 {data.data&&<div className="aw-table"><table><thead><tr><th>策略</th><th>算法与语义</th><th>来源</th><th>操作</th></tr></thead><tbody>
 {(data.data.policies as Obj[]).map(p=><tr key={p.sha256+p.job_id}><td><code>{p.sha256}</code><small>{p.game_ref.game_id}</small></td><td>{p.algorithm}<small>{p.policy_kind} · {p.output_semantics}</small></td><td><Link to={'/jobs/'+enc(p.job_id)}>训练任务</Link><small>{p.source.commit}</small></td><td><Link to={'/projects/'+enc(id)+'/play?policy='+enc(p.sha256)}>与策略对战</Link><Link to={'/jobs/new?project='+enc(id)+'&entrypoint='+'evaluate_mrjack_policy'+'&policy='+enc(p.sha256)}>评测策略</Link><button onClick={()=>void download(p.sha256)}>下载策略</button></td></tr>)}
 </tbody></table>{!data.data.policies.length&&<p>暂无已登记策略</p>}</div>}<ResearchEligibility project={id} policies={data.data?.policies??[]}/></>
}
