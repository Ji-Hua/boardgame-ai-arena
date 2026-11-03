import {useState} from 'react'
import {type Obj,useV1,request,enc,ErrorBox} from './shared'
export function ResearchEligibility({project,policies}:{project:string,policies:Obj[]}){
 const data=useV1('/jobs/projects/'+enc(project)+'/eligibility')
 const [sha,setSha]=useState(''),[role,setRole]=useState(''),[evaluation,setEvaluation]=useState('')
 const [note,setNote]=useState(''),[token,setToken]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false)
 const policy=policies.find(p=>p.sha256===sha),roles=policy?.agent_specs?.map((s:Obj)=>s.role)??[]
 async function decide(decision:string){
  setBusy(true);setError('')
  try{await request('/api/v1/jobs/projects/'+enc(project)+'/eligibility',{method:'POST',headers:{'Content-Type':'application/json','X-Platform-Token':token},body:JSON.stringify({sha256:sha,role,target:'research-play',evaluation_job:evaluation,decision,note})});data.retry()}
  catch(e){setError(String(e))}finally{setBusy(false)}
 }
 return <section><h2>研究资格审计</h2><p>按角色记录人工评审，不修改线上部署或现有天梯。评审必须引用该策略已完成的评测任务。</p>
 <ErrorBox text={error||data.error}/>
 <label>策略<select aria-label="资格策略" value={sha} onChange={e=>{setSha(e.target.value);setRole('')}}><option value="">选择策略</option>{policies.map(p=><option key={p.sha256+p.job_id} value={p.sha256}>{p.sha256.slice(0,12)}</option>)}</select></label>
 <label>角色<select aria-label="资格角色" value={role} onChange={e=>setRole(e.target.value)}><option value="">选择角色</option>{roles.map((r:string)=><option key={r}>{r}</option>)}</select></label>
 <label>评测任务<input aria-label="资格评测任务" value={evaluation} onChange={e=>setEvaluation(e.target.value)}/></label>
 <label>评审理由<input aria-label="资格评审理由" value={note} onChange={e=>setNote(e.target.value)}/></label>
 <label>写入令牌<input aria-label="资格写入令牌" type="password" autoComplete="off" value={token} onChange={e=>setToken(e.target.value)}/></label>
 {['approve','withdraw'].map(d=><button key={d} disabled={busy||!sha||!role||!evaluation||!note||!token} onClick={()=>void decide(d)}>{d==='approve'?'记录认可':'记录撤回'}</button>)}
 <ul>{(data.data?.current??[]).map((r:Obj)=><li key={r.scope_hash}>{r.role} · {r.agent_spec.policy_sha256.slice(0,12)} · {r.decision} · {r.note}</li>)}</ul></section>
}
