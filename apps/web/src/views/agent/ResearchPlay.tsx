import {useEffect,useState} from 'react'
import {Link,useLocation,useNavigate,useParams,useSearchParams} from 'react-router-dom'
import {type Obj,useV1,request,enc,ErrorBox,Json} from './shared'

import {MrJackBoard} from './MrJackBoard'

export function ResearchPlay(){
 const {id=''}=useParams(),[params]=useSearchParams(),location=useLocation(),navigate=useNavigate()
 const projects=useV1('/jobs/projects'),policies=useV1('/jobs/projects/'+enc(id)+'/policies')
 const project=(projects.data?.projects??[]).find((p:Obj)=>p.id===id)
 const game=(projects.data?.games??[]).find((g:Obj)=>g.ref.game_id===project?.game_ref.game_id&&g.ref.ruleset===project?.game_ref.ruleset&&g.ref.version===project?.game_ref.version)
 const [role,setRole]=useState(''),[opponent,setOpponent]=useState(params.get('policy')??'random')
 const [view,setView]=useState<Obj|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false)
 const [watch,setWatch]=useState(''),[replay,setReplay]=useState<Obj|null>(null),[frame,setFrame]=useState(0)
 const session=params.get('session'),token=new URLSearchParams(location.hash.slice(1)).get('token')??''
 useEffect(()=>{if(game&&!game.roles.includes(role))setRole(game.roles[0])},[game,role])
 useEffect(()=>{
  setView(null);setReplay(null);setError('')
  if(!session)return
  let active=true
  async function load(){try{const d=await request('/api/v1/play-v2/'+enc(session!),{headers:{'X-Game-Token':token}});if(active)setView(d)}catch(e){if(active)setError(String(e))}}
  void load();const t=setInterval(()=>void load(),3000)
  return()=>{active=false;clearInterval(t)}
 },[session,token])
 async function create(){
  setBusy(true);setError('')
  try{const d=await request('/api/v1/play-v2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:id,game_ref:project.game_ref,human_role:role,opponent})})
   setWatch('/projects/'+enc(id)+'/play?session='+enc(d.id)+'#token='+enc(d.watch_token))
   navigate('/projects/'+enc(id)+'/play?session='+enc(d.id)+'#token='+enc(d.token));setView(d.view)
  }catch(e){setError(String(e))}finally{setBusy(false)}
 }
 async function step(action?:number){
  setBusy(true);setError('')
  try{const d=await request('/api/v1/play-v2/'+enc(session!)+'/step',{method:'POST',headers:{'Content-Type':'application/json','X-Game-Token':token},body:JSON.stringify({revision:view!.revision,action:action??null})});setView(d);setReplay(null)}
  catch(e){setError(String(e))}finally{setBusy(false)}
 }
 async function showReplay(){
  try{const d=await request('/api/v1/play-v2/'+enc(session!)+'/replay',{headers:{'X-Game-Token':token}});setReplay(d);setFrame(d.frames.length-1)}catch(e){setError(String(e))}
 }
 async function close(){
  try{await request('/api/v1/play-v2/'+enc(session!),{method:'DELETE',headers:{'X-Game-Token':token}});navigate('/projects/'+enc(id)+'/play');setWatch('')}
  catch(e){setError(String(e))}
 }
 const shown=replay?.frames[frame]??view
 return <><Link to="/projects">研究项目</Link><h1>研究对局 · {project?.title??id}</h1>
 <ErrorBox text={error||projects.error||policies.error}/>
 {!session&&project&&<div className="pf-form">
 <label>你的角色<select aria-label="研究对局角色" value={role} onChange={e=>setRole(e.target.value)}>{(game?.roles??[]).map((r:string)=><option key={r}>{r}</option>)}</select></label>
 <label>对手<select aria-label="研究对局对手" value={opponent} onChange={e=>setOpponent(e.target.value)}><option value="random">随机策略</option>{game?.ref.game_id==='mr_jack'&&<option value="heuristic">基础规则策略</option>}{(policies.data?.policies??[]).map((p:Obj)=><option key={p.sha256+p.job_id} value={p.sha256}>{p.algorithm} · {p.sha256.slice(0,12)}</option>)}</select></label>
 <button disabled={busy||!role} onClick={()=>void create()}>开始研究对局</button></div>}
 {watch&&<p><Link to={watch} target="_blank" rel="noreferrer">打开观战视图</Link>（仅公开信息）</p>}
 {view&&<><p>视角：{view.viewer} · 决策 {view.revision} · {view.status} · 当前角色 {view.actor??'终局'}</p>
 {game?.ref.game_id==='mr_jack'?<MrJackBoard observation={shown?.observation}/>:<Json value={shown?.observation}/>}
 {view.status==='active'&&view.viewer!=='watch'&&!replay&&<div className="aw-toolbar">{view.actor===view.viewer?view.legal_actions.map((a:Obj)=><button key={a.id} disabled={busy} onClick={()=>void step(a.id)}>{a.label}</button>):<button disabled={busy} onClick={()=>void step()}>执行对手动作</button>}</div>}
 {Object.keys(view.returns??{}).length>0&&<Json value={view.returns}/>}
 <button onClick={()=>void showReplay()}>查看本视角回放</button>{replay&&<><input aria-label="研究回放进度" type="range" min={0} max={replay.frames.length-1} value={frame} onChange={e=>setFrame(Number(e.target.value))}/><button onClick={()=>setReplay(null)}>返回当前对局</button></>}
 {view.viewer!=='watch'&&<button onClick={()=>void close()}>关闭对局</button>}</>}
 </> 
}
