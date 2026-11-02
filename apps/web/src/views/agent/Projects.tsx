import {useState} from 'react'
import {Link} from 'react-router-dom'
import {type Obj, useV1, request, enc, ErrorBox} from './shared'

export function Projects(){
 const data=useV1('/jobs/projects')
 const [id,setId]=useState(''),[title,setTitle]=useState(''),[question,setQuestion]=useState('')
 const [game,setGame]=useState('quoridor'),[token,setToken]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false)
 const games:Obj[]=data.data?.games??[]
 async function create(){
  setBusy(true);setError('')
  try{
   const selected=games.find(g=>g.ref.game_id===game)
   if(!selected)throw new Error('请选择已注册游戏')
   await request('/api/v1/jobs/projects',{method:'POST',headers:{'Content-Type':'application/json','X-Platform-Token':token},body:JSON.stringify({id,title,question,game_ref:selected.ref})})
   setId('');setTitle('');setQuestion('');data.retry()
  }catch(e){setError(String(e))}finally{setBusy(false)}
 }
 return <><h1>研究项目</h1><p>项目绑定固定游戏规则，任务与模型保留所属项目。历史 Quoridor 证据继续使用原有入口。</p>
 <ErrorBox text={data.error} retry={data.retry}/>
 {!data.data&&!data.error&&<p role="status">加载项目…</p>}
 {data.data&&<><div className="aw-table"><table><thead><tr><th>项目</th><th>研究问题</th><th>游戏</th><th>任务</th></tr></thead><tbody>
 {(data.data.projects as Obj[]).map(p=><tr key={p.id}><td>{p.title}<small>{p.id}</small></td><td>{p.question}</td><td>{p.game_ref.game_id} · {p.game_ref.ruleset} v{p.game_ref.version}</td><td><Link to={'/jobs?project='+enc(p.id)}>查看任务</Link>{games.find(g=>g.ref.game_id===p.game_ref.game_id)?.training_backend?<Link to={'/jobs/new?project='+enc(p.id)}>新建任务</Link>:<small>训练后端尚未接入</small>}</td></tr>)}
 </tbody></table></div>
 <h2>新建项目</h2><div className="pf-form">
 <label>项目 ID<input aria-label="项目 ID" value={id} onChange={e=>setId(e.target.value)} placeholder="例如 quoridor-ppo-study"/></label>
 <label>标题<input aria-label="项目标题" value={title} onChange={e=>setTitle(e.target.value)}/></label>
 <label>研究问题<textarea aria-label="研究问题" value={question} onChange={e=>setQuestion(e.target.value)}/></label>
 <label>游戏<select aria-label="项目游戏" value={game} onChange={e=>setGame(e.target.value)}>{games.map(g=><option key={g.ref.game_id} value={g.ref.game_id}>{g.title} · {g.ref.ruleset} v{g.ref.version}</option>)}</select></label>
 <label>写入令牌<input aria-label="项目写入令牌" type="password" autoComplete="off" value={token} onChange={e=>setToken(e.target.value)}/></label>
 <ErrorBox text={error}/><button disabled={busy||!id||!title||!question} onClick={()=>void create()}>{busy?'创建中…':'创建项目'}</button>
 </div></>}</>
}
