import type {CSSProperties} from 'react'
import {Link} from 'react-router-dom'
import {replayRoute} from '../../platform'
import {type Obj, enc, fmt} from './shared'
import {EvalCurve} from './JobCurves'

const RESULT:Record<string,string>={win:'胜',loss:'负',draw:'平'}
const opponentOf=(r:Obj)=>String(r.opponent_name??r.opponent??r.opponent_id??'—')
const rate=(r:Obj)=>typeof r.win_rate==='number'?r.win_rate:null

/** One row per saved checkpoint: evaluations at that episode, the recorded sample games, diagnosis and next steps. */
export function CheckpointTimeline({job,evaluations,replays}:{job:Obj;evaluations:Obj[];replays:Obj[]}){
 const rows:Obj[]=job.checkpoints??[]
 if(!rows.length)return <p>{['queued','preparing','running'].includes(job.status)?'还没有保存 checkpoint':'此任务没有 checkpoint'}</p>
 const opponents=[...new Set(evaluations.filter(r=>r.episode!=null).map(opponentOf))]
 const evalAt=(episode:number,opponent:string)=>evaluations.filter(r=>r.episode===episode&&opponentOf(r)===opponent).pop()
 return <>
  <EvalCurve rows={evaluations}/>
  <div className="aw-table pf-timeline"><table><thead><tr><th>Checkpoint</th>{opponents.map(o=><th key={o}>对 {o}<small>胜 / 平 / 负</small></th>)}<th>样例对局</th><th>诊断</th><th>下一步</th></tr></thead><tbody>
  {rows.map(c=>{const registered=c.sha256&&(job.registered??[]).includes(c.sha256),diagnosis=c.sha256&&(job.diagnoses??[]).find((d:Obj)=>d.sha256===c.sha256)
   const samples=replays.filter(r=>r.episode===c.episode&&(!r.checkpoint||r.checkpoint===c.name))
   return <tr key={c.file}>
    <td><strong>episode {c.episode??'—'}</strong><small>{c.step!=null?`环境步数 ${c.step} · `:''}{c.name}</small>{c.sha256&&(registered?<Link to={'/models/'+c.sha256}>{c.sha256.slice(0,12)}</Link>:<small>{c.sha256.slice(0,12)}</small>)}</td>
    {opponents.map(o=>{const e=c.episode!=null?evalAt(c.episode,o):undefined;const w=e&&rate(e)
     return <td key={o}>{e?<span className="pf-wdl" style={{'--rate':w??0} as CSSProperties}>{e.wins??'—'} / {e.draws??'—'} / {e.losses??'—'}{w!=null&&<small>{fmt(100*w,0)}%</small>}</span>:'—'}</td>})}
    <td>{samples.length?<ul className="pf-samples">{samples.map(r=><li key={r.replay_path}><Link to={replayRoute(job.run_id,{replay:r.replay_path,ply:0})}>{opponentOf(r)} · {RESULT[r.result_from_dqn_perspective]??r.result_from_dqn_perspective??'?'} · {r.game_length} 步</Link></li>)}</ul>:'—'}</td>
    <td>{diagnosis?<>{diagnosis.tactical_wins??'—'} / {diagnosis.tactical_count??'—'} 战术题<small>开发对局 {diagnosis.counts?.wins??'—'} / {diagnosis.counts?.losses??'—'}</small></>:'—'}</td>
    <td>{registered?<div className="pf-next"><Link to={'/jobs/new?entrypoint=diagnose_checkpoint&sha='+enc(c.sha256)}>诊断</Link><Link to={'/jobs/new?entrypoint=run_arena_matrix&sha='+enc(c.sha256)}>门槛评测</Link></div>:<small>{c.sha256?'未登记':'任务结束后登记'}</small>}</td>
   </tr>})}
  </tbody></table></div></>
}
