import {useEffect, useState, type ReactNode} from 'react'
import {Link, Navigate, Route, Routes, useLocation, useNavigate, useParams, useSearchParams} from 'react-router-dom'
import {Grid3X3, LayoutDashboard, FlaskConical, Activity, ListChecks, Trophy, Boxes, Microscope, Swords, Sun, Moon} from 'lucide-react'
import '../App.css'
import './AgentWorkspace.css'
import './Refinement.css'
import './Polish.css'
import './Platform.css'
import './Theme.css'
import {TrainingWorkspace} from './TrainingWorkspace'
import {RunDetailView} from './RunDetailView'
import {ReplaysView} from './ReplaysView'
import {ArenaView} from './ArenaView'
import {LabView} from './LabView'
import {type Obj, useData, enc, route, ErrorBox} from './agent/shared'
import {Overview} from './agent/Overview'
import {ExperimentList, ExperimentDetail} from './agent/Experiments'
import {ModelList, ModelDetail} from './agent/Models'
import {JobList, JobNew, JobDetail} from './agent/Jobs'
import {JobCompare} from './agent/JobCompare'
import {Ladder} from './agent/Ladder'
import {AgentDetail} from './agent/AgentDetail'
import {CheckpointPage} from './agent/CheckpointPage'
import {Replay} from './agent/Replay'
import {Play} from './agent/Play'
export {evidence, route, Participant} from './agent/shared'
export {StepAnalysis} from './agent/StepAnalysis'

const NAV:[string,string,typeof Trophy,(path:string)=>boolean][]=[
 ['/','总览',LayoutDashboard,p=>p==='/'],
 ['/experiments','实验',FlaskConical,p=>p.startsWith('/experiments')],
 ['/training','训练',Activity,p=>p.startsWith('/training')||p.startsWith('/replays')],
 ['/jobs','任务',ListChecks,p=>p.startsWith('/jobs')],
 ['/ladder','天梯',Trophy,p=>['/ladder','/agents','/games','/arena'].some(x=>p.startsWith(x))],
 ['/models','模型',Boxes,p=>p.startsWith('/models')||p.startsWith('/checkpoints')],
 ['/lab','实验室',Microscope,p=>p.startsWith('/lab')],
 ['/play','对战',Swords,p=>p.startsWith('/play')],
]

function RunPage(){const {runId=''}=useParams();const navigate=useNavigate();return <RunDetailView key={runId} runId={runId} onBack={()=>navigate('/training')}/>}

export default function AgentWorkspace(){
 const [params]=useSearchParams();const seasons=useData('/seasons');const list:Obj[]=seasons.data?.seasons??[]
 // Gate seasons only pair candidates with fixed opponents; default to the newest complete ladder season.
 const season=params.get('season')??(list.find(s=>!s.gate)??list[0])?.id??''
 const roster=useData(season?'/seasons/'+enc(season):null)
 const location=useLocation()
 const [theme,setTheme]=useState(()=>localStorage.getItem('quoridor-workbench-theme')==='light'?'light':'dark')
 useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem('quoridor-workbench-theme',theme)},[theme])
 const target=(path:string)=>season&&(path==='/ladder'||path==='/play')?route(path,season):path
 // Ladder, Agent, Game and Play pages need the selected season's roster; other workspaces do not wait for it.
 const rostered=(render:(roster:Obj)=>ReactNode)=><>
  <ErrorBox text={seasons.error||roster.error} retry={seasons.error?seasons.retry:roster.retry}/>
  {!seasons.data&&!seasons.error&&<p role="status">加载评测批次…</p>}
  {seasons.data?.seasons?.length===0&&<p>暂无已完成的天梯报告</p>}
  {season&&!roster.data&&!roster.error&&<p role="status">加载 Agent…</p>}
  {roster.data&&render(roster.data)}</>
 return <div className="app agent-workspace"><header className="aw-header"><Link to="/" className="aw-brand"><Grid3X3 aria-hidden="true"/>Quoridor</Link><nav aria-label="主要入口">{NAV.map(([path,label,Icon,active])=><Link key={path} to={target(path)} aria-current={active(location.pathname)?'page':undefined}><Icon aria-hidden="true"/>{label}</Link>)}</nav><a className="aw-preview" href="/mlflow/" title="MLflow 跟踪服务器（同一入口代理）">MLflow</a><button className="aw-theme-toggle" onClick={()=>setTheme(t=>t==='light'?'dark':'light')}>{theme==='light'?<Moon aria-hidden="true"/>:<Sun aria-hidden="true"/>}{theme==='light'?'深色':'浅色'}</button></header><main className="aw-main">
 <Routes>
  <Route path="/" element={<Overview season={season} roster={roster.data}/>}/>
  <Route path="/experiments" element={<ExperimentList/>}/>
  <Route path="/experiments/:slug" element={<ExperimentDetail season={season}/>}/>
  <Route path="/training" element={<TrainingWorkspace season={season}/>}/>
  <Route path="/training/runs/:runId" element={<RunPage/>}/>
  <Route path="/training/compare" element={<Navigate replace to="/training"/>}/>
  <Route path="/replays" element={<ReplaysView/>}/>
  <Route path="/replays/:runId" element={<ReplaysView/>}/>
  <Route path="/jobs" element={<JobList/>}/>
  <Route path="/jobs/new" element={<JobNew/>}/>
  <Route path="/jobs/compare" element={<JobCompare/>}/>
  <Route path="/jobs/:id" element={<JobDetail/>}/>
  <Route path="/arena" element={<ArenaView/>}/>
  <Route path="/models" element={<ModelList season={season}/>}/>
  <Route path="/models/:sha" element={<ModelDetail season={season}/>}/>
  <Route path="/checkpoints/:sha" element={<CheckpointPage season={season}/>}/>
  <Route path="/lab" element={<LabView/>}/>
  <Route path="/ladder" element={rostered(r=><Ladder season={season} roster={r} seasons={seasons.data?.seasons??[]}/>)}/>
  <Route path="/agents/:agentId" element={rostered(r=><AgentDetail season={season} roster={r}/>)}/>
  <Route path="/games/:gameId" element={rostered(r=><Replay season={season} roster={r}/>)}/>
  <Route path="/play" element={rostered(r=><Play season={season} roster={r}/>)}/>
  <Route path="*" element={<Navigate replace to="/"/>}/>
 </Routes>
 </main></div>
}
