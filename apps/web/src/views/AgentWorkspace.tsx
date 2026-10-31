import {useEffect, useState} from 'react'
import {Link, Navigate, Route, Routes, useLocation, useSearchParams} from 'react-router-dom'
import {Grid3X3, Trophy, Swords, Sun, Moon} from 'lucide-react'
import '../App.css'
import './AgentWorkspace.css'
import './Refinement.css'
import './Polish.css'
import {TrainingWorkspace} from './TrainingWorkspace'
import {useData, enc, route, ErrorBox} from './agent/shared'
import {Ladder} from './agent/Ladder'
import {AgentDetail} from './agent/AgentDetail'
import {CheckpointPage} from './agent/CheckpointPage'
import {Replay} from './agent/Replay'
import {Play} from './agent/Play'
export {evidence, route, Participant} from './agent/shared'
export {StepAnalysis} from './agent/StepAnalysis'

export default function AgentWorkspace(){
 const [params]=useSearchParams();const seasons=useData('/seasons');const season=params.get('season')??seasons.data?.seasons?.[0]?.id??''
 const roster=useData(season?'/seasons/'+enc(season):null)
 const location=useLocation()
 const [theme,setTheme]=useState(()=>localStorage.getItem('quoridor-workbench-theme')==='light'?'light':'dark')
 useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem('quoridor-workbench-theme',theme)},[theme])
 return <div className="app agent-workspace"><header className="aw-header"><Link to={route('/ladder',season)} className="aw-brand"><Grid3X3 aria-hidden="true"/>Quoridor</Link><nav aria-label="主要入口"><Link to={route('/ladder',season)} aria-current={location.pathname!=='/play'?'page':undefined}><Trophy aria-hidden="true"/>天梯</Link><Link to={route('/play',season)} aria-current={location.pathname==='/play'?'page':undefined}><Swords aria-hidden="true"/>对战</Link></nav><span className="aw-preview">独立预览</span><button className="aw-theme-toggle" onClick={()=>setTheme(t=>t==='light'?'dark':'light')}>{theme==='light'?<Moon aria-hidden="true"/>:<Sun aria-hidden="true"/>}{theme==='light'?'深色':'浅色'}</button></header><main className="aw-main">
 <ErrorBox text={seasons.error||roster.error} retry={seasons.error?seasons.retry:roster.retry}/>
 {!seasons.data&&!seasons.error&&<p role="status">加载评测批次…</p>}
 {seasons.data?.seasons?.length===0&&<p>暂无已完成的天梯报告</p>}
 {season&&!roster.data&&!roster.error&&<p role="status">加载 Agent…</p>}
 {roster.data&&<Routes><Route path="/" element={<Navigate replace to={route('/ladder',season)}/>}/><Route path="/ladder" element={<Ladder season={season} roster={roster.data} seasons={seasons.data?.seasons??[]}/>}/><Route path="/agents/:agentId" element={<AgentDetail season={season} roster={roster.data}/>}/><Route path="/games/:gameId" element={<Replay season={season} roster={roster.data}/>}/><Route path="/play" element={<Play season={season} roster={roster.data}/>}/><Route path="/checkpoints/:sha" element={<CheckpointPage season={season}/>}/><Route path="/training" element={<TrainingWorkspace season={season}/>}/><Route path="*" element={<Navigate replace to={route('/ladder',season)}/>}/></Routes>}
 </main></div>
}
