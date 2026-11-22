import {useEffect, useState} from 'react'
import {Link, useParams, useSearchParams} from 'react-router-dom'
import {positionBoard, type Position} from '../../lab'
import {EvalReplayBoard} from '../../components/EvalReplayBoard'
import {EvidenceViewer} from '../EvidenceViewer'
import {type Obj, useData, enc, fmt, route, ErrorBox, Participant, notation} from './shared'
import {StepAnalysis} from './StepAnalysis'

export function Replay({season,roster}:{season:string;roster:Obj}){
 const {gameId=''}=useParams();const [params,setParams]=useSearchParams();const r=useData('/seasons/'+enc(season)+'/replay/'+enc(gameId))
 const [derived,setDerived]=useState<Position|null>(null),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[panel,setPanel]=useState('analysis')
 const total=r.data?.actions.length??0
 const ply=Math.max(0,Math.min(Math.trunc(Number(params.get('ply'))||0),total));const position=r.data?.positions[ply]
 const participant=(id:string)=>roster.agents.find((a:Obj)=>a.id===id)??{id,name:id,kind:'unknown'}
 const actor=position?participant(r.data?.[position.turn===1?'p1':'p2']):null
 const actual=r.data?.actions[ply];const occurrence=r.data?.actions.slice(0,ply).filter((a:Obj)=>a.player===position?.turn).length??0
 const latency=actor?r.data?.decision_milliseconds?.[actor.id]?.[occurrence]:undefined
 function seek(n:number,pause=true){if(pause)setPlaying(false);const p=new URLSearchParams(params);p.set('ply',String(Math.max(0,Math.min(n,total))));setParams(p,{replace:true})}
 function toggle(){if(!total)return;if(ply===total)seek(0,false);setPlaying(v=>!v)}
 useEffect(()=>{setDerived(null)},[ply,gameId])
 useEffect(()=>{setPlaying(false)},[gameId,season])
 useEffect(()=>{if(!playing||!r.data)return;if(ply>=total){setPlaying(false);return}const t=setTimeout(()=>seek(ply+1,false),1000/speed);return()=>clearTimeout(t)},[playing,ply,total,speed,r.data])
 useEffect(()=>{function key(e:KeyboardEvent){const el=e.target as HTMLElement;if(e.altKey||e.ctrlKey||e.metaKey||el.closest?.('input,select,textarea,button,a,[contenteditable="true"],[role="tab"]'))return;if(['ArrowLeft','ArrowRight','Home','End',' '].includes(e.key)){e.preventDefault();if(e.key===' ')toggle();else seek(e.key==='Home'?0:e.key==='End'?total:ply+(e.key==='ArrowLeft'?-1:1))}}window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key)},[ply,total,playing,params])
 return <><div className="aw-breadcrumb"><Link to={route('/ladder',season)}>天梯</Link> / <Link to={route('/agents/'+enc(params.get('agent')??r.data?.p1??''),season,{opponent:params.get('opponent')??r.data?.p2})}>交手记录</Link> / {gameId.slice(0,12)}</div><ErrorBox text={r.error} retry={r.retry}/>{!r.data&&!r.error&&<p role="status">验证原始棋谱…</p>}{r.data&&position&&<>
 <div className="aw-toolbar"><h1>对局回放</h1><small>{r.data.winner?participant(r.data.winner).name+' 胜':'平局'} · {r.data.termination}</small><Link to={route('/play',season,{p1:r.data.p1,p2:r.data.p2})}>双方再战</Link></div>
 <div className="aw-replay aw-workbench"><section className="aw-board-column">
 <Participant agent={participant(r.data.p2)} seat={2} active={!derived&&!!actual&&position.turn===2}/>
 <EvalReplayBoard state={positionBoard(derived??position,r.data.actions.slice(0,ply),derived?[]:r.data.positions.slice(0,ply+1))} size={430} onWallInspect={n=>{setDerived(null);setPanel('analysis');seek(n)}}/>
 <Participant agent={participant(r.data.p1)} seat={1} active={!derived&&!!actual&&position.turn===1}/>
 <div className="aw-transport" aria-label="回放控制">
 <input type="range" aria-label="回放步数" min={0} max={total} value={ply} onChange={e=>seek(+e.target.value)}/>
 <div className="aw-toolbar"><button aria-label="回到开头" title="Home" disabled={ply===0} onClick={()=>seek(0)}>⏮</button><button aria-label="上一步" title="←" disabled={ply===0} onClick={()=>seek(ply-1)}>◀</button><button aria-label={playing?'暂停回放':'播放回放'} title="空格" disabled={!total} onClick={toggle}>{playing?'暂停':'播放'}</button><button aria-label="下一步" title="→" disabled={ply===total} onClick={()=>seek(ply+1)}>▶</button><button aria-label="跳到结束" title="End" disabled={ply===total} onClick={()=>seek(total)}>⏭</button><select aria-label="播放速度" value={speed} onChange={e=>setSpeed(+e.target.value)}>{[0.5,1,2,4].map(v=><option key={v} value={v}>{v}×</option>)}</select><label><input aria-label="跳转步数" type="number" min={0} max={total} value={ply} onChange={e=>seek(Math.trunc(+e.target.value))}/> / {total}</label></div>
 <div className="aw-toolbar"><small>{derived?'候选分支 · 原始第 '+ply+' 步':'原始回放'}</small>{derived&&<button onClick={()=>setDerived(null)}>返回原局面</button>}</div></div>
 </section><section className="aw-panel aw-inspector">
 <div className="aw-tabs" role="tablist" aria-label="回放视图">{[['analysis','逐步分析'],['moves','棋谱'],['source','证据']].map(([id,label])=><button key={id} role="tab" aria-selected={panel===id} onClick={()=>setPanel(id)}>{label}</button>)}</div>
 <div className={'aw-inspector-body '+(panel==='source'?'aw-evidence-body':'')}><div hidden={panel!=='analysis'}><h2>第 {ply} 步 · P{position.turn}</h2><div className="aw-record"><strong>{actor?.name}</strong> · 实际动作 {actual?notation(actual):'终局'}<small>耗时 {fmt(latency)} ms · 候选分数 / 搜索树：未记录</small></div>
 <div hidden={!actual}><StepAnalysis key={gameId} position={position} actor={actor} roster={roster} onBranch={setDerived} onAnalyze={()=>setPlaying(false)} actual={actual} autoAnalyze={!playing&&!!actual} suspended={playing||!actual}/></div>{!actual&&<p>对局已结束</p>}</div>
 {panel==='moves'&&<div className="aw-move-list">{r.data.actions.map((a:Obj,i:number)=><button key={i} aria-current={ply===i?'step':undefined} onClick={()=>seek(i)}><span>{i+1}</span><span>P{a.player}</span><strong>{notation(a)}</strong></button>)}</div>}
 {panel==='source'&&<EvidenceViewer filename={gameId.replace(':','-')+'-evidence.json'} value={{replay:r.data,evaluation:{season,protocol:roster.protocol,manifest_sha256:roster.manifest_sha256},participants:[participant(r.data.p1),participant(r.data.p2)],view:{ply,position,branch_preview:derived}}}/>}
 </div></section></div></>}</>
}
