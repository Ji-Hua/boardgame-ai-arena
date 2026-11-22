import {useEffect, useState} from 'react'

export type Obj = Record<string, any>
export async function request(url:string, options:RequestInit={}) {
  const response=await fetch(url,options)
  const data=await response.json()
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail??data))
  return data
}
export function evidence(path:string, options:RequestInit={}) {return request('/api/v1'+path,options)}
function useRequest(url:string|null){
 const [data,setData]=useState<Obj|null>(null),[error,setError]=useState(''),[retry,setRetry]=useState(0)
 useEffect(()=>{setData(null);setError('');if(!url)return;const c=new AbortController();request(url,{signal:c.signal}).then(d=>{if(!c.signal.aborted)setData(d)}).catch(e=>{if(!c.signal.aborted)setError(String(e))});return()=>c.abort()},[url,retry])
 return {data,error,retry:()=>setRetry(v=>v+1)}
}
export const useData=(path:string|null)=>useRequest(path?'/api/v1'+path:null)
export const useV1=(path:string|null)=>useRequest(path?'/api/v1'+path:null)
export const useRuns=()=>useRequest('/api/v1/research/runs')
/** Epoch seconds or an ISO timestamp, shown in local time. */
export const when=(t?:number|string|null)=>{const ms=typeof t==='number'?t*1000:typeof t==='string'?Date.parse(t):NaN;return Number.isFinite(ms)?new Date(ms).toLocaleString('zh-CN',{hour12:false}):'—'}
export const STATES:[string,string][]=[['analyzable','可分析'],['playable','可对战'],['research_playable','研究对战'],['ranked','已上天梯'],['deployed','已上线']]
export function States({model}:{model:Obj}){return <span className="pf-states">{STATES.map(([k,label])=><span key={k} className={'pf-state '+(model[k]?'on':'off')} title={(model.reasons?.[k]??[]).join(' · ')}>{label}</span>)}</span>}
export const enc=encodeURIComponent
export const kind=(a:Obj)=>({neural:'神经网络',heuristic:'启发式',random:'随机',human:'玩家'}[a.kind as string]??a.kind)
export const fmt=(x:any,n=2)=>typeof x==='number'?x.toFixed(n):'—'
export const pct=(x:any)=>typeof x==='number'?(100*x).toFixed(1)+'%':'—'
export const short=(s?:string)=>s?s.slice(0,12):'—'
export function route(path:string,season:string,values:Record<string,string|number|undefined>={}){const q=new URLSearchParams({season});Object.entries(values).forEach(([k,v])=>{if(v!==undefined)q.set(k,String(v))});return path+'?'+q}
export function ErrorBox({text,retry}:{text:string;retry?:()=>void}){return text?<div className="error" role="alert">{text}{retry&&<button onClick={retry}>重试</button>}</div>:null}
export function Json({value}:{value:unknown}){return <pre className="aw-json">{JSON.stringify(value,null,2)}</pre>}
export function Tag({agent}:{agent:Obj}){return <span className={'aw-tag '+agent.kind}>{kind(agent)}</span>}
export function agentVersion(a:Obj){const sha=a.actual_sha256??a.sha256;return sha?'checkpoint '+short(sha):a.spec?.algo?.type==='minimax'?'depth '+a.spec.algo.params?.depth:a.spec?.algo?.type??'版本未记录'}
export function notation(a:Obj){return 'abcdefghi'[a.target[0]]+(a.target[1]+1)+({horizontal:' H',vertical:' V',pawn:''}[a.type as string]??'')}

export function Participant({agent,seat,active=false,winner}:{agent:Obj;seat:number;active?:boolean;winner?:number|null}){
 const sha=agent.actual_sha256??agent.sha256
 return <div className={'aw-participant aw-p'+seat} aria-label={'P'+seat+' 身份'}>
  <span className="aw-seat-dot">P{seat}</span><div className="aw-participant-name"><strong>{agent.kind==='human'?'本机玩家':agent.name??agent.id}</strong><small>{kind(agent)} · {agent.kind==='human'?'你控制此方':agentVersion(agent)}</small></div>
  <span className="aw-turn">{winner===seat?'获胜':active?'行棋中':''}</span>
  {agent.kind!=='human'&&<details><summary>版本</summary><dl><dt>Agent</dt><dd>{agent.id}</dd><dt>策略</dt><dd>{agent.spec?.policy?.type??'—'} · k={agent.spec?.policy?.k??'—'}</dd>{sha&&<><dt>Checkpoint</dt><dd className="aw-hash">{sha}</dd></>}</dl><Json value={agent.spec??agent}/></details>}
 </div>
}
