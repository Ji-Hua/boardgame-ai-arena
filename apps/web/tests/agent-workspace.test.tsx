import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor,within} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import AgentWorkspace,{StepAnalysis} from '../src/views/AgentWorkspace'
import * as lab from '../src/lab'
const agents=[{id:'greedy',name:'Greedy',kind:'heuristic',spec:{algo:{type:'greedy'},policy:{type:'top_k',k:1}},ranking:{rank:1,elo:1500,wins:4,draws:0,losses:1}},{id:'cnn',name:'CNN',sha256:'a'.repeat(64),kind:'neural',spec:{algo:{type:'dqn',params:{}},policy:{type:'top_k',k:1}},ranking:null}]
afterEach(()=>{cleanup();vi.unstubAllGlobals()})
it('keeps Ladder and Play independent and uses reported ratings only',async()=>{
 vi.stubGlobal('fetch',vi.fn(async(url:string)=>({ok:true,json:async()=>url.endsWith('/seasons')?{seasons:[{id:'s'}]}:{agents,matrix:{},protocol:'paired'}})))
 render(<MemoryRouter initialEntries={['/ladder?season=s']}><AgentWorkspace/></MemoryRouter>)
 expect(await screen.findByRole('heading',{name:'天梯榜'})).toBeTruthy()
 expect(screen.getByText('1500')).toBeTruthy()
 fireEvent.click(within(screen.getByRole('navigation',{name:'主要入口'})).getByRole('link',{name:'对战'}))
 expect(await screen.findByRole('heading',{name:'对战'})).toBeTruthy()
 expect(screen.getByLabelText('P1 对战方')).toBeTruthy()
 expect(screen.getByLabelText('P2 对战方')).toBeTruthy()
})
it('defaults to the newest non-gate season and labels gate seasons',async()=>{
 const urls:string[]=[]
 vi.stubGlobal('fetch',vi.fn(async(url:string)=>{urls.push(url);return {ok:true,json:async()=>url.endsWith('/seasons')?{seasons:[{id:'gate-job',gate:'gate-v1'},{id:'full'}]}:{agents,matrix:{},protocol:'paired'}}}))
 render(<MemoryRouter initialEntries={['/ladder']}><AgentWorkspace/></MemoryRouter>)
 expect(await screen.findByRole('heading',{name:'天梯榜'})).toBeTruthy()
 expect(urls).toContain('/api/v1/seasons/full')
 expect(urls).not.toContain('/api/v1/seasons/gate-job')
 expect(screen.getByRole('option',{name:'gate-job · 门槛 gate-v1'})).toBeTruthy()
})
it('invalidates old analysis when Agent changes and never auto-runs',async()=>{
 let resolve:(v:any)=>void=()=>{}
 const request=vi.spyOn(lab,'labRequest').mockImplementation(()=>new Promise(r=>{resolve=r}))
 render(<StepAnalysis position={lab.opening} actor={agents[0]} roster={{agents}} onBranch={()=>{}}/>)
 expect(request).not.toHaveBeenCalled()
 fireEvent.click(screen.getByRole('button',{name:'运行分析'}))
 fireEvent.click(screen.getByRole('button',{name:'对照分析'}))
 fireEvent.change(screen.getByLabelText('分析 Agent'),{target:{value:'cnn'}})
 resolve({candidates:[{id:'stale',label:'STALE',value:999}]})
 await waitFor(()=>expect(screen.queryByText('STALE')).toBeNull())
 expect(screen.getByRole('button',{name:'运行分析'}).hasAttribute('disabled')).toBe(false)
})
