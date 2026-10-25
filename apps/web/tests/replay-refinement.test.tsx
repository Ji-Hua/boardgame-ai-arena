import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {act,cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import AgentWorkspace,{StepAnalysis} from '../src/views/AgentWorkspace'
import * as lab from '../src/lab'
const agents=[{id:'greedy',name:'Greedy',kind:'heuristic',playable:true,spec:{algo:{type:'greedy'},policy:{type:'top_k',k:1}}},{id:'cnn',name:'CNN',kind:'neural',sha256:'a'.repeat(64),playable:true,spec:{algo:{type:'dqn',params:{}},policy:{type:'top_k',k:1}}}]
const positions=[lab.opening,{...lab.opening,turn:2 as const},{...lab.opening,pawns:[[4,1],[4,8]]}]
function fixture(){vi.stubGlobal('fetch',vi.fn(async(url:string)=>({ok:true,json:async()=>url.endsWith('/seasons')?{seasons:[{id:'s'}]}:url.includes('/replay/')?{p1:'cnn',p2:'greedy',positions,actions:[{player:1,type:'pawn',target:[4,1]},{player:2,type:'pawn',target:[4,7]}],decision_milliseconds:{},winner:'cnn',termination:'goal'}:{agents,matrix:{},protocol:'paired'}})))}
afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks();vi.unstubAllGlobals()})
it('supports transport, stops at end, keeps identity and ignores shortcuts in inputs',async()=>{
 fixture();render(<MemoryRouter initialEntries={['/games/g?season=s&ply=1']}><AgentWorkspace/></MemoryRouter>)
 await screen.findByRole('heading',{name:'对局回放'})
 expect(screen.getByLabelText('P1 身份').textContent).toContain('CNN')
 expect(screen.getByLabelText('P2 身份').textContent).toContain('启发式')
 fireEvent.click(screen.getByRole('button',{name:'回到开头'}))
 expect((screen.getByLabelText('跳转步数') as HTMLInputElement).value).toBe('0')
 vi.useFakeTimers();fireEvent.click(screen.getByRole('button',{name:'播放回放'}))
 await act(async()=>{vi.advanceTimersByTime(1000)})
 expect((screen.getByLabelText('跳转步数') as HTMLInputElement).value).toBe('1')
 await act(async()=>{vi.advanceTimersByTime(1000)})
 expect(screen.getByRole('button',{name:'播放回放'})).toBeTruthy()
 expect((screen.getByLabelText('跳转步数') as HTMLInputElement).value).toBe('2')
 fireEvent.keyDown(screen.getByLabelText('跳转步数'),{key:'Home'})
 expect((screen.getByLabelText('跳转步数') as HTMLInputElement).value).toBe('2')
 fireEvent.keyDown(window,{key:'Home'})
 expect((screen.getByLabelText('跳转步数') as HTMLInputElement).value).toBe('0')
})
it('keeps selected analyst across seeking and terminal position',async()=>{
 fixture();render(<MemoryRouter initialEntries={['/games/g?season=s']}><AgentWorkspace/></MemoryRouter>)
 await screen.findByRole('button',{name:'对照分析'})
 fireEvent.click(screen.getByRole('button',{name:'对照分析'}))
 fireEvent.change(screen.getByLabelText('分析 Agent'),{target:{value:'greedy'}})
 fireEvent.click(screen.getByRole('button',{name:'跳到结束'}))
 fireEvent.click(screen.getByRole('button',{name:'回到开头'}))
 expect((screen.getByLabelText('分析 Agent') as HTMLSelectElement).value).toBe('greedy')
})
it('invalidates in-flight results when the position changes without changing analyst',async()=>{
 let resolve:(value:any)=>void=()=>{}
 vi.spyOn(lab,'labRequest').mockImplementation(()=>new Promise(r=>{resolve=r}))
 const onBranch=vi.fn();const view=render(<StepAnalysis position={lab.opening} actor={agents[0]} roster={{agents}} onBranch={onBranch}/>)
 fireEvent.click(screen.getByRole('button',{name:'对照分析'}))
 fireEvent.change(screen.getByLabelText('分析 Agent'),{target:{value:'cnn'}})
 fireEvent.click(screen.getByRole('button',{name:'运行分析'}))
 view.rerender(<StepAnalysis position={{...lab.opening,turn:2}} actor={agents[0]} roster={{agents}} onBranch={onBranch}/>)
 await act(async()=>resolve({candidates:[{id:'stale',label:'STALE',value:9}]}))
 await waitFor(()=>expect(screen.queryByText('STALE')).toBeNull())
 expect((screen.getByLabelText('分析 Agent') as HTMLSelectElement).value).toBe('cnn')
 expect(screen.getByText('当前局面未分析')).toBeTruthy()
})
it('shows the selected model and locally controlled player before starting',async()=>{
 fixture();render(<MemoryRouter initialEntries={['/play?season=s&p1=cnn&p2=human']}><AgentWorkspace/></MemoryRouter>)
 await screen.findByLabelText('P1 身份')
 expect(screen.getByLabelText('P1 身份').textContent).toContain('checkpoint aaaaaaaaaaaa')
 expect(screen.getByLabelText('P2 身份').textContent).toContain('你控制此方')
 fireEvent.click(screen.getByRole('button',{name:'交换'}))
 expect(screen.getByLabelText('P1 身份').textContent).toContain('你控制此方')
 expect(screen.getByLabelText('P2 身份').textContent).toContain('CNN')
})
