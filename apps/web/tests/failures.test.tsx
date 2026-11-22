import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import {Failures} from '../src/views/agent/Failures'

const roster={agents:[{id:'cand',name:'Candidate'},{id:'greedy',name:'Greedy'},{id:'d2',name:'Minimax D2'}]}
const data={agent:'cand',games:100,totals:{missed_win:1,loss:2,cycle:1,ply_limit:0},failures:[
 {id:'aaaaaaaaaaaaaaaaaaaa:0',opponent:'greedy',seat:1,kinds:['missed_win','loss'],missed_immediate_win:1,steps:30,termination:'goal',available:true,opening:3},
 {id:'bbbbbbbbbbbbbbbbbbbb:1',opponent:'d2',seat:2,kinds:['loss'],missed_immediate_win:0,steps:40,termination:'goal',available:true,opening:4},
 {id:'cccccccccccccccccccc:0',opponent:'d2',seat:1,kinds:['cycle'],missed_immediate_win:0,steps:90,termination:'deterministic_cycle',available:true,opening:5}]}
afterEach(()=>{cleanup();vi.restoreAllMocks()})
it('filters recorded failures and locates the missed winning ply with the engine',async()=>{
 const urls:string[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation((input:RequestInfo|URL)=>{const url=String(input);urls.push(url)
  const body=url.includes('/missed-wins')?{plies:[{ply:12,actual:{player:1,type:'horizontal',target:[3,6]},winning:[{player:1,type:'pawn',target:[4,8]}]}]}:data
  return Promise.resolve(new Response(JSON.stringify(body)))})
 render(<MemoryRouter><Failures season="s1" agent={{id:'cand'}} roster={roster}/></MemoryRouter>)
 expect(await screen.findByText('错过立即获胜、负')).toBeTruthy()
 expect(screen.queryByText('循环和棋')).toBeTruthy()
 expect(screen.getAllByRole('row')).toHaveLength(2)
 fireEvent.click(screen.getByRole('button',{name:/^全部/}))
 fireEvent.change(screen.getByLabelText('失败样本对手'),{target:{value:'d2'}})
 expect(screen.getAllByRole('row')).toHaveLength(3)
 fireEvent.change(screen.getByLabelText('失败样本对手'),{target:{value:'all'}})
 fireEvent.click(screen.getByRole('button',{name:'定位错失步'}))
 const link=await screen.findByRole('link',{name:/第 12 步 · 实际 d7 H · 可走 e9/})
 expect(new URL(link.getAttribute('href')!,'http://x').searchParams.get('ply')).toBe('12')
 expect(urls.some(u=>u.endsWith('/missed-wins?agent=cand'))).toBe(true)
})
