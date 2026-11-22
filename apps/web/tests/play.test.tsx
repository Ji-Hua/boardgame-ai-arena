import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import {Play} from '../src/views/agent/Play'

const roster={agents:[{id:'greedy',name:'Greedy',kind:'heuristic',playable:true,spec:{algo:{type:'greedy'},policy:{type:'top_k',k:1}}}]}
const position={pawns:[[4,0],[4,8]],walls_remaining:[10,10],walls:[],turn:1}
const game=(extra:Record<string,unknown>)=>({id:'g1',season:'s',position,positions:[],actions:[],revision:0,winner:null,limit:false,
 legal_actions:[{player:1,type:'pawn',target:[4,1]}],identities:[{id:'human',name:'玩家',kind:'human'},{id:'greedy',name:'Greedy',kind:'heuristic'}],
 status:'active',listed:false,role:'player',...extra})
function mock(routes:Record<string,(init?:RequestInit)=>unknown>){
 const calls:{url:string,init?:RequestInit}[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation((input:RequestInfo|URL,init?:RequestInit)=>{const url=String(input);calls.push({url,init})
  const route=routes[url];return Promise.resolve(new Response(JSON.stringify(route?route(init):{detail:'missing'}),{status:route?200:404}))})
 return calls
}
afterEach(()=>{cleanup();vi.restoreAllMocks();sessionStorage.clear()})

it('lists published finished games before a game starts',async()=>{
 mock({'/api/v1/play':()=>({games:[{id:'g9',season:'s',winner:2,plies:41,updated:1761900000,players:[{id:'human',kind:'human'},{id:'greedy',name:'Greedy',kind:'heuristic'}]}]})})
 render(<MemoryRouter initialEntries={['/play']}><Play season="s" roster={roster}/></MemoryRouter>)
 const link=await screen.findByRole('link',{name:'玩家 × Greedy'})
 expect(new URL(link.getAttribute('href')!,'http://x').searchParams.get('game')).toBe('g9')
 expect(screen.getByText('P2 胜')).toBeTruthy()
})

it('opens a watch link read-only with the watch token',async()=>{
 const calls=mock({'/api/v1/play/g1':()=>game({role:'watcher',legal_actions:[]})})
 render(<MemoryRouter initialEntries={['/play?game=g1&watch=w-token']}><Play season="s" roster={roster}/></MemoryRouter>)
 expect(await screen.findByText('观战 · 只读')).toBeTruthy()
 expect((calls[0].init?.headers as Record<string,string>)['X-Game-Token']).toBe('w-token')
 expect(screen.queryByRole('button',{name:'关闭对局'})).toBeNull()
 expect(screen.queryByLabelText('合法动作')).toBeNull()
 expect(screen.queryByRole('button',{name:'复制观战链接'})).toBeNull()
})

it('lets the player publish a finished game',async()=>{
 sessionStorage.setItem('agent-preview:g1','p-token')
 const calls=mock({'/api/v1/play/g1':()=>game({status:'finished',winner:1}),'/api/v1/play/g1/listing':()=>game({status:'finished',winner:1,listed:true})})
 render(<MemoryRouter initialEntries={['/play?game=g1']}><Play season="s" roster={roster}/></MemoryRouter>)
 fireEvent.click(await screen.findByLabelText('加入公开对局列表'))
 await waitFor(()=>expect((screen.getByLabelText('加入公开对局列表') as HTMLInputElement).checked).toBe(true))
 const put=calls.find(c=>c.url==='/api/v1/play/g1/listing')!
 expect(put.init?.method).toBe('PUT')
 expect((put.init?.headers as Record<string,string>)['X-Game-Token']).toBe('p-token')
 expect(JSON.parse(String(put.init?.body))).toEqual({listed:true})
 expect(screen.getByRole('button',{name:'复制观战链接'})).toBeTruthy()
})
