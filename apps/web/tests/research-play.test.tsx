import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter,Routes,Route} from 'react-router-dom'
import {MrJackBoard} from '../src/views/agent/MrJackBoard'
import {ResearchPlay} from '../src/views/agent/ResearchPlay'
afterEach(()=>{cleanup();vi.restoreAllMocks()})
const ref={game_id:'mr_jack',ruleset:'london-2016-base',version:1}
const obs={positions:[28,2,57,61,44,41,83,24],lights:[36,10,75,12,73,49],covered:[11,74],cordons:[11,74],orientation:4,round:1,phase:'choose',witness:true,characters:['Lestrade','Stealthy','Bert','Watson','Holmes','Smith','Goodley','Gull'],visible:[true,false,true,false,true,true,false,false],suspects:[0,1,2,3,4,5,6,7],identity:null,private_alibis:[]}
it('creates a role-scoped game and sends capability only in the request header',async()=>{
 const calls:{url:string,init?:RequestInit}[]=[]
 const view={id:'session-a',revision:0,status:'active',viewer:'detective',actor:'detective',observation:obs,legal_actions:[{id:0,label:'选择 Lestrade'}]}
 vi.spyOn(globalThis,'fetch').mockImplementation((url,init)=>{
  calls.push({url:String(url),init})
  let data:unknown=view
  if(String(url).endsWith('/jobs/projects'))data={projects:[{id:'jack',game_ref:ref,title:'Jack'}],games:[{ref,roles:['detective','jack']}]}
  else if(String(url).endsWith('/policies'))data={policies:[]}
  else if(init?.method==='POST'&&String(url).endsWith('play-v2'))data={id:'session-a',token:'private-token',watch_token:'watch-token',view}
  return Promise.resolve(new Response(JSON.stringify(data)))
 })
 render(<MemoryRouter initialEntries={['/projects/jack/play']}><Routes><Route path="/projects/:id/play" element={<ResearchPlay/>}/></Routes></MemoryRouter>)
 fireEvent.click(await screen.findByRole('button',{name:'开始研究对局'}))
 expect(await screen.findByRole('img',{name:'伦敦棋盘，格号与动作列表对应'})).toBeTruthy()
 fireEvent.click(await screen.findByRole('button',{name:'选择 Lestrade'}))
 await waitFor(()=>expect(calls.some(c=>c.url.endsWith('/step'))).toBe(true))
 const action=calls.find(c=>c.url.endsWith('/step'))!
 expect(action.init?.headers).toMatchObject({'X-Game-Token':'private-token'})
 expect(JSON.parse(String(action.init?.body))).toEqual({revision:0,action:0})
 expect(calls.every(c=>!c.url.includes('private-token'))).toBe(true)
 expect(screen.getByRole('link',{name:'打开观战视图'}).getAttribute('href')).toContain('#token=watch-token')
})
it('renders spectator projection without action controls',async()=>{
 vi.spyOn(globalThis,'fetch').mockImplementation(url=>Promise.resolve(new Response(JSON.stringify(
  String(url).endsWith('/jobs/projects')?{projects:[{id:'jack',game_ref:ref}],games:[{ref,roles:['detective','jack']}]}:
  String(url).endsWith('/policies')?{policies:[]}:
  {id:'a',revision:2,status:'active',viewer:'watch',actor:'jack',observation:obs,legal_actions:[]}))))
 render(<MemoryRouter initialEntries={['/projects/jack/play?session=a#token=watch']}><Routes><Route path="/projects/:id/play" element={<ResearchPlay/>}/></Routes></MemoryRouter>)
 await screen.findByRole('button',{name:'查看本视角回放'})
 expect(screen.queryByRole('button',{name:'执行对手动作'})).toBeNull()
 expect(screen.queryByRole('button',{name:'关闭对局'})).toBeNull()
})


it('allows only server-provided board moves and supports keyboard activation',()=>{
 const act=vi.fn()
 const view=render(<MrJackBoard observation={{...obs,phase:'move'}} legalActions={[{id:105,label:'移动到 5'}]} onAction={act}/>)
 fireEvent.click(screen.getByRole('button',{name:'移动到 5'}))
 fireEvent.keyDown(screen.getByRole('button',{name:'移动到 5'}),{key:'Enter'})
 expect(act.mock.calls).toEqual([[105],[105]])
 expect(screen.queryByRole('button',{name:'移动到 6'})).toBeNull()
 view.rerender(<MrJackBoard observation={obs} legalActions={[{id:105}]}/>)
 expect(screen.queryByRole('button',{name:'移动到 5'})).toBeNull()
})
