import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import {PromotionPanel} from '../src/views/agent/Promotion'

const sha='a'.repeat(64)
const gate={season:'job-gate',gate:'gate-v1',rank:1,roster_size:7,elo:1610,score:.7,ci95:[.5,.9],wins:7,draws:0,losses:3,published:1761900000,opponents:{greedy:{games:10,wins:7,draws:0,losses:3}}}
afterEach(()=>{cleanup();vi.restoreAllMocks();localStorage.clear()})

it('promotes a gate candidate with a reason and the token, then shows the decision',async()=>{
 localStorage.setItem('quoridor-platform-token','secret')
 let deployed:unknown=null;const posts:RequestInit[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation((input:RequestInfo|URL,init?:RequestInit)=>{
  if(init?.method==='POST'){posts.push(init);deployed={time:1761900100,season:'job-gate',gate:'gate-v1',note:'beats the gate',evidence:gate};return Promise.resolve(new Response(JSON.stringify(deployed)))}
  return Promise.resolve(new Response(JSON.stringify({sha256:sha,deployed,gates:[gate]})))})
 const changed=vi.fn()
 render(<MemoryRouter><PromotionPanel sha={sha} onChange={changed}/></MemoryRouter>)
 expect(await screen.findByText('greedy · 7 / 0 / 3')).toBeTruthy()
 const confirm=screen.getByRole('button',{name:'确认上线'}) as HTMLButtonElement
 expect(confirm.disabled).toBe(true)
 fireEvent.change(screen.getByLabelText('上线理由'),{target:{value:'beats the gate'}})
 fireEvent.click(confirm)
 expect(await screen.findByText('已上线')).toBeTruthy()
 expect(JSON.parse(posts[0].body as string)).toEqual({sha256:sha,note:'beats the gate',season:'job-gate'})
 expect((posts[0].headers as Record<string,string>)['X-Platform-Token']).toBe('secret')
 expect(changed).toHaveBeenCalled()
 expect(screen.getByRole('button',{name:'确认撤下'})).toBeTruthy()
})

it('points to a gate Job when there is no gate evidence',async()=>{
 vi.spyOn(globalThis,'fetch').mockImplementation(()=>Promise.resolve(new Response(JSON.stringify({sha256:sha,deployed:null,gates:[]}))))
 render(<MemoryRouter><PromotionPanel sha={sha} onChange={()=>{}}/></MemoryRouter>)
 expect((await screen.findByRole('link',{name:'新建门槛评测任务'})).getAttribute('href')).toBe('/jobs/new?entrypoint=run_arena_matrix&sha='+sha)
 expect(screen.queryByRole('button',{name:'确认上线'})).toBeNull()
})
