import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter,Routes,Route} from 'react-router-dom'
import {ResearchPolicies} from '../src/views/agent/ResearchPolicies'
import {Projects} from '../src/views/agent/Projects'
import {JobList,JobNew} from '../src/views/agent/Jobs'

afterEach(()=>{cleanup();vi.restoreAllMocks();localStorage.clear()})
const game={game_id:'quoridor',ruleset:'standard-9x9',version:1}
const response=(body:unknown,status=200)=>Promise.resolve(new Response(JSON.stringify(body),{status}))
it('creates a project with explicit rules and preserves project task links',async()=>{
 const calls:{url:string;init?:RequestInit}[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation((url,init)=>{
  calls.push({url:String(url),init})
  return response(init?.method==='POST'?{id:'new'}:{projects:[{id:'study',title:'Study',question:'Q',game_ref:game}],games:[{ref:game,title:'Quoridor',training_backend:'quoridor_legacy'}]})
 })
 render(<MemoryRouter><Projects/></MemoryRouter>)
 expect((await screen.findByRole('link',{name:'查看任务'})).getAttribute('href')).toBe('/jobs?project=study')
 fireEvent.change(screen.getByLabelText('项目 ID'),{target:{value:'new'}})
 fireEvent.change(screen.getByLabelText('项目标题'),{target:{value:'New'}})
 fireEvent.change(screen.getByLabelText('研究问题'),{target:{value:'Does PPO improve?'}})
 fireEvent.change(screen.getByLabelText('项目写入令牌'),{target:{value:'secret'}})
 fireEvent.click(screen.getByRole('button',{name:'创建项目'}))
 await waitFor(()=>expect(calls.some(c=>c.init?.method==='POST')).toBe(true))
 const call=calls.find(c=>c.init?.method==='POST')!
 expect(JSON.parse(String(call.init?.body))).toEqual({id:'new',title:'New',question:'Does PPO improve?',game_ref:game})
 expect((call.init?.headers as Record<string,string>)['X-Platform-Token']).toBe('secret')
})
it('filters jobs on the server and keeps project on the create link',async()=>{
 const urls:string[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation(url=>{
  urls.push(String(url))
  return response(String(url).includes('/system')?{}:{jobs:[],counts:{}})
 })
 render(<MemoryRouter initialEntries={['/jobs?project=study']}><JobList/></MemoryRouter>)
 await waitFor(()=>expect(urls).toContain('/api/v1/jobs?project_id=study'))
 expect(screen.getByRole('link',{name:'新建任务'}).getAttribute('href')).toBe('/jobs/new?project=study')
})
it('submits an explicitly selected project',async()=>{
 const calls:{url:string;init?:RequestInit}[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation((url,init)=>{
  calls.push({url:String(url),init})
  if(init?.method==='POST')return response({id:'job-a'})
  return response({source_ref:'test',source_commit:'c'.repeat(40),image:'test',entrypoints:{evaluate_checkpoints:{title:'Eval',format:'json',template:{checkpoints:[]},gpu:false}}})
 })
 render(<MemoryRouter initialEntries={['/jobs/new?project=study&entrypoint=evaluate_checkpoints']}><JobNew/></MemoryRouter>)
 fireEvent.click(await screen.findByRole('button',{name:'提交任务'}))
 await waitFor(()=>expect(calls.some(c=>c.init?.method==='POST')).toBe(true))
 expect(JSON.parse(String(calls.find(c=>c.init?.method==='POST')!.init?.body)).project_id).toBe('study')
})

it('selects the compatible research entrypoint and blocks an unconfigured worker',async()=>{
 const urls:string[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation(url=>{
  urls.push(String(url))
  return response({source_ref:'test',source_commit:'c'.repeat(40),entrypoints:{
   train_openspiel:{title:'OpenSpiel',format:'json',template:{algorithm:'cfr',iterations:20},
    enabled:false,unavailable_reason:'Worker image is not configured',gpu:false}
  }})
 })
 render(<MemoryRouter initialEntries={['/jobs/new?project=poker']}><JobNew/></MemoryRouter>)
 await waitFor(()=>expect((screen.getByLabelText('任务入口') as HTMLSelectElement).value).toBe('train_openspiel'))
 expect(urls).toContain('/api/v1/jobs/entrypoints?project_id=poker')
 expect(screen.getByRole('alert').textContent).toContain('Worker image is not configured')
 expect((screen.getByRole('button',{name:'提交任务'}) as HTMLButtonElement).disabled).toBe(true)
 expect(screen.queryByRole('checkbox')).toBeNull()
})

it('lists project policy identities and links evaluation to the exact SHA',async()=>{
 const sha='a'.repeat(64),urls:string[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation(url=>{
  urls.push(String(url))
  return response({policies:[{sha256:sha,job_id:'training-1',game_ref:{game_id:'kuhn_poker'},source:{commit:'c'.repeat(40)},algorithm:'cfr',policy_kind:'average_policy',output_semantics:'action_probabilities'}]})
 })
 render(<MemoryRouter initialEntries={['/projects/poker/policies']}><Routes><Route path="/projects/:id/policies" element={<ResearchPolicies/>}/></Routes></MemoryRouter>)
 const link=await screen.findByRole('link',{name:'评测策略'})
 expect(urls).toContain('/api/v1/jobs/projects/poker/policies')
 expect(link.getAttribute('href')).toBe('/jobs/new?project=poker&entrypoint=evaluate_research_policy&policy='+sha)
 expect(screen.getByText(sha)).toBeTruthy()
 expect(screen.getByLabelText('策略下载令牌').getAttribute('type')).toBe('password')
})
it('prefills evaluation with immutable policy identity from the project link',async()=>{
 const sha='b'.repeat(64)
 vi.spyOn(globalThis,'fetch').mockImplementation(()=>response({source_ref:'test',source_commit:'c'.repeat(40),entrypoints:{
  evaluate_research_policy:{title:'Eval',format:'json',enabled:true,template:{candidate_sha:'',opponent_sha:null,pairs:100,seed:42}}
 }}))
 render(<MemoryRouter initialEntries={['/jobs/new?project=poker&entrypoint=evaluate_research_policy&policy='+sha]}><JobNew/></MemoryRouter>)
 await waitFor(()=>expect(JSON.parse((screen.getByLabelText('任务配置') as HTMLTextAreaElement).value).candidate_sha).toBe(sha))
})
