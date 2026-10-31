import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter,Route,Routes} from 'react-router-dom'
import {JobList,JobNew,JobDetail} from '../src/views/agent/Jobs'

const commit='c'.repeat(40)
const job={id:'job-1',name:'smoke',entrypoint:'train_dqn',experiment:null,status:'running',created:1761900000,started:1761900000,finished:null,exit_code:null,error:null,cancel_requested:false,registered:[],spec:{entrypoint:'train_dqn',resources:{gpu:true,cpus:6,memory_gb:24,timeout_h:12},source:{ref:'feature/x',commit},image:'img'}}
const reply=(body:unknown,status=200)=>Promise.resolve(new Response(typeof body==='string'?body:JSON.stringify(body),{status}))
function mock(routes:Record<string,(init?:RequestInit)=>unknown>){
 const calls:{url:string;init?:RequestInit}[]=[]
 vi.spyOn(globalThis,'fetch').mockImplementation((input:RequestInfo|URL,init?:RequestInit)=>{const url=String(input);calls.push({url,init});const handler=routes[url];return handler?reply(handler(init) as any):reply({detail:'not mocked'},404)})
 return calls
}
afterEach(()=>{cleanup();vi.restoreAllMocks();localStorage.clear()})
const at=(path:string)=>render(<MemoryRouter initialEntries={[path]}><Routes><Route path="/jobs" element={<JobList/>}/><Route path="/jobs/new" element={<JobNew/>}/><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>)

it('lists Jobs with status, pinned commit and GPU state',async()=>{
 mock({'/api/v1/jobs':()=>({jobs:[job],counts:{running:1}}),'/api/v1/jobs/system':()=>({gpu:{available:true,gpus:[{name:'RTX',memory_used_mib:100,memory_total_mib:12288,utilization:5}]},writes_enabled:true})})
 at('/jobs')
 expect(await screen.findByRole('link',{name:'smoke'})).toBeTruthy()
 expect(screen.getByText('运行中').className).toContain('running')
 expect(screen.getByText(/GPU · c{8}/)).toBeTruthy()
 expect(await screen.findByText(/RTX · 显存 100\/12288 MiB/)).toBeTruthy()
})

it('submits a templated training Job with the saved token and opens it',async()=>{
 const calls=mock({'/api/v1/jobs/entrypoints':()=>({source_ref:'feature/x',source_commit:commit,image:'img',entrypoints:{train_dqn:{title:'DQN',gpu:true,format:'yaml',templates:['smoke.yaml'],description:'d'},evaluate_checkpoints:{title:'Eval',gpu:false,format:'json',template:{checkpoints:[],opponents:[]},description:'e'}}}),
  '/api/v1/jobs/templates/train_dqn/smoke.yaml':()=>'episodes: 10\n','/api/v1/jobs':()=>({...job,status:'queued'}),'/api/v1/jobs/job-1':()=>({...job,events:[],run_id:'job-1',run_files:[]}),'/api/v1/jobs/job-1/metrics':()=>({metrics:[],evaluations:[]})})
 at('/jobs/new')
 fireEvent.change(await screen.findByLabelText('配置模板'),{target:{value:'smoke.yaml'}})
 await waitFor(()=>expect((screen.getByLabelText('任务配置') as HTMLTextAreaElement).value).toBe('episodes: 10\n'))
 fireEvent.change(screen.getByLabelText('写入令牌'),{target:{value:'secret'}})
 fireEvent.change(screen.getByLabelText('CPU'),{target:{value:'4'}})
 fireEvent.click(screen.getByRole('button',{name:'提交任务'}))
 expect(await screen.findByRole('heading',{name:'smoke'})).toBeTruthy()
 const post=calls.find(c=>c.url==='/api/v1/jobs'&&c.init?.method==='POST')!
 expect((post.init!.headers as Record<string,string>)['X-Platform-Token']).toBe('secret')
 expect(JSON.parse(post.init!.body as string)).toMatchObject({entrypoint:'train_dqn',config:'episodes: 10\n',resources:{cpus:4}})
 expect(localStorage.getItem('quoridor-platform-token')).toBe('secret')
})

it('prefills a gate Job and links a published season',async()=>{
 const sha='a'.repeat(64)
 mock({'/api/v1/jobs/entrypoints':()=>({source_ref:'feature/x',source_commit:commit,image:'img',entrypoints:{run_arena_matrix:{title:'Gate',kind:'evaluate',gpu:false,format:'json',template:{candidates:[],gate:'gate-v1'},description:'g'}}}),
  '/api/v1/jobs/job-2':()=>({...job,id:'job-2',status:'succeeded',events:[],run_id:'job-2',run_files:['arena'],season:'job-2',diagnoses:[]}),'/api/v1/jobs/job-2/metrics':()=>({metrics:[],evaluations:[]})})
 at('/jobs/new?entrypoint=run_arena_matrix&sha='+sha)
 await waitFor(()=>expect(JSON.parse((screen.getByLabelText('任务配置') as HTMLTextAreaElement).value).candidates).toEqual([sha]))
 expect(screen.queryByRole('checkbox')).toBeNull()
 cleanup()
 at('/jobs/job-2')
 expect((await screen.findByRole('link',{name:'天梯评测批次 job-2'})).getAttribute('href')).toBe('/ladder?season=job-2')
})

it('sends the gate follow-up for training Jobs',async()=>{
 const calls=mock({'/api/v1/jobs/entrypoints':()=>({source_ref:'feature/x',source_commit:commit,image:'img',entrypoints:{train_dqn:{title:'DQN',kind:'train',gpu:true,format:'yaml',templates:[],description:'d'}}}),'/api/v1/jobs':()=>({...job,status:'queued'})})
 at('/jobs/new')
 fireEvent.change(await screen.findByLabelText('任务配置'),{target:{value:'episodes: 1\n'}})
 fireEvent.click(screen.getByRole('checkbox'))
 fireEvent.click(screen.getByRole('button',{name:'提交任务'}))
 await waitFor(()=>expect(calls.some(c=>c.init?.method==='POST')).toBe(true))
 expect(JSON.parse(calls.find(c=>c.init?.method==='POST')!.init!.body as string).follow_up).toEqual({gate:'gate-v1'})
})

it('prefills an evaluation Job with the model SHA',async()=>{
 const sha='a'.repeat(64)
 mock({'/api/v1/jobs/entrypoints':()=>({source_ref:'feature/x',source_commit:commit,image:'img',entrypoints:{evaluate_checkpoints:{title:'Eval',gpu:false,format:'json',template:{checkpoints:[],opponents:[]},description:'e'}}})})
 at('/jobs/new?entrypoint=evaluate_checkpoints&sha='+sha)
 await waitFor(()=>expect(JSON.parse((screen.getByLabelText('任务配置') as HTMLTextAreaElement).value).checkpoints).toEqual([sha]))
})

it('shows research metrics source and per-checkpoint diagnoses',async()=>{
 mock({'/api/v1/jobs/job-1':()=>({...job,status:'succeeded',finished:1761900100,events:[],run_id:'job-1',run_files:['experiment'],diagnoses:[{model_id:'arm_ep01024',sha256:'d'.repeat(64),games:50,unique_openings:24,counts:{wins:38,losses:12,missed_immediate_win:0,immediate_win_available:38},tactical_wins:18,tactical_count:18,repetition_rate:0,mean_decision_ms:0.73,output_semantics:'policy_logits'}]}),
  '/api/v1/jobs/job-1/metrics':()=>({metrics:[{episode:32,'totals.learner_wins':6,imitation_ce:1.38}],metric_source:'experiment/research-metrics.jsonl',evaluations:[]})})
 at('/jobs/job-1?tab=diagnosis')
 expect(await screen.findByRole('link',{name:'arm_ep01024'})).toBeTruthy()
 expect(screen.getByText('18 / 18')).toBeTruthy()
 expect(screen.getByText('38 / 12')).toBeTruthy()
 fireEvent.click(screen.getByRole('tab',{name:'训练曲线'}))
 expect(await screen.findByText(/experiment\/research-metrics.jsonl/)).toBeTruthy()
})

it('shows live curves, registered models and cancels with the token',async()=>{
 localStorage.setItem('quoridor-platform-token','secret')
 const calls=mock({'/api/v1/jobs/job-1':()=>({...job,events:[{event:'started',time:1761900000}],run_id:'job-1',run_files:['metrics.jsonl'],registered:['b'.repeat(64)]}),
  '/api/v1/jobs/job-1/metrics':()=>({metrics:[{type:'train',episode:10,step:100,loss:0.5},{type:'train',episode:20,step:200,loss:0.3}],evaluations:[]}),
  '/api/v1/jobs/job-1/cancel':()=>({...job,cancel_requested:true})})
 at('/jobs/job-1')
 expect(await screen.findByText(/2 个记录点/)).toBeTruthy()
 expect(screen.getByRole('link',{name:'在训练工作区查看'}).getAttribute('href')).toBe('/training?run=job-1')
 expect(screen.getByRole('link',{name:/模型 bbbbbb/}).getAttribute('href')).toBe('/models/'+'b'.repeat(64))
 fireEvent.click(screen.getByRole('button',{name:'取消任务'}))
 await waitFor(()=>expect(calls.some(c=>c.url==='/api/v1/jobs/job-1/cancel'&&(c.init?.headers as Record<string,string>)['X-Platform-Token']==='secret')).toBe(true))
})
