import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor,act} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import {RunInspector,TrainingWorkspace,flattenConfig} from '../src/views/TrainingWorkspace'
import AgentWorkspace from '../src/views/AgentWorkspace'
import {api} from '../src/api'
const sha='a'.repeat(64)
const metric=(scope='train')=>({run_id:'a',points:[{step:1,episode:null,values:{loss:3}},{step:2,episode:null,values:{loss:1}}],columns:['loss'],metadata:{loss:{semantic:'cross_entropy',unit:'nats',scope,step_type:'gradient_updates'}},axes:[{key:'step',label:'Updates'}],warnings:[],snapshots:[],provenance:{}} as any)
const detail=(id:string)=>({run_id:id,run_name:id,source:'mlflow',experiment_id:'9',status:'FINISHED',algorithm:'policy_distillation',config:{net:{width:id==='a'?32:64},shared:true},provenance:{},checkpoints:[{filename:'same.pt',sha256:sha,step:12000,progress_type:'update'}],warnings:[]} as any)
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals()})
function mock(){vi.spyOn(api,'getRun').mockImplementation(async id=>detail(id));vi.spyOn(api,'getMetrics').mockResolvedValue(metric())}
it('compares compatible curves and nested configuration differences; checkpoints use exact SHA',async()=>{
 mock();render(<MemoryRouter><RunInspector ids={['a','b']} season="s"/></MemoryRouter>)
 const chart=await screen.findByRole('img',{name:/loss/});expect(chart.querySelectorAll('path')).toHaveLength(2)
 fireEvent.click(screen.getByRole('tab',{name:'配置对比'}))
 expect(screen.getByText('config.net.width')).toBeTruthy();expect(screen.queryByText('config.shared')).toBeNull()
 fireEvent.click(screen.getByLabelText('仅差异'));expect(screen.getByText('config.shared')).toBeTruthy()
 fireEvent.click(screen.getByRole('tab',{name:'Checkpoints'}))
 expect(screen.getAllByRole('link',{name:'评测 / 回放 / 对战'})[0].getAttribute('href')).toContain('/checkpoints/'+sha)
})
it('does not overlay training and validation curves just because their metric names match',async()=>{
 mock();vi.mocked(api.getMetrics).mockImplementation(async id=>metric(id==='a'?'train':'validation'))
 render(<MemoryRouter><RunInspector ids={['a','b']} season="s"/></MemoryRouter>)
 expect(await screen.findByText('暂无语义、单位、范围和横轴一致的曲线')).toBeTruthy()
 expect(screen.queryByRole('img')).toBeNull()
})
it('discards old Run requests after changing the selection',async()=>{
 mock();let resolve:(v:any)=>void=()=>{};vi.mocked(api.getRun).mockImplementation(id=>id==='a'?new Promise(r=>{resolve=r}):Promise.resolve(detail(id)))
 const view=render(<MemoryRouter><RunInspector ids={['a']} season="s"/></MemoryRouter>)
 view.rerender(<MemoryRouter><RunInspector ids={['b']} season="s"/></MemoryRouter>)
 await screen.findByText('Run b')
 await act(async()=>resolve(detail('a')))
 expect(screen.queryByText('Run a')).toBeNull()
 expect(screen.getByRole('heading',{name:'b'})).toBeTruthy()
})
it('keeps multiple selected Run IDs in the route and supports removal',async()=>{
 mock();vi.spyOn(api,'listRuns').mockResolvedValue({runs:[detail('a'),detail('b')],warnings:[]} as any)
 render(<MemoryRouter initialEntries={['/training?season=s&run=a']}><TrainingWorkspace season="s"/></MemoryRouter>)
 const b=await screen.findByLabelText('选择 b');fireEvent.click(b)
 await screen.findByRole('heading',{name:'2 Runs 对比'})
 fireEvent.click(screen.getByLabelText('选择 a'))
 await screen.findByRole('heading',{name:'b'})
})
it('flattens nested keys without losing array or null values',()=>{
 expect(flattenConfig({a:{b:3},x:[1,2],nil:null})).toEqual({'a.b':'3',x:'[1,2]',nil:'null'})
})
it('never associates a checkpoint with an Agent having a different SHA',async()=>{
 const wrong={id:'wrong',name:'Same filename',sha256:'b'.repeat(64),kind:'neural',spec:{algo:{type:'dqn'},policy:{type:'top_k',k:1}}}
 vi.stubGlobal('fetch',vi.fn(async(url:string)=>({ok:true,json:async()=>url.endsWith('/seasons')?{seasons:[{id:'s'}]}:{season:'s',agents:[wrong],matrix:{}}})))
 render(<MemoryRouter initialEntries={['/checkpoints/'+sha+'?season=s']}><AgentWorkspace/></MemoryRouter>)
 await waitFor(()=>expect(screen.getByText('已完成评测中没有此 checkpoint；暂不可从评测 Agent 发起对战')).toBeTruthy())
 expect(screen.queryByRole('link',{name:'发起对战'})).toBeNull()
})
