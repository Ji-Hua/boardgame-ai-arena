import React from 'react'
import {afterEach,beforeEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import {LabView} from '../src/views/LabView'
import {api} from '../src/api'
import * as lab from '../src/lab'
const sha='a'.repeat(64)
const response={position:lab.opening,selected_id:'pawn:4:1',candidates:[{id:'pawn:4:1',label:'e2',value:1,bound:'exact',action:{type:'pawn',target:[4,1],player:1}}],neural:{model:{architecture:'cnn',output_semantics:'policy_logits',parameters:42,metadata:{eval_summary:{value_head_status:'untrained'}}},input:{views:[]},outputs:[],layers:[{name:'conv',type:'Conv2d',shape:[8,9,9],weight:{shape:[8,6,3,3]}}]}}
function mount(){return render(<MemoryRouter><LabView/></MemoryRouter>)}
beforeEach(()=>{
 vi.spyOn(api,'context').mockResolvedValue({position:lab.opening as any,return_route:'/models',analysis_model:{sha256:sha}})
 vi.spyOn(lab,'labRequest').mockImplementation(async(path,body:any)=>{
  if(path==='meta')return {presets:[]}
  if(path==='models')return {models:[{id:sha,sha256:sha,available:true}]}
  if(path==='position')return {position:body}
  if(path==='analyze')return response
  if(path==='neural/tensor')return {values:[[1,-1]],layout:'matrix',layer:'conv',stats:{},window:{}}
  throw new Error('Unexpected request '+path)
 })
})
afterEach(cleanup)
it('keeps analysis explicit, in-shell and capability gated',async()=>{
 mount();await waitFor(()=>expect(screen.getByRole('button',{name:'Run analysis'}).hasAttribute('disabled')).toBe(false))
 expect(vi.mocked(lab.labRequest).mock.calls.some(c=>c[0]==='analyze')).toBe(false)
 fireEvent.click(screen.getByRole('button',{name:'Run analysis'}))
 await screen.findByRole('heading',{name:'Network diagnostics'})
 fireEvent.click(screen.getByRole('button',{name:'search',exact:true}))
 expect(screen.getByRole('button',{name:'Run PUCT'}).hasAttribute('disabled')).toBe(true)
 fireEvent.click(screen.getByRole('button',{name:'layers',exact:true}))
 fireEvent.click(screen.getByRole('button',{name:'Load tensor slice'}))
 expect(await screen.findByRole('img',{name:'Tensor slice by index'})).toBeTruthy()
 expect(document.querySelector('iframe')).toBeNull()
})
it('does not paint an old analysis after the algorithm changes',async()=>{
 let resolve:(x:any)=>void=()=>{}
 const original=vi.mocked(lab.labRequest).getMockImplementation()!
 vi.mocked(lab.labRequest).mockImplementation((path,...rest)=>path==='analyze'?new Promise(r=>{resolve=r}):original(path,...rest))
 mount();await waitFor(()=>expect(screen.getByRole('button',{name:'Run analysis'}).hasAttribute('disabled')).toBe(false))
 fireEvent.click(screen.getByRole('button',{name:'Run analysis'}))
 fireEvent.change(screen.getByLabelText('Analysis algorithm'),{target:{value:'greedy'}})
 resolve(response);await new Promise(r=>setTimeout(r,0))
 expect(screen.queryByRole('heading',{name:'Network diagnostics'})).toBeNull()
})
it('retains the last valid position when engine validation rejects an edit',async()=>{
 const original=vi.mocked(lab.labRequest).getMockImplementation()!
 vi.mocked(lab.labRequest).mockImplementation((path,body:any,...rest)=>path==='position'&&body.turn===7?Promise.reject(new Error('Invalid turn')):original(path,body,...rest))
 mount();await waitFor(()=>expect(screen.getByRole('button',{name:'Run analysis'}).hasAttribute('disabled')).toBe(false))
 fireEvent.change(screen.getByLabelText('Position JSON'),{target:{value:JSON.stringify({...lab.opening,turn:7})}})
 fireEvent.click(screen.getByRole('button',{name:'Validate and import'}))
 expect((await screen.findByRole('alert')).textContent).toContain('Invalid turn')
 expect(screen.getByRole('heading',{name:'Root position · P1 to move'})).toBeTruthy()
})
