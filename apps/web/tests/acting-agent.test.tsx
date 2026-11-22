import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {act,cleanup,fireEvent,render,screen} from '@testing-library/react'
import {StepAnalysis} from '../src/views/AgentWorkspace'
import {DecisionSummary} from '../src/views/DecisionSummary'
import {Tensor} from '../src/views/LabView'
import * as lab from '../src/lab'
const greedy={id:'g',name:'Greedy',kind:'heuristic',spec:{algo:{type:'greedy'},policy:{type:'top_k',k:1}}}
const cnn={id:'n',name:'CNN',kind:'neural',sha256:'a'.repeat(64),spec:{algo:{type:'dqn',params:{}},policy:{type:'top_k',k:1}}}
afterEach(()=>{cleanup();vi.restoreAllMocks()})
it('follows the actor and exact checkpoint, without auto-computing or offering a default model picker',async()=>{
 const request=vi.spyOn(lab,'labRequest').mockResolvedValue({candidates:[]})
 const view=render(<StepAnalysis position={lab.opening} actor={greedy} roster={{agents:[greedy,cnn]}} onBranch={()=>{}}/>)
 expect(screen.queryByLabelText('分析 Agent')).toBeNull();expect(request).not.toHaveBeenCalled()
 view.rerender(<StepAnalysis position={{...lab.opening,turn:2}} actor={cnn} roster={{agents:[greedy,cnn]}} onBranch={()=>{}}/>)
 expect(screen.getByText('跟随行棋方 · CNN')).toBeTruthy()
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'运行分析'})))
 expect(request.mock.calls[0][1]).toMatchObject({model_id:cnn.sha256,algorithm:'dqn'})
 fireEvent.click(screen.getByRole('button',{name:'对照分析'}));fireEvent.change(screen.getByLabelText('分析 Agent'),{target:{value:'g'}})
 expect(screen.getByText('对照 · Greedy')).toBeTruthy()
 fireEvent.click(screen.getByRole('button',{name:'当前 Agent'}));expect(screen.getByText('跟随行棋方 · CNN')).toBeTruthy()
})
it('does not substitute a model for a human actor',()=>{
 render(<StepAnalysis position={lab.opening} actor={{id:'human',kind:'human',name:'玩家'}} roster={{agents:[greedy,cnn]}} onBranch={()=>{}}/>)
 expect(screen.getByRole('button',{name:'运行分析'}).hasAttribute('disabled')).toBe(true)
 expect(screen.getByText('玩家行动没有 Agent 评分；可选择对照分析')).toBeTruthy()
})
it('pins the actual historical action even when it is outside the top ten',async()=>{
 const actual={type:'pawn',target:[4,1],player:1}
 const candidates=Array.from({length:11},(_,i)=>({id:i===10?'pawn:4:1':'candidate'+i,label:'move'+i,value:11-i,bound:'exact',order:i,action:actual}))
 vi.spyOn(lab,'labRequest').mockResolvedValue({candidates,selected_id:'candidate0'})
 render(<StepAnalysis position={lab.opening} actor={greedy} roster={{agents:[greedy]}} actual={actual} onBranch={()=>{}}/>)
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'运行分析'})))
 expect(screen.getAllByRole('button',{name:'变化'})).toHaveLength(11)
 expect(screen.getByText('复算 Top1 与历史落子不一致')).toBeTruthy()
})
it('does not claim a random historical choice was reproduced, nor subtract minimax bounds',()=>{
 const actor={...greedy,spec:{...greedy.spec,policy:{type:'top_k',k:3}}}
 render(<DecisionSummary actor={actor} actual={{type:'pawn',target:[4,1]}} comparison={false} result={{selected_id:'x',candidates:[{id:'x',value:2,bound:'exact'},{id:'pawn:4:1',value:1,bound:'upper'}]}}/>)
 expect(screen.getByText('实际动作在本次评分的 Top-k 内；历史抽样状态未记录')).toBeTruthy()
 expect(screen.getByText(/排序包含搜索界/)).toBeTruthy()
})
it('formats short tensor labels, retains full hover values and can hide labels',()=>{
 const {container}=render(<Tensor data={{layout:'matrix',values:[[0.123456789,0.000000123456]],name:'activation'}}/>)
 expect(container.querySelector('text')?.textContent).toBe('0.123')
 expect(container.querySelector('title')?.textContent).toContain('0.123456789')
 expect(container.querySelectorAll('text')[1].textContent).toBe('1.23e-7')
 expect(screen.getByText(/非负值/)).toBeTruthy()
 fireEvent.click(screen.getByLabelText('显示数值'));expect(container.querySelectorAll('svg text')).toHaveLength(0)
})
