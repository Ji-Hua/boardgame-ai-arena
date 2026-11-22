import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {act,cleanup,fireEvent,render,screen} from '@testing-library/react'
import {StepAnalysis} from '../src/views/AgentWorkspace'
import * as lab from '../src/lab'
const actor={id:'g',name:'Greedy',kind:'heuristic',spec:{algo:{type:'greedy'},policy:{type:'top_k',k:1}}}
const props={actor,roster:{agents:[actor]},onBranch:()=>{}}
const advance=async(ms=400)=>{await act(async()=>{vi.advanceTimersByTime(ms)})}
afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks()})
it('debounces seeking, suppresses playback and reuses successful paused positions',async()=>{
 vi.useFakeTimers();const request=vi.spyOn(lab,'labRequest').mockResolvedValue({candidates:[]})
 const view=render(<StepAnalysis {...props} position={lab.opening} autoAnalyze/>)
 await advance(100);expect(request).not.toHaveBeenCalled()
 const next={...lab.opening,turn:2 as const}
 view.rerender(<StepAnalysis {...props} position={next} autoAnalyze/>);await advance()
 expect(request).toHaveBeenCalledTimes(1);expect((request.mock.calls[0][1] as any).position.turn).toBe(2)
 view.rerender(<StepAnalysis {...props} position={next} suspended/>);await advance(2000)
 expect(request).toHaveBeenCalledTimes(1)
 view.rerender(<StepAnalysis {...props} position={next} autoAnalyze/>);await advance()
 expect(request).toHaveBeenCalledTimes(1)
 view.rerender(<StepAnalysis {...props} position={lab.opening} autoAnalyze/>);await advance()
 expect(request).toHaveBeenCalledTimes(2)
})
it('invalidates a late response when playback starts and retries once on pause',async()=>{
 vi.useFakeTimers();let resolve:(v:any)=>void=()=>{}
 const request=vi.spyOn(lab,'labRequest').mockImplementation(()=>new Promise(r=>{resolve=r}))
 const view=render(<StepAnalysis {...props} position={lab.opening} autoAnalyze/>);await advance()
 const signal=request.mock.calls[0][2] as AbortSignal
 view.rerender(<StepAnalysis {...props} position={lab.opening} suspended/>);expect(signal.aborted).toBe(true)
 await act(async()=>resolve({candidates:[{id:'stale',label:'STALE'}]}));expect(screen.queryByText('STALE')).toBeNull()
 view.rerender(<StepAnalysis {...props} position={lab.opening} autoAnalyze/>);await advance();expect(request).toHaveBeenCalledTimes(2)
})
it('does not auto-loop on errors or auto-run comparison mode',async()=>{
 vi.useFakeTimers();const request=vi.spyOn(lab,'labRequest').mockRejectedValue(new Error('busy'))
 render(<StepAnalysis {...props} position={lab.opening} autoAnalyze/>);await advance();await advance(5000)
 expect(request).toHaveBeenCalledTimes(1)
 fireEvent.click(screen.getByRole('button',{name:'对照分析'}));await advance();expect(request).toHaveBeenCalledTimes(1)
})
