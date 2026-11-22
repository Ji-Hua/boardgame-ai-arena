import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {act,cleanup,fireEvent,render,screen} from '@testing-library/react'
import {PairComparison,differenceViews} from '../src/views/PairComparison'
import * as lab from '../src/lab'
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.useRealTimers()})
const view=(n:number)=>({name:'pawn',layout:'board_xy',values:[[n,2*n]]})
const evidence=(n:number)=>({position_hash:'same',checkpoint_sha256:'model',neural:{sensitivities:[{objective:'q',views:[view(n)]},{objective:'margin',views:[view(100)]}]}})
it('subtracts raw output gradients rather than independently selected margins',()=>{expect(differenceViews(evidence(3),evidence(1))[0].values).toEqual([[2,4]])})
it('rejects mixed checkpoints and malformed gradient shapes',()=>{expect(()=>differenceViews(evidence(1),{...evidence(2),checkpoint_sha256:'other'})).toThrow();const b=evidence(2);b.neural.sensitivities[0].views[0].values=[[1]];expect(()=>differenceViews(evidence(1),b)).toThrow()})
it('sums broadcast inventory derivatives as one scalar, preserving board coordinates elsewhere',()=>{const make=(n:number)=>({neural:{sensitivities:[{objective:'q',views:Array.from({length:7},(_,i)=>({...view(n),name:String(i)}))}]}});const d=differenceViews(make(3),make(1));expect(d[4].values).toEqual([[6]]);expect(d[5].layout).toBe('scalar');expect(d[0].values).toEqual([[2,4]])})
it('pins historical A, cancels stale B results, reuses evidence across tabs',async()=>{
 vi.useFakeTimers();let release:any;const pending=new Promise(r=>release=r)
 const request=vi.spyOn(lab,'labRequest').mockImplementationOnce(()=>pending as any).mockResolvedValue({value:1,bound:'exact',after:{position:lab.opening}})
 const rows=[0,1,2].map(x=>({id:`pawn:${x}:1`,label:String(x),action:{player:1,type:'pawn',target:[x,1]},value:x,bound:'exact'}));const actor={spec:{algo:{type:'greedy'}}};const result={candidates:rows,selected_id:rows[2].id};const branch=vi.fn(),onB=vi.fn(),onView=vi.fn()
 const props={position:lab.opening,actor,result,actual:rows[0].action,onB,onBranch:branch,onView}
 const v=render(<PairComparison {...props} bId={rows[1].id}/>);expect(screen.getByText(/实际动作：0/)).toBeTruthy();expect(screen.getByLabelText('对照动作 B').querySelectorAll('option').length).toBe(2)
 fireEvent.click(screen.getByRole('tab',{name:'分支比较'}));await act(async()=>vi.advanceTimersByTime(260));expect(request).toHaveBeenCalledTimes(1)
 v.rerender(<PairComparison {...props} bId={rows[2].id}/>);await act(async()=>vi.advanceTimersByTime(260));await act(async()=>release({value:99,bound:'exact'}));expect(request).toHaveBeenCalledTimes(3);expect(request.mock.calls[0][2]?.aborted).toBe(true)
 expect(screen.queryByText('99')).toBeNull();fireEvent.click(screen.getByRole('button',{name:'查看 B 分支'}));expect(branch).toHaveBeenLastCalledWith(lab.opening)
 fireEvent.click(screen.getByRole('tab',{name:'输入证据'}));expect(branch).toHaveBeenLastCalledWith(null);expect(request).toHaveBeenCalledTimes(3)
})
