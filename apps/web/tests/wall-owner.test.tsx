import React from 'react'
import {expect,it} from 'vitest'
import {render,cleanup,fireEvent} from '@testing-library/react'
import {EvalReplayBoard} from '../src/components/EvalReplayBoard'
import {positionBoard,opening,colorBranch} from '../src/lab'
it('colors recorded owners in either orientation and leaves unknown walls neutral',()=>{
 const actions=[{player:2,type:'horizontal',target:[1,2]},{player:1,type:'vertical',target:[4,5]}]
 const p={...opening,walls:[{x:1,y:2,orientation:'horizontal' as const},{x:4,y:5,orientation:'vertical' as const},{x:0,y:0,orientation:'horizontal' as const}]}
 const {container}=render(<EvalReplayBoard state={positionBoard(p,actions)}/>);expect(container.querySelector('[data-wall="horizontal:1:2"]')?.getAttribute('fill')).toBe('#2196F3');expect(container.querySelector('[data-wall="vertical:4:5"]')?.getAttribute('fill')).toBe('#4CAF50');expect(container.querySelector('[data-wall="horizontal:0:0"]')?.getAttribute('data-owner')).toBe('unknown');cleanup()
})
it('keeps branch wall ownership independent of history and never mutates engine snapshots',()=>{
 const a={player:1,type:'horizontal',target:[2,3]},b={player:2,type:'vertical',target:[6,7]}
 const data={after:{position:opening},principal_variation:[{action:a,position:opening},{action:b,position:opening}]};const branch=colorBranch(data,a)
 expect(positionBoard(branch.principal_variation[1].position,[{player:2,type:'horizontal',target:[0,1]}]).wall_owners).toEqual({'horizontal:0:1':2,'horizontal:2:3':1,'vertical:6:7':2})
 expect(positionBoard(branch.after.position).wall_owners).toEqual({'horizontal:2:3':1});expect((opening as any).wall_owners).toBeUndefined()
})

it('renders snapshot trails without stationary duplicates, can hide them and does not intercept board clicks',()=>{
 const end={...opening,pawns:[[4,1],[4,8]] as [[number,number],[number,number]]}
 const state=positionBoard(end,[],[opening,opening,end]);expect(state.trails).toEqual([[[4,0],[4,1]],[[4,8]]])
 const view=render(<EvalReplayBoard state={state}/>);const trails=view.getByTestId('historical-trails');expect(trails.getAttribute('pointer-events')).toBe('none');expect(trails.getAttribute('opacity')).toBe('0.28');expect(trails.querySelector('polyline')?.getAttribute('points')).toBe('238,442 238,391');fireEvent.click(view.getByRole('button',{name:'历史轨迹'}));expect(view.queryByTestId('historical-trails')).toBeNull();cleanup()
})
