import {describe,it,expect} from 'vitest'
import {findCycle,positionKey} from '../src/cycles'
import {opening,type Position} from '../src/lab'
const state=(x:number,turn:1|2=1):Position=>({...opening,pawns:[[x,2],[4,7]],turn})
describe('recorded full-position cycles',()=>{
 it('finds shortest repeating suffix and the first participating position',()=>{
  const a=state(1),b=state(2,2),c=state(2),d=state(1,2)
  expect(findCycle([opening,a,b,c,d,a,b,c,d,a])).toEqual({start:1,end:9,period:4,rounds:2,phases:[1,2,3,4]})
 })
 it('distinguishes turn, inventory and walls; ignores wall serialization order',()=>{
  expect(positionKey(state(1))).not.toBe(positionKey(state(1,2)))
  expect(positionKey(state(1))).not.toBe(positionKey({...state(1),walls_remaining:[9,10]}))
  const walls:Position['walls']=[{orientation:'horizontal',x:0,y:0},{orientation:'vertical',x:4,y:3}]
  expect(positionKey({...state(1),walls})).toBe(positionKey({...state(1),walls:[...walls].reverse()}))
  expect(findCycle([state(1),{...state(1),walls}])).toBeNull()
 })
 it('finds an earlier loop after an escape and handles no repetition',()=>{
  expect(findCycle([state(1),state(2),state(1),state(3)])?.end).toBe(2)
  expect(findCycle([state(1),state(2),state(3)])).toBeNull()
  expect(findCycle([])).toBeNull()
 })
})
