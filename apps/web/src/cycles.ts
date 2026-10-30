import type {Position} from './lab'
// Equality of recorded rule positions, not a reimplementation of game rules.
export function positionKey(p:Position){return JSON.stringify([p.turn,p.pawns,p.walls_remaining,p.walls.map(w=>[w.orientation,w.x,w.y].join(':')).sort()])}
export function findCycle(positions:Position[]){
 const keys=positions.map(positionKey)
 // Latest repeated segment; smallest period first. Include side-to-move and stocks.
 for(let end=keys.length-1;end>0;end--){
  for(let period=1;period<=Math.min(16,end);period++){
   if(keys[end]!==keys[end-period])continue
   let start=end-period
   while(start>0&&keys[start-1]===keys[start-1+period])start--
   return {start,end,period,rounds:Math.floor((end-start)/period),phases:Array.from({length:period},(_,i)=>start+i)}
  }
 }
 return null
}
