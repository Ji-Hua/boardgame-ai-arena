export interface Position {
  pawns: [[number, number], [number, number]]
  walls_remaining: [number, number]
  walls: { x: number; y: number; orientation: 'horizontal' | 'vertical' }[]
  turn: 1 | 2
}
export const opening: Position = { pawns: [[4,0],[4,8]], walls_remaining: [10,10], walls: [], turn: 1 }
export async function labRequest(path: string, body?: unknown, signal?: AbortSignal): Promise<any> {
  const response = await fetch('/research-lab/api/' + path, {method: body === undefined ? 'GET' : 'POST',
    headers: {'Content-Type': 'application/json'}, ...(body === undefined ? {} : {body: JSON.stringify(body)}), signal})
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail ?? data))
  return data
}
export function positionBoard(position: Position, actions: {type?:string;target?:number[];player?:number}[] = []) {
  const wall_history:Record<string,{player:1|2;ply:number}>={}
  actions.forEach((a,ply)=>{if((a.type==='horizontal'||a.type==='vertical')&&a.target?.length===2&&(a.player===1||a.player===2))wall_history[`${a.type}:${a.target[0]}:${a.target[1]}`]={player:a.player,ply}})
  // Branch-only walls have no historical move, even if supplied by a caller.
  for(const key of Object.keys((position as any).wall_owners??{}))delete wall_history[key]
  return {wall_history,wall_owners:{...wallOwners(actions),...(position as Position & {wall_owners?:Record<string,1|2>}).wall_owners},p1: position.pawns[0], p2: position.pawns[1],
    h_walls: position.walls.filter(w => w.orientation === 'horizontal').map(w => [w.x,w.y] as [number,number]),
    v_walls: position.walls.filter(w => w.orientation === 'vertical').map(w => [w.x,w.y] as [number,number])}
}

// Display metadata only. No ownership is inferred from wall order or alternating turns.
export function wallOwners(actions: {type?:string;target?:number[];player?:number}[]) {
 const owners:Record<string,1|2>={}
 for(const a of actions)if((a.type==='horizontal'||a.type==='vertical')&&a.target?.length===2&&(a.player===1||a.player===2))owners[`${a.type}:${a.target[0]}:${a.target[1]}`]=a.player
 return owners
}
export function colorBranch(data:any, action:any) {
 const annotate=(p:any,actions:any[])=>p&&Object.keys(wallOwners(actions)).length?{...p,wall_owners:wallOwners(actions)}:p
 const steps:any[]=[]
 const principal_variation=data.principal_variation?.map((step:any,i:number)=>{
  steps.push(step.action??(i===0?action:{}))
  return {...step,position:annotate(step.position,steps),after:step.after?{...step.after,position:annotate(step.after.position,steps)}:step.after}
 })
 // Search-tree nodes with no explicit action path remain neutral for new walls.
 return {...data,after:data.after?{...data.after,position:annotate(data.after.position,[action])}:data.after,principal_variation}
}
