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
export function positionBoard(position: Position) {
  return {p1: position.pawns[0], p2: position.pawns[1],
    h_walls: position.walls.filter(w => w.orientation === 'horizontal').map(w => [w.x,w.y] as [number,number]),
    v_walls: position.walls.filter(w => w.orientation === 'vertical').map(w => [w.x,w.y] as [number,number])}
}
