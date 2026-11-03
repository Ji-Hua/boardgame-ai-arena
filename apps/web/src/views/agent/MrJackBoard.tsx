import type {Obj} from './shared'
const columns=[[5,5],[7,2],[6,3],[6,2],[7,1],[8,0],[8,1],[8,2],[7,3],[6,4],[6,3],[7,2],[5,3]]
const coords=columns.flatMap(([count,start],x)=>Array.from({length:count},(_,y)=>[x,start+2*y]))
const buildings=new Set([7,8,18,14,20,22,33,35,37,48,50,52,63,65,67,71,77,78])
const lamps=new Set([10,12,32,36,49,53,73,75]),sewers=new Set([6,11,15,31,43,54,70,74,79])
const exits=new Set([5,11,74,80])
const colours=['#235fa4','#843fb2','#995b27','#116e70','#a63440','#946900','#4d6c2a','#595470']
const directions=['北','南','西北','西南','东北','东南']
const phases:Record<string,string>={choose:'选择角色',order:'选择技能顺序',move:'移动角色',ability:'使用技能',orient:'调整 Watson 灯光',terminal:'对局结束'}
export function MrJackBoard({observation:o,legalActions=[],onAction}:{observation?:Obj,legalActions?:Obj[],onAction?:(action:number)=>void}){
 if(!o)return null
 return <section aria-label="Mr. Jack 棋盘">
 <p>第 {o.round} / 8 轮 · {phases[o.phase]??o.phase} · 上次目击：{o.witness?'可见':'不可见'} · Watson 灯光：{directions[o.orientation]}</p>
 <svg viewBox="0 0 660 610" role="img" aria-label="伦敦棋盘，格号与动作列表对应" style={{width:'100%',maxWidth:760,background:'#f3eee1',borderRadius:12}}>
 {coords.map(([x,y],cell)=>{
  const cx=40+x*47,cy=35+y*27
  const person=o.positions.indexOf(cell),lit=o.lights.includes(cell),closed=o.covered.includes(cell)
  const action=legalActions.find(a=>a.id===100+cell)
  const canMove=!!onAction&&!!action
  const fill=buildings.has(cell)?'#5a6268':lamps.has(cell)?(lit?'#ffd34e':'#b4b1a6'):'#e4dbc3'
  return <g key={cell} role={canMove?'button':undefined} tabIndex={canMove?0:undefined} aria-label={canMove?'移动到 '+cell:undefined} style={canMove?{cursor:'pointer'}:undefined} onClick={()=>{if(canMove)onAction!(action.id)}} onKeyDown={e=>{if(canMove&&(e.key==='Enter'||e.key===' ')){e.preventDefault();onAction!(action.id)}}}><polygon points={[[31,0],[16,27],[-16,27],[-31,0],[-16,-27],[16,-27]].map(([a,b])=>(cx+a)+','+(cy+b)).join(' ')} fill={fill} stroke={canMove?'#16856a':exits.has(cell)?'#a14032':'#b1a78e'} strokeWidth={canMove?4:exits.has(cell)?3:1}/>
  <text x={cx-15} y={cy-14} fontSize="9" fill={buildings.has(cell)?'#fff':'#38332b'}>{cell}</text>
  {sewers.has(cell)&&<circle cx={cx+12} cy={cy+14} r={7} fill={closed?'#555':'#c5beb0'} stroke="#333" strokeWidth={closed?3:1}/>}
  {lamps.has(cell)&&<text x={cx} y={cy+5} textAnchor="middle" fontSize="11">{lit?'灯 '+o.lights.indexOf(cell):'空灯'}</text>}
  {o.cordons.includes(cell)&&<path d={'M'+(cx-20)+','+(cy+20)+' l40,-40'} stroke="#b72724" strokeWidth="5"/>}
  {person>=0&&<><circle cx={cx} cy={cy} r={18} fill={colours[person]} stroke={o.visible[person]?'#fff5a3':'#20252a'} strokeWidth={4}/><text x={cx} y={cy+4} textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">{o.characters[person].slice(0,2)}</text></>}
  </g>
 })}
 </svg>
 <p>人物黄色描边表示可见，深色描边表示不可见；红线为封锁出口，小圆为窨井。{onAction&&'绿色格框是可点击的合法落点。'}</p>
 <ul>{o.characters.map((name:string,i:number)=><li key={name}><strong style={{color:colours[i]}}>{name}</strong> · 格 {o.positions[i]} · {o.suspects.includes(i)?'尚未排除':'已排除'}{o.identity===i?' · 你的 Jack 身份':''}{o.private_alibis?.includes(i)?' · 你持有其不在场证明':''}</li>)}</ul>
 </section>
}
