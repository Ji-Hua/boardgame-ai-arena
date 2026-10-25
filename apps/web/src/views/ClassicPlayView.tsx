import React, {useState} from 'react'
import {Link} from 'react-router-dom'
import {ConfigPage} from '../../../../frontend/src/view/pages/ConfigPage'
import {GamePage} from '../../../../frontend/src/view/pages/GamePage'
import type {GameConfig} from '../../../../frontend/src/modes/live/LiveController'
export function ClassicPlayView(){
 const [config,setConfig]=useState<GameConfig|null>(null)
 return <section className="classic-play"><div className="page-heading"><h1>Classic Play</h1><Link to="/play">Exact checkpoint research Play</Link></div>{config?<GamePage gameConfig={config} onBack={()=>setConfig(null)}/>:<ConfigPage onStart={setConfig} onBack={()=>window.history.back()}/>}</section>
}
