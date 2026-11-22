import React, {useState} from 'react'

export function EvidenceViewer({value,filename='evidence.json'}:{value:unknown;filename?:string}){
 const [wrap,setWrap]=useState(true)
 const text=JSON.stringify(value,null,2)
 function download(){const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
 return <div className="aw-evidence"><div className="aw-toolbar"><strong>完整证据</strong><label><input type="checkbox" checked={wrap} onChange={e=>setWrap(e.target.checked)}/>自动换行</label><button onClick={download}>下载 JSON</button></div><pre className="aw-evidence-json" aria-label="完整证据 JSON" tabIndex={0} style={{whiteSpace:wrap?'pre-wrap':'pre'}}>{text}</pre></div>
}
