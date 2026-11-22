export const cellPoint = (x, y) => [34 + x * 50 + 25, 34 + (8-y) * 50 + 25];
export const wallRect = (type, x, y) => type === "horizontal"
  ? [34+x*50+3, 34+(8-y)*50-4, 94, 8]
  : [34+(x+1)*50-4, 34+(7-y)*50+3, 8, 94];
export const moveId = a => `${a.type}:${a.target[0]}:${a.target[1]}`;
export const numeric = v => v === "+inf" ? Infinity : v === "-inf" ? -Infinity : Number(v);
export const scoreText = v => v === "+inf" ? "+∞" : v === "-inf" ? "−∞" :
  Number.isInteger(Number(v)) && Math.abs(Number(v)) < 10000 ? String(v) : Number(v).toPrecision(4);
export const escapeHTML = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function heat(value, min, max) {
  if (!Number.isFinite(value)) return value > 0 ? "#6ca686" : "#d5a589";
  const t = max === min ? .6 : (value-min)/(max-min);
  return t > .7 ? "#a8c9ae" : t < .3 ? "#e3c6b2" : "#dee5ce";
}

export function boardSVG({snapshot, candidates=[], selected, recommended, map="pawn", paths=true, editing=false, tool="p1", preview=false}) {
  const p = snapshot.position;
  const active = preview || editing ? [] : candidates.filter(c => c.action.type === map);
  const byId = new Map(active.map(c => [c.id, c]));
  const finite = active.map(c => numeric(c.value)).filter(Number.isFinite);
  const min = finite.length ? Math.min(...finite) : 0, max = finite.length ? Math.max(...finite) : 0;
  let out = `<svg viewBox="0 0 518 518" role="group" aria-label="棋盘，a1 在左下，P1 向上，P2 向下"><defs><marker id="arrow1" markerWidth="5" markerHeight="5" refX="3" refY="2" orient="auto"><path d="M0,0 L4,2 L0,4" fill="none" stroke="#227c68"/></marker><marker id="arrow2" markerWidth="5" markerHeight="5" refX="3" refY="2" orient="auto"><path d="M0,0 L4,2 L0,4" fill="none" stroke="#d88848"/></marker></defs><rect x="26" y="26" width="466" height="466" rx="9" fill="#e9e5d8"/>`;
  for (let x=0;x<9;x++) {
    out += `<text x="${cellPoint(x,0)[0]}" y="511" text-anchor="middle" fill="#87917e" font-size="11" font-family="ui-monospace,monospace">${String.fromCharCode(97+x)}</text>`;
    for(let y=0;y<9;y++) {
      const [cx,cy] = cellPoint(x,y), id=`pawn:${x}:${y}`, c=byId.get(id);
      out += `<rect x="${cx-23}" y="${cy-23}" width="46" height="46" rx="3" fill="${c?heat(numeric(c.value),min,max):"#f8f7ee"}" stroke="${id===selected&&!preview&&!editing?"#b88046":"none"}" stroke-width="2" data-cell="${x},${y}" ${c?`data-action="${id}"`:""} tabindex="0" role="button" aria-label="${String.fromCharCode(97+x)}${y+1}"><title>${String.fromCharCode(97+x)}${y+1}${c?` · ${c.bound==="upper"?"≤ ":c.bound==="lower"?"≥ ":""}${scoreText(c.value)}`:""}</title></rect>`;
      if(c) out+=`<text x="${cx}" y="${cy+5}" text-anchor="middle" fill="#4d654c" font-size="11" font-family="ui-monospace,monospace" pointer-events="none">${c.bound==="upper"?"≤":c.bound==="lower"?"≥":""}${scoreText(c.value)}</text>`;
      if(id===recommended&&!preview&&!editing) out+=`<rect x="${cx-19}" y="${cy-19}" width="38" height="38" rx="3" fill="none" stroke="#236c56" stroke-width="2" pointer-events="none"/>`;
    }
  }
  for(let y=0;y<9;y++) out+=`<text x="13" y="${cellPoint(0,y)[1]+4}" text-anchor="middle" fill="#87917e" font-size="11" font-family="ui-monospace,monospace">${y+1}</text>`;
  if(paths) snapshot.paths?.forEach((path,i)=>{
    const points=path.map(([x,y])=>{const [a,b]=cellPoint(x,y);return `${a+(i===0?-7:7)},${b}`;}).join(" ");
    if(path.length>1) out+=`<polyline points="${points}" fill="none" stroke="${i===0?"#227c68":"#d88848"}" opacity=".48" stroke-width="2.5" stroke-dasharray="4 5" stroke-linecap="round" marker-end="url(#arrow${i+1})" pointer-events="none"/>`;
  });
  if(map!=="pawn") {
    // Adjacent legal walls overlap geometrically. Use separate anchor badges,
    // then project only the inspected wall's full footprint.
    const inspected = byId.get(selected) || byId.get(recommended);
    if(inspected) {
      const [x,y,w,h] = wallRect(inspected.action.type,...inspected.action.target);
      out+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="#b88046" fill-opacity=".45" stroke="#b88046" stroke-width="2" pointer-events="none"/>`;
    }
    for(const c of active) {
      const [ax,ay]=c.action.target, cx=34+(ax+1)*50, cy=34+(8-ay)*50;
      const value=`${c.bound==="upper"?"≤":c.bound==="lower"?"≥":""}${scoreText(c.value)}`;
      out+=`<rect x="${cx-15}" y="${cy-8}" width="30" height="16" rx="4" fill="${heat(numeric(c.value),min,max)}" stroke="${c.id===selected?"#b88046":c.id===recommended?"#236c56":"#bac6ad"}" stroke-width="${c.id===selected||c.id===recommended?2:1}" data-action="${c.id}" tabindex="0" role="button" aria-label="${c.label}"><title>${c.label} · ${value}</title></rect><text x="${cx}" y="${cy+3}" text-anchor="middle" fill="#425641" font-size="8" font-family="ui-monospace,monospace" pointer-events="none">${value}</text>`;
    }
  }
  p.walls.forEach(w=>{
    const [x,y,width,height]=wallRect(w.orientation,w.x,w.y);
    out+=`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="3" fill="#5e685b" data-wall="${w.orientation}:${w.x}:${w.y}" tabindex="0" role="button" aria-label="已有墙 ${String.fromCharCode(97+w.x)}${w.y+1}${w.orientation==="horizontal"?"h":"v"}"><title>已有${w.orientation==="horizontal"?"横":"竖"}墙</title></rect>`;
  });
  p.pawns.forEach(([x,y],i)=>{
    const [cx,cy]=cellPoint(x,y);
    out+=`<circle cx="${cx}" cy="${cy+2}" r="15" fill="#24392318" pointer-events="none"/><circle cx="${cx}" cy="${cy}" r="15" fill="${i===0?"#227c68":"#d88848"}" stroke="#fffefa" stroke-width="3" pointer-events="none"/><text x="${cx}" y="${cy+4}" text-anchor="middle" fill="white" font-size="10" font-weight="600" pointer-events="none">P${i+1}</text>`;
  });
  if(editing && (tool==="horizontal"||tool==="vertical")) for(let x=0;x<8;x++) for(let y=0;y<8;y++){
    out+=`<circle cx="${34+(x+1)*50}" cy="${34+(8-y)*50}" r="6" fill="#fffefabb" stroke="#7e9674" data-anchor="${x},${y}" tabindex="0" role="button" aria-label="墙锚点 ${String.fromCharCode(97+x)}${y+1}"/>`;
  }
  return out+"</svg>";
}
