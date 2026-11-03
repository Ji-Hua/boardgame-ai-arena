"""Authoritative headless London 2016 base-game rules, with staged integer actions."""
from collections import deque
import copy
import json
from .contracts import MR_JACK
from .mrjack_board import (COORDS,NEIGHBORS,BUILDINGS,LAMP_SITES,SEWERS,STREETS,
                           EXITS,NAMES,INITIAL_POSITIONS,INITIAL_LIGHTS)

# Action ranges: choose 0..7; before/after 10/11; move 100+cell;
# exchange 200+character; orientation 220+direction; exits 230+exit index;
# lamps 300+tile*8+site index; sewer 400+closed-index*9+site index;
# cordon 500+index*4+exit index; whistle 1000+character*258+cell*3+(cost-1).
ACTION_COUNT=3072

class MrJack:
    ref=MR_JACK
    roles=("detective","jack")
    def initial_state(self):
        return dict(positions=list(INITIAL_POSITIONS),lights=list(INITIAL_LIGHTS),
            covered=[11,74],cordons=[11,74],orientation=4,round=1,turn=0,
            phase="identity",jack=None,alibis=list(range(8)),private={"detective":[],"jack":[]},
            hand=[],remaining=list(range(8)),suspects=list(range(8)),witness=True,
            selected=None,moved=False,ability_done=False,pull_remaining=0,
            resume=None,winner=None,history=[])

    def actor(self,s):
        if s["winner"] is not None:return None
        if s["phase"] in ("identity","deal","alibi"):return "chance"
        first="detective" if s["round"]%2 else "jack"
        return first if s["turn"] in (0,3) else self.roles[1-self.roles.index(first)]

    def chance_outcomes(self,s):
        if s["phase"]=="identity": pool=list(range(8))
        elif s["phase"]=="deal":pool=s["remaining"]
        elif s["phase"]=="alibi":pool=s["alibis"]
        else:return ()
        return tuple((c,1/len(pool)) for c in pool)

    def _neighbors(self,s,cell,stealth=False,tunnels=True):
        out=[n for n in NEIGHBORS[cell] if n>=0 and (stealth or n in STREETS)]
        if tunnels and cell in SEWERS and cell not in s["covered"]:
            out += [n for n in SEWERS if n not in s["covered"] and n!=cell]
        return out

    def distances(self,s,start,stealth=False,tunnels=True):
        found={start:0};queue=deque([start])
        while queue:
            cell=queue.popleft()
            for n in self._neighbors(s,cell,stealth,tunnels):
                if n not in found:
                    found[n]=found[cell]+1;queue.append(n)
        return found

    def visible(self,s):
        lit=set()
        for lamp in s["lights"]:
            if lamp>=0:lit.update(n for n in NEIGHBORS[lamp] if n in STREETS)
        cell=s["positions"][3]
        while True:
            cell=NEIGHBORS[cell][s["orientation"]]
            if cell not in STREETS:break
            lit.add(cell)
        return [p in lit or any(q in NEIGHBORS[p] for j,q in enumerate(s["positions"]) if j!=i)
                for i,p in enumerate(s["positions"])]

    def _moves(self,s):
        c=s["selected"];start=s["positions"][c]
        limit=4 if c==1 else 3
        distances=self.distances(s,start,stealth=c==1)
        occupied=set(s["positions"])
        actions=[100+n for n,d in distances.items() if 0<d<=limit and n in STREETS
                 and (n not in occupied or self.actor(s)=="detective")]
        if self.actor(s)=="jack" and c==s["jack"] and not s["witness"]:
            actions += [230+i for i,e in enumerate(EXITS) if e not in s["cordons"] and distances.get(e,99)+1<=limit]
        if c==7:actions += [200+i for i in range(8) if i!=7]
        return actions

    def _pulls(self,s):
        result=[];distance=self.distances(s,s["positions"][6],tunnels=False)
        occupied=set(s["positions"])
        for c,start in enumerate(s["positions"]):
            if c==6:continue
            reachable={start}
            for cost in range(1,s["pull_remaining"]+1):
                reachable={n for p in reachable for n in self._neighbors(s,p,tunnels=False)}
                for n in sorted(reachable):
                    if n not in occupied and distance[n]<distance[start]:
                        result.append(1000+c*258+n*3+cost-1)
        return result

    def legal_actions(self,s):
        if self.actor(s) is None:return ()
        if self.actor(s)=="chance":return tuple(a for a,_ in self.chance_outcomes(s))
        phase=s["phase"]
        if phase=="choose":out=s["hand"]
        elif phase=="order":out=[10,11]
        elif phase=="move":out=self._moves(s)
        elif phase=="orient":out=list(range(220,226))
        elif phase=="ability":
            c=s["selected"]
            if c==0:out=[500+i*4+j for i in range(2) for j,e in enumerate(EXITS) if e not in s["cordons"]]
            elif c==2:out=[400+i*9+j for i in range(2) for j,e in enumerate(sorted(SEWERS)) if e not in s["covered"]]
            elif c==5:out=[300+i*8+j for i,p in enumerate(s["lights"]) if p>=0
                          for j,e in enumerate(sorted(LAMP_SITES)) if e not in s["lights"]]
            elif c==6:out=self._pulls(s)
            else:raise ValueError("Invalid ability phase")
        else:raise ValueError("Unknown phase")
        return tuple(sorted(out))

    def _finish(self,s):
        s["hand"].remove(s["selected"]);s["selected"]=None
        s["turn"]+=1
        if s["turn"]<4:
            s["phase"]="choose";return
        visibility=self.visible(s)
        s["witness"]=visibility[s["jack"]]
        s["suspects"]=[i for i in s["suspects"] if visibility[i]==s["witness"]]
        if s["round"]<=4:s["lights"][s["round"]]=-1
        if s["round"]==8:
            s["winner"]="jack";s["phase"]="terminal";return
        s["round"]+=1;s["turn"]=0
        if s["round"]%2:
            s["remaining"]=list(range(8));s["phase"]="deal"
        else:
            s["hand"]=s["remaining"][:];s["remaining"]=[];s["phase"]="choose"

    def _after_move(self,s):
        c=s["selected"];s["moved"]=True
        if c==4 and s["alibis"]:s["phase"]="alibi";return
        if c==3:
            s["resume"]="finish";s["phase"]="orient";return
        if c in (0,2,5,6) and not s["ability_done"]:
            s["phase"]="ability"
            if c==6:s["pull_remaining"]=3
        else:self._finish(s)

    def _after_ability(self,s):
        s["ability_done"]=True
        if s["moved"]:self._finish(s)
        else:s["phase"]="move"

    def apply(self,s,action):
        if type(action) is not int or action not in self.legal_actions(s):
            raise ValueError("Illegal Mr. Jack action")
        s=copy.deepcopy(s);phase=s["phase"];actor=self.actor(s)
        if actor!="chance":s["history"].append([actor,action])
        if phase=="identity":
            s["jack"]=action;s["alibis"].remove(action);s["phase"]="deal"
        elif phase=="deal":
            s["remaining"].remove(action);s["hand"].append(action);s["hand"].sort()
            if len(s["hand"])==4:s["phase"]="choose"
        elif phase=="alibi":
            # The actor is determined by round/turn; only the activating player sees the card.
            first="detective" if s["round"]%2 else "jack"
            owner=first if s["turn"] in (0,3) else self.roles[1-self.roles.index(first)]
            s["private"][owner].append(action);s["alibis"].remove(action);self._finish(s)
        elif phase=="choose":
            s.update(selected=action,moved=False,ability_done=False,pull_remaining=0)
            s["phase"]="order" if action in (0,2,5,6) else "move"
        elif phase=="order":
            s["phase"]="ability" if action==10 else "move"
            if s["selected"]==6:s["pull_remaining"]=3
        elif phase=="move":
            c=s["selected"]
            if 230<=action<234:
                s["winner"]="jack";s["phase"]="terminal"
            elif 200<=action<208:
                other=action-200
                s["positions"][c],s["positions"][other]=s["positions"][other],s["positions"][c]
                if other==3:
                    s["resume"]="finish";s["phase"]="orient"
                else:self._finish(s)
            else:
                dest=action-100
                target=next((i for i,p in enumerate(s["positions"]) if p==dest),None)
                s["positions"][c]=dest
                if target is not None:
                    s["winner"]="detective" if target==s["jack"] else "jack";s["phase"]="terminal"
                else:self._after_move(s)
        elif phase=="orient":
            s["orientation"]=action-220
            resume=s["resume"];s["resume"]=None
            if resume=="finish":self._finish(s)
            elif resume=="pull":
                s["phase"]="ability"
                if s["pull_remaining"]==0:self._after_ability(s)
            else:raise ValueError("Invalid orientation continuation")
        elif phase=="ability":
            c=s["selected"]
            if c==0:
                i,j=divmod(action-500,4);s["cordons"][i]=EXITS[j];self._after_ability(s)
            elif c==2:
                i,j=divmod(action-400,9);s["covered"][i]=sorted(SEWERS)[j];self._after_ability(s)
            elif c==5:
                i,j=divmod(action-300,8);s["lights"][i]=sorted(LAMP_SITES)[j];self._after_ability(s)
            elif c==6:
                who,remainder=divmod(action-1000,258);dest,k=divmod(remainder,3);cost=k+1
                s["positions"][who]=dest;s["pull_remaining"]-=cost
                if who==3:
                    s["resume"]="pull";s["phase"]="orient"
                elif s["pull_remaining"]==0:self._after_ability(s)
        # Explicit forced-action-impossible convention: no invented move or accusation.
        if s["phase"]=="ability" and s["selected"]==6 and not self._pulls(s):
            self._after_ability(s)
        return s

    def public_observe(self,s):
        keys=("positions","lights","covered","cordons","orientation","round","turn","phase",
              "hand","suspects","witness","selected","moved","ability_done","pull_remaining","winner","history")
        return {**{k:copy.deepcopy(s[k]) for k in keys},"visible":self.visible(s),
                "roles":list(self.roles),"characters":list(NAMES)}

    def observe(self,s,role):
        if role not in self.roles:raise ValueError("Unknown role")
        result=self.public_observe(s)
        result["private_alibis"]=s["private"][role][:]
        result["identity"]=s["jack"] if role=="jack" else None
        return result

    def returns(self,s):
        return {} if s["winner"] is None else {r:1.0 if r==s["winner"] else -1.0 for r in self.roles}

    def serialize(self,s):
        return json.dumps(s,sort_keys=True,separators=(",",":"))

    def restore(self,value):
        return json.loads(value)

    def action_label(self,s,action):
        phase=s["phase"]
        if phase=="choose":return "选择 "+NAMES[action]
        if phase=="order":return "先用技能" if action==10 else "先移动"
        if 100<=action<186:return "移动到 "+str(action-100)
        if 200<=action<208:return "与 "+NAMES[action-200]+" 交换"
        if 220<=action<226:return "灯光方向 "+str(action-220)
        if 230<=action<234:return "从出口 "+str(EXITS[action-230])+" 逃离"
        if 300<=action<348:
            i,j=divmod(action-300,8);return f"路灯 {i} → {sorted(LAMP_SITES)[j]}"
        if 400<=action<418:
            i,j=divmod(action-400,9);return f"井盖 {i} → {sorted(SEWERS)[j]}"
        if 500<=action<508:
            i,j=divmod(action-500,4);return f"路障 {i} → {EXITS[j]}"
        if action>=1000:
            c,n=divmod(action-1000,258);dest,k=divmod(n,3)
            return f"召集 {NAMES[c]} → {dest}（{k+1} 步）"
        return str(action)
