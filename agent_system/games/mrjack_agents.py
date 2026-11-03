"""Visible-information heuristic: avoids unsupported accusations, prioritizes escape."""
import random

class MrJackHeuristic:
    def reset(self,seed):
        self.rng=random.Random(seed)
    def act(self,decision):
        o=decision.observation;actions=list(decision.legal_actions)
        escapes=[a for a in actions if 230<=a<234]
        if escapes:return self.rng.choice(escapes)
        if decision.role=="detective" and o["phase"]=="move":
            suspects=set(o["suspects"])-set(o["private_alibis"])
            safe=[];certain=[]
            for a in actions:
                if 100<=a<186 and a-100 in o["positions"]:
                    target=o["positions"].index(a-100)
                    if suspects=={target}:certain.append(a)
                else:safe.append(a)
            if certain:return self.rng.choice(certain)
            if safe:return self.rng.choice(safe)
        if o["phase"]=="choose" and 4 in actions:return 4
        return self.rng.choice(actions)
