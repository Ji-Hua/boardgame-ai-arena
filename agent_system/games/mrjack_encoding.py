"""Versioned visible-summary encoder. Never accepts true engine state."""
import numpy as np
from .mrjack import ACTION_COUNT
PHASES=("identity","deal","choose","order","move","ability","orient","alibi","terminal")
ENCODER="mrjack_visible_summary_v1"

def onehot(value,size):
    return [float(value==i) for i in range(size)]

def encode(observation,role):
    o=observation
    vector=[x/85 for x in o["positions"]]+[x/85 for x in o["lights"]]
    vector += [x/85 for x in o["covered"]]+[x/85 for x in o["cordons"]]
    vector += onehot(o["orientation"],6)+[o["round"]/8,o["turn"]/3]
    vector += onehot(PHASES.index(o["phase"]),len(PHASES))
    for key in ("hand","suspects","private_alibis"):
        vector += [float(i in o.get(key,[])) for i in range(8)]
    vector += [float(o["witness"]),float(o["moved"]),float(o["ability_done"]),o["pull_remaining"]/3]
    vector += onehot(8 if o["selected"] is None else o["selected"],9)
    vector += onehot(8 if o.get("identity") is None else o["identity"],9)
    vector += [float(role=="detective"),float(role=="jack")]+[float(x) for x in o["visible"]]
    return np.asarray(vector,dtype=np.float32)

def mask(actions):
    result=np.zeros(ACTION_COUNT,dtype=bool)
    result[list(actions)]=True
    return result
