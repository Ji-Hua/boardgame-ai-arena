"""Safe numeric inference for explicit research policy formats; no pickle loading."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
from agent_system.games.contracts import MR_JACK,stable_hash

FORMAT="mrjack_mlp_tanh_v1"

def validate_bundle(bundle):
    if bundle.get("format")!=FORMAT:
        if bundle.get("files"):raise ValueError("Unexpected tabular policy dependencies")
        from agent_system.training.openspiel_job import PolicyAgent
        PolicyAgent(bundle)
        return
    if (bundle.get("schema_version")!=1 or bundle.get("game_ref")!=MR_JACK.wire()
            or bundle.get("output_semantics")!="policy_logits"
            or bundle.get("encoder")!="mrjack_visible_summary_v1"
            or bundle.get("supported_roles")!=["detective","jack"]
            or bundle.get("memory_mode")!="visible_summary"
            or bundle.get("action_count")!=3072
            or type(bundle.get("input_size")) is not int):
        raise ValueError("Incompatible neural policy contract")
    if stable_hash({k:v for k,v in bundle.items() if k!="digest"})!=bundle.get("digest"):
        raise ValueError("Policy integrity mismatch")
    files=bundle.get("files")
    if not isinstance(files,list) or len(files)!=1 or files[0].get("name")!="actor.npz":
        raise ValueError("Invalid model files")
    if not re.fullmatch(r"[0-9a-f]{64}",files[0].get("sha256","")):
        raise ValueError("Invalid model identity")

class NeuralPolicy:
    def __init__(self,bundle,model_path):
        import numpy as np
        from agent_system.games.mrjack_encoding import encode
        from agent_system.games.mrjack import MrJack
        validate_bundle(bundle)
        raw=Path(model_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=bundle["files"][0]["sha256"]:
            raise ValueError("Neural model SHA mismatch")
        size=len(encode(MrJack().observe(MrJack().initial_state(),"detective"),"detective"))
        if bundle["input_size"]!=size:raise ValueError("Encoder shape mismatch")
        with np.load(model_path,allow_pickle=False) as archive:
            self.weights={k:archive[k] for k in archive.files}
        shapes={"w0":(64,size),"b0":(64,),"w1":(64,64),"b1":(64,),
                "w2":(3072,64),"b2":(3072,)}
        if set(self.weights)!=set(shapes) or any(
            self.weights[k].shape!=v or not np.isfinite(self.weights[k]).all() for k,v in shapes.items()):
            raise ValueError("Invalid neural weight structure")
        self.bundle=bundle
    def reset(self,seed):
        self.rng=random.Random(seed)
    def probabilities(self,decision):
        import numpy as np
        from agent_system.games.mrjack_encoding import encode
        x=encode(decision.observation,decision.role);w=self.weights
        x=np.tanh(w["w0"]@x+w["b0"]);x=np.tanh(w["w1"]@x+w["b1"])
        logits=(w["w2"]@x+w["b2"])[list(decision.legal_actions)].astype(np.float64)
        p=np.exp(logits-logits.max());p/=p.sum()
        return p.tolist()
    def act(self,decision):
        return self.rng.choices(decision.legal_actions,weights=self.probabilities(decision))[0]

def runtime(bundle,root,stored=False):
    validate_bundle(bundle)
    if bundle.get("format")==FORMAT:
        filename=bundle["files"][0]["sha256"]+".npz" if stored else "actor.npz"
        return NeuralPolicy(bundle,Path(root)/filename)
    from agent_system.training.openspiel_job import PolicyAgent
    return PolicyAgent(bundle)
