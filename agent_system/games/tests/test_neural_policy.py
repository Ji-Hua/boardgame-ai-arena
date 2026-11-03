import hashlib
import json
import numpy as np
import pytest
from agent_system.games.mrjack import MrJack
from agent_system.games.mrjack_encoding import encode
from agent_system.games.contracts import MR_JACK,Decision,stable_hash
from agent_system.runtime.research_policy import FORMAT,runtime
from apps.jobs.policy_store import PolicyStore

def test_numeric_bundle_integrity_and_secret_independence(tmp_path):
    g=MrJack();s=g.initial_state();size=len(encode(g.observe(s,"detective"),"detective"))
    arrays=dict(w0=np.zeros((64,size),np.float32),b0=np.zeros(64,np.float32),
        w1=np.zeros((64,64),np.float32),b1=np.zeros(64,np.float32),
        w2=np.zeros((3072,64),np.float32),b2=np.zeros(3072,np.float32))
    np.savez(tmp_path/"actor.npz",**arrays)
    sha=hashlib.sha256((tmp_path/"actor.npz").read_bytes()).hexdigest()
    bundle=dict(schema_version=1,format=FORMAT,game_ref=MR_JACK.wire(),
        output_semantics="policy_logits",encoder="mrjack_visible_summary_v1",
        supported_roles=["detective","jack"],memory_mode="visible_summary",action_count=3072,
        input_size=size,files=[dict(name="actor.npz",sha256=sha)],project_id="jack",
        config=dict(algorithm="maskable_ppo"),source=dict(commit="c"*40),policy_kind="stochastic_masked_policy")
    bundle["digest"]=stable_hash(bundle)
    (tmp_path/"policy.json").write_text(json.dumps(bundle))
    agent=runtime(bundle,tmp_path)
    assert agent.probabilities(Decision("detective",g.observe(s,"detective"),(0,1),0))==[.5,.5]
    store=PolicyStore(tmp_path/"store")
    meta=store.register(dict(id="j",project_id="jack",spec=dict(game_ref=MR_JACK.wire())),tmp_path/"policy.json")
    assert len(meta["agent_specs"])==2
    assert store.get("jack",meta["sha256"])[0]==meta
    with pytest.raises(KeyError):store.get("other",meta["sha256"])
    stored=store.root/(sha+".npz");stored.chmod(0o644);stored.write_bytes(b"corrupted")
    with pytest.raises(ValueError):store.get("jack",meta["sha256"])
    with pytest.raises(ValueError):store.bundle("jack",meta["sha256"])
