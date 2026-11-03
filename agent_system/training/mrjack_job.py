"""SB3-Contrib Maskable PPO training, exported as numeric actor-only policy."""
import importlib.metadata
import json
from pathlib import Path
from agent_system.games.contracts import MR_JACK,Decision,stable_hash
from agent_system.runtime.research_policy import FORMAT

DEFAULTS=dict(timesteps=1024,seed=42,n_steps=64,batch_size=64)

def validate_config(config):
    if not isinstance(config,dict) or set(config)-set(DEFAULTS):
        raise ValueError("Expected timesteps, seed, n_steps and batch_size")
    config={**DEFAULTS,**config}
    for key,lo,hi in (("timesteps",64,1000000),("seed",0,2**32-1),("n_steps",64,2048),("batch_size",8,2048)):
        if type(config[key]) is not int or not lo<=config[key]<=hi:raise ValueError("Invalid "+key)
    if config["n_steps"]%config["batch_size"] or config["timesteps"]%config["n_steps"]:
        raise ValueError("Use batch_size dividing n_steps and timesteps divisible by n_steps")
    return config

def train(job,run,events):
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from agent_system.training.mrjack_env import MrJackEnv
    from agent_system.games.mrjack_encoding import ENCODER,encode,mask
    from agent_system.runtime.research_policy import runtime
    from agent_system.training.jobs import sha256
    if job.get("game_ref")!=MR_JACK.wire():raise ValueError("Incompatible training game")
    config=validate_config(json.loads(job["config"]) if isinstance(job["config"],str) else job["config"])
    torch.set_num_threads(max(1,min(4,int(job.get("resources",{}).get("cpus",1)))))
    env=MrJackEnv()
    model=MaskablePPO("MlpPolicy",env,n_steps=config["n_steps"],batch_size=config["batch_size"],
        n_epochs=4,seed=config["seed"],device="cpu",policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])))
    with (run/"research-metrics.jsonl").open("x") as stream:
        class Metrics(BaseCallback):
            def _on_step(self):
                if self.num_timesteps%config["n_steps"]==0:
                    recent=env.completed[-100:]
                    row=dict(type="train",step=self.num_timesteps,progress_unit="learner_decision",
                             engine_decisions=env.total_engine_decisions,episodes=len(env.completed),
                             mean_return=float(np.mean([r["return_"] for r in recent])) if recent else 0.0)
                    stream.write(json.dumps(row)+"\n");stream.flush()
                    events.emit("training",**row)
                return True
        model.learn(config["timesteps"],callback=Metrics())
    net=model.policy.mlp_extractor.policy_net
    arrays={}
    for i,layer in enumerate((net[0],net[2],model.policy.action_net)):
        arrays["w"+str(i)]=layer.weight.detach().cpu().numpy()
        arrays["b"+str(i)]=layer.bias.detach().cpu().numpy()
    np.savez(run/"actor.npz",**arrays)
    bundle=dict(schema_version=1,format=FORMAT,game_ref=job["game_ref"],project_id=job["project_id"],
        source=job["source"],image=job["image"],config={**config,"algorithm":"maskable_ppo"},
        config_sha256=stable_hash(config),policy_kind="stochastic_masked_policy",output_semantics="policy_logits",
        encoder=ENCODER,input_size=env.observation_space.shape[0],action_count=3072,
        supported_roles=["detective","jack"],memory_mode="visible_summary",
        files=[dict(name="actor.npz",sha256=sha256(run/"actor.npz"))],
        package_versions={name:importlib.metadata.version(name) for name in ("sb3-contrib","stable-baselines3","torch","numpy","gymnasium")})
    bundle["digest"]=stable_hash(bundle)
    (run/"policy.json").write_text(json.dumps(bundle,sort_keys=True,indent=2))
    loaded=runtime(bundle,run)
    # Numerical parity at observed legal decision states; no deterministic override.
    max_error=0.0
    for seed in range(5):
        obs,_=env.reset(seed=seed)
        decision=Decision(env.role,env.game.observe(env.state,env.role),env.game.legal_actions(env.state),0)
        with torch.no_grad():
            tensor=torch.as_tensor(obs).reshape(1,-1)
            actual=model.policy.get_distribution(tensor,action_masks=mask(decision.legal_actions)).distribution.probs[0]
        error=float(np.max(np.abs(actual.cpu().numpy()[list(decision.legal_actions)]-loaded.probabilities(decision))))
        max_error=max(max_error,error)
    if max_error>1e-5:raise ValueError("Exported policy probability mismatch")
    (run/"export-verification.json").write_text(json.dumps(dict(max_probability_error=max_error,states=5)))
    (run/"research-config.json").write_text(json.dumps(bundle,indent=2))
    manifest=dict(schema_version=1,artifacts=[dict(file="policy.json",sha256=sha256(run/"policy.json"),
        kind=FORMAT,output_semantics="policy_logits",game_ref=job["game_ref"])])
    (run/"research-artifacts.json").write_text(json.dumps(manifest,indent=2))
    events.emit("policy_exported",**manifest["artifacts"][0])
    return []
