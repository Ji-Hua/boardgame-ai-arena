"""Import representative historical checkpoints into an immutable local store."""
import hashlib
import json
from pathlib import Path
import shutil
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/jihua/quoridor-data")
ARTIFACTS = ROOT/"agent_system/training/artifacts/dqn"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def historical_peak(run):
    rows = [json.loads(x) for x in (ARTIFACTS/run/"eval_results.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("opponent_name",r.get("opponent"))=="random_legal"]
    row = max(rows,key=lambda r:r["win_rate"])
    matches = list((ARTIFACTS/run/"checkpoints").glob(f'ep{row["episode"]:05d}_step*.pt'))
    if len(matches)!=1:
        raise ValueError(f"Ambiguous historical checkpoint for {run}: {matches}")
    return matches[0], f'historical peak vs random_legal: episode {row["episode"]}, rate {row["win_rate"]}; selected on old data'


def main():
    (DATA/"models").mkdir(parents=True,exist_ok=True)
    selections = [
        ("mlp_stable", ARTIFACTS/"baselines/dqn_v0_stable.pt", "published stable MLP baseline"),
        ("cnn_random_2k", ARTIFACTS/"dqn_cnn_distance_delta_w004_random_2k_001/checkpoints/ep02000_step433050.pt", "final random-opponent CNN"),
        ("cnn_observation_2k", ARTIFACTS/"cnn_curriculum_observation_2k_001/checkpoints/ep02000_step232954.pt", "original completed 2k checkpoint; source directory later reused"),
        ("cnn_curriculum_peak", *historical_peak("cnn_curriculum_random_to_minimax_5k_001")),
        ("cnn_mixed_peak", *historical_peak("cnn_mixed_random80_minimaxd1_5k_001")),
        ("cnn_5k_final", ARTIFACTS/"cnn_curriculum_observation_5k_002/checkpoints/ep05000_step1663335.pt", "final long-run checkpoint with historical divergence evidence"),
    ]
    specs = [
        {"id":"random_legal","display_name":"Uniform random (all legal actions)","deterministic":False,
         "algo":{"type":"random"},"policy":{"type":"top_k","k":209}},
        {"id":"greedy","display_name":"Greedy","algo":{"type":"greedy"},"policy":{"type":"top_k","k":1}},
        {"id":"minimax_d1","display_name":"Minimax depth 1","algo":{"type":"minimax","params":{"depth":1}},"policy":{"type":"top_k","k":1}},
        {"id":"minimax_d2","display_name":"Minimax depth 2","algo":{"type":"minimax","params":{"depth":2}},"policy":{"type":"top_k","k":1}},
    ]
    models = []
    for label,path,reason in selections:
        digest = sha(path)
        dest = DATA/"models"/(digest+".pt")
        if dest.exists():
            if sha(dest)!=digest:
                raise ValueError(f"Immutable store corruption: {dest}")
        else:
            shutil.copyfile(path,dest)
            dest.chmod(0o444)
        raw = torch.load(dest,map_location="cpu",weights_only=True)
        metadata = {key:raw.get(key) for key in [
            "checkpoint_id","agent_id","episode_count","training_step","observation_version",
            "action_space_version","action_count","created_at","model_config",
        ]}
        models.append({"id":label,"sha256":digest,"checkpoint":str(dest),
                       "source_path":str(path.relative_to(ROOT)),"selection_reason":reason,
                       "historical_training_commit":"unknown","metadata":metadata})
        specs.append({
            "id":label,"display_name":label,"deterministic":True,"category":"ai",
            "algo":{"type":"dqn","params":{"checkpoint":str(dest),"sha256":digest,"device":"cpu"}},
            "policy":{"type":"top_k","k":1},
            "model_metadata":{"sha256":digest,"parameters":raw["model_config"].get("param_count"),
                              "episode":raw["episode_count"],"selection_reason":reason},
        })
    roster={"agents":specs,"selection":"4 fixed baselines + 6 representative learned checkpoints",
            "models":models}
    target=DATA/"catalog.json"
    if target.exists() and json.loads(target.read_text())!=roster:
        raise ValueError("Catalog already exists with different content; create a new catalog version")
    target.write_text(json.dumps(roster,indent=2))
    print(json.dumps({"catalog":str(target),"agents":[s["id"] for s in specs],
                      "models":[{"id":m["id"],"sha256":m["sha256"],"selection":m["selection_reason"]} for m in models]},indent=2))


if __name__=="__main__":
    main()
