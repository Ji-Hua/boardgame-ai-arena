"""Run inside the MLflow container; record immutable checkpoint versions."""
import json
import argparse
from pathlib import Path
import tempfile
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

DATA=Path("/quoridor-data")
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,default=Path("/data/registry.json"))
parser.add_argument("--catalog",type=Path,default=DATA/"catalog.json")
args=parser.parse_args()
client=MlflowClient(tracking_uri="http://127.0.0.1:8742")
name="Quoridor / historical checkpoints"
experiment=client.get_experiment_by_name(name)
exp_id=experiment.experiment_id if experiment else client.create_experiment(name)
catalog=json.loads(args.catalog.read_text())
registered=[]
for model in catalog["models"]:
    arch=model["metadata"]["model_config"].get("model_arch","mlp")
    registry_name="quoridor-"+arch+"-dqn"
    try:
        client.get_registered_model(registry_name)
    except MlflowException as exc:
        if exc.error_code!="RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(registry_name,description="Immutable Quoridor checkpoint bundles; use the Quoridor loader, not generic pyfunc.")
    versions=client.search_model_versions(f"name = '{registry_name}'")
    found=next((v for v in versions if v.tags.get("checkpoint_sha256")==model["sha256"]),None)
    if found:
        registered.append({"id":model["id"],"name":registry_name,"version":found.version,"run_id":found.run_id,"sha256":model["sha256"]})
        continue
    run=client.create_run(exp_id,tags={
        "mlflow.runName":model["id"],"checkpoint_sha256":model["sha256"],
        "historical_training_commit":"unknown","source_path":model["source_path"],
        "selection_reason":model["selection_reason"],"validation_status":"pending_benchmark",
    })
    rid=run.info.run_id
    for key,value in model["metadata"].items():
        if key!="model_config":
            client.log_param(rid,key,str(value))
    for key,value in model["metadata"]["model_config"].items():
        client.log_param(rid,key,json.dumps(value))
    local=DATA/"models"/(model["sha256"]+".pt")
    client.log_artifact(rid,str(local),artifact_path="checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        metadata=Path(tmp)/"metadata.json"
        metadata.write_text(json.dumps(model,indent=2))
        client.log_artifact(rid,str(metadata),artifact_path="checkpoint")
    version=client.create_model_version(
        name=registry_name,source=f"runs:/{rid}/checkpoint",run_id=rid,
        tags={"checkpoint_sha256":model["sha256"],"agent_id":model["id"],
              "validation_status":"pending_benchmark","format":"quoridor_dqn_checkpoint"},
        description=model["selection_reason"],
    )
    client.set_registered_model_alias(registry_name,model["id"],version.version)
    client.set_terminated(rid)
    registered.append({"id":model["id"],"name":registry_name,"version":version.version,"run_id":rid,"sha256":model["sha256"]})
result=json.dumps({"models":registered},indent=2)
temporary=args.output.with_suffix(".tmp")
temporary.write_text(result)
temporary.replace(args.output)
print(result)
