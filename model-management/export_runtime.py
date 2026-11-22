"""Export a pinned catalog as canonical YAML definitions for online play."""
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--catalog", type=Path, default=Path("/home/jihua/quoridor-data/catalog.json"))
parser.add_argument("--output", type=Path, default=Path("/home/jihua/quoridor-data/agent_defs"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
definitions = {}
for path in (ROOT / "agent_system/definition/agent_defs").glob("*.yaml"):
    definition = yaml.safe_load(path.read_text())
    definitions[definition["id"]] = definition
catalog = json.loads(args.catalog.read_text())
for definition in catalog["agents"]:
    definitions[definition["id"]] = {
        key: value for key, value in definition.items() if key != "model_metadata"
    }
for agent_id, definition in definitions.items():
    path = args.output / (agent_id + ".yaml")
    if path.exists() and yaml.safe_load(path.read_text()) != definition:
        raise ValueError(f"Definition changed: use a new output directory for {agent_id}")
    path.write_text(yaml.safe_dump(definition, sort_keys=False))
print(json.dumps({"directory": str(args.output), "agents": sorted(definitions)}, indent=2))
