"""Safe project checks and release preparation; deliberately has no deploy command."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = {"dev": ("web", "gameplay"), "training": ("runtime", "lab", "research", "mlflow")}
DOCUMENTS = ("AGENTS.md", "docs/AGENT_WORKFLOW.md", "docs/MIGRATION_STATUS.md",
             "documents/system/design/monorepo-deployment.md", "deploy/README.md")
DIGEST = re.compile(r"[a-z0-9][a-z0-9./_:-]*@sha256:[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")

def validate_release(data, role):
    if not isinstance(data, dict):
        raise ValueError("release manifest must be an object")
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not COMMIT.fullmatch(str(data.get("source_commit", ""))):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", str(data.get("release_id", ""))):
        raise ValueError("invalid release_id")
    contracts = data.get("contracts")
    if not isinstance(contracts, dict) or not all(
        isinstance(contracts.get(k), str) and re.fullmatch(r"[a-zA-Z0-9._-]+", contracts[k])
        for k in ("agent_api", "position_schema", "observation", "action_space")
    ):
        raise ValueError("explicit contract versions required")
    images = data.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("images must be an object")
    result = {}
    for component in ROLES[role]:
        ref = images.get(component, "")
        if not isinstance(ref, str) or not DIGEST.fullmatch(ref):
            raise ValueError(component + " must use an immutable sha256 image digest")
        result[component.upper() + "_IMAGE"] = ref
    return result

def check():
    missing = [p for p in DOCUMENTS if not (ROOT / p).is_file()]
    if missing:
        raise ValueError("missing authority documents: " + ", ".join(missing))
    for name in ROLES:
        text = (ROOT / ("deploy/compose." + name + ".yaml")).read_text()
        if "build:" in text or "/workspace/" in text:
            raise ValueError("release Compose must not build or mount development source")
    print("Documentation entry points and release source-isolation checks passed.")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    release = sub.add_parser("release-env")
    release.add_argument("manifest", type=Path)
    release.add_argument("--role", choices=ROLES, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            check()
        else:
            data = json.loads(args.manifest.read_text())
            for key, value in validate_release(data, args.role).items():
                print(key + "=" + value)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, str(exc) + "\n")

if __name__ == "__main__":
    main()
