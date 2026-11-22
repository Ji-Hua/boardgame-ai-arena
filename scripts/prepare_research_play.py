"""Generate an external, SHA-addressed research Play approval snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training-dashboard"))
from platform_entry import dashboard_module


def prepare():
    index = dashboard_module("platform_models").ModelIndex()
    models = []
    for row in index.listing()["models"]:
        if not row["research_playable"]:
            continue
        paths = sorted(checkpoint["path"] for checkpoint in row["checkpoints"] if checkpoint["available"])
        models.append(dict(sha256=row["sha256"], checkpoint=paths[0], architecture=row["architecture"],
            output_semantics=row["output_semantics"], metadata=row["metadata"],
            name=" / ".join(row["aliases"]), conflicts=row["conflicts"]))
    document = dict(purpose="session_scoped_research_play_only", models=models)
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    directory = index.data_root / "platform-p3-a" / "catalogs"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(payload).hexdigest() + ".json")
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("Existing approval snapshot has changed")
    else:
        with target.open("xb") as handle:
            handle.write(payload)
    return target


if __name__ == "__main__":
    print(prepare())