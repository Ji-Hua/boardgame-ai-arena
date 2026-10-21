"""Read the existing model-management catalog; requests never supply file paths."""
import json
import os
from pathlib import Path


def catalog():
    path = Path(os.environ.get("LAB_MODEL_CATALOG", Path.home() / "quoridor-data/catalog.json"))
    if not path.is_file():
        return {"models": [], "message": "未配置模型目录；设置 LAB_MODEL_CATALOG 后重启组件。"}
    try:
        data = json.loads(path.read_text())
        registry_path = path.with_name("registry.json")
        registry = json.loads(registry_path.read_text()).get("models", []) if registry_path.is_file() else []
        registered = {m["sha256"]: m for m in registry}
        rows, seen = [], set()
        for item in data["models"]:
            identifier, digest = item["id"], item["sha256"]
            if identifier in seen or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("模型目录存在重复 ID 或非法 SHA256")
            seen.add(identifier)
            checkpoint = Path(item["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = path.parent / checkpoint
            metadata = item.get("metadata", {})
            rows.append({"id": identifier, "sha256": digest, "checkpoint": str(checkpoint.resolve()),
                         "architecture": metadata.get("model_config", {}).get("model_arch", "mlp"),
                         "metadata": metadata, "registry": registered.get(digest),
                         "training_status": item.get("training_status"),
                         "available": checkpoint.is_file()})
        return {"models": rows, "message": None}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"模型目录读取失败：{exc}") from exc


def resolve(identifier):
    match = next((m for m in catalog()["models"] if m["id"] == identifier), None)
    if match is None:
        raise ValueError("模型 ID 不在配置的 catalog 中。")
    if not match["available"]:
        raise ValueError("该模型的 checkpoint 文件不存在。")
    return match
