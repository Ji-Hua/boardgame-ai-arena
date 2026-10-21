"""CNN experiments on the training host, with blocking per-checkpoint diagnosis.

Run: .venv/bin/python scripts/train_cnn_repair.py --config experiment.json
All generated artifacts are written to the configured external run directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend-server")]

import torch
from agent_system.evaluation.cnn_diagnostics import diagnose_checkpoint
from scripts.train_dqn import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads(args.config.read_text())
    cfg = TrainConfig(**experiment["training"])
    if cfg.model_arch != "cnn" or not cfg.device.startswith("cuda"):
        raise ValueError("This experiment runner requires CNN and CUDA")
    output = Path(experiment["output"]).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Training artifacts must be outside the source repository")
    if Path(cfg.checkpoint_dir).resolve() != output / "checkpoints":
        raise ValueError("checkpoint_dir must equal output/checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    book = json.loads(Path(experiment["book_manifest"]).read_text())["book"]
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2))
    (output / "book.json").write_text(json.dumps(book, indent=2))
    inputs = {}
    for path in (cfg.warm_start_checkpoint, cfg.demonstration_dataset, experiment.get("demonstrations", {}).get("dataset"),
                 experiment.get("distillation", {}).get("dataset"),
                 experiment.get("research", {}).get("parent_checkpoint"),
                 experiment.get("research", {}).get("dataset"),
                 experiment.get("alphazero", {}).get("parent_checkpoint")):
        if path:
            with Path(path).open("rb") as stream:
                inputs[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "input-sha256.json").write_text(json.dumps(inputs, indent=2))
    sources = sorted({p for base in ["agent_system", "scripts", "backend-server/backend/adapters"]
                      for p in (ROOT/base).rglob("*.py") if "artifacts" not in p.parts})
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / "source-manifest.json").write_text(json.dumps({"files": manifest,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}, indent=2))
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    diagnoses = []
    search_diagnoses = []

    def evaluate(path, result):
        identifier = f"{cfg.agent_id}_{path.stem}"
        diagnoses.append(diagnose_checkpoint(path, identifier, book, output / "diagnostics" / identifier))
        (output / "diagnoses.json").write_text(json.dumps(diagnoses, indent=2, ensure_ascii=False))
        if "alphazero" in experiment:
            search = {k: experiment["alphazero"]["search"][k] for k in ("simulations", "c_puct", "max_depth")}
            search_diagnoses.append(diagnose_checkpoint(path, identifier + "_mcts", book,
                output / "search-diagnostics" / identifier, search=search))
            (output / "search-diagnoses.json").write_text(json.dumps(search_diagnoses, indent=2))

    if "alphazero" in experiment:
        from agent_system.training.dqn.alphazero_lite import train_alphazero
        result = train_alphazero(cfg, experiment["alphazero"], output, evaluate)
    elif "research" in experiment:
        from agent_system.training.dqn.policy_research import train_research
        result = train_research(cfg, experiment["research"], output, evaluate)
    elif "distillation" in experiment:
        from agent_system.training.dqn.distillation import train_distillation
        result = train_distillation(cfg, experiment["distillation"], output, evaluate)
    elif "teacher" in experiment:
        from agent_system.training.dqn.teacher_training import train_teacher
        result = train_teacher(cfg, experiment["teacher"], output, evaluate)
    elif "demonstrations" in experiment:
        from agent_system.training.dqn.demonstrations import train_demonstrations
        result = train_demonstrations(cfg, experiment["demonstrations"], output, evaluate)
    else:
        result = asdict(train(cfg, checkpoint_callback=evaluate))
    (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
