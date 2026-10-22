"""Read-only SHA-addressed model index; no checkpoint loading or production edits.

App integration: ``ModelIndex(unified_run_service=service).listing()`` and
``index.detail(full_sha, verify_sha=True)`` return JSON-ready dictionaries.
``analyzable``/``playable`` are *static eligibility*, not successful load claims.
Only server configuration supplies paths; request parameters must only be SHAs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping


SHA = re.compile(r"[0-9a-f]{64}\Z")
ARCHITECTURE_CATALOG = "experiments/cnn-architecture-20261021-213018/architecture-catalog.json"
LAB_CATALOG = "catalog-cnn-alphazero-lite-20261021.json"


def _sha(value: Any) -> str | None:
    return value if isinstance(value, str) and SHA.fullmatch(value) else None


class ModelIndex:
    """Index explicit catalogs, registry and one flat AgentSpec directory.

    Environment: QUORIDOR_DATA_ROOT, PLATFORM_MODEL_BASELINE_CATALOG,
    LAB_MODEL_CATALOG, PLATFORM_MODEL_ARCHITECTURE_CATALOG,
    PLATFORM_MODEL_CATALOGS (JSON list of paths or {name,path} objects),
    PLATFORM_MODEL_REGISTRY, PLATFORM_AGENT_SPEC_DIR and
    PLATFORM_MODEL_APPROVED_ROOTS (os.pathsep-separated extra read roots).
    Explicit ``catalog_paths`` replaces, rather than extends, defaults.
    ``deployed_shas`` is an optional trusted *observed* deployment snapshot;
    registry aliases and AgentSpec presence alone never establish deployment.
    """

    def __init__(self, *, catalog_paths=None, registry_path=None,
                 agent_spec_dir=None, approved_roots=None, data_root=None,
                 unified_run_service=None, deployed_shas: Mapping[str, str] | None = None):
        self.data_root = Path(data_root or os.environ.get(
            "QUORIDOR_DATA_ROOT", "/home/jihua/quoridor-data")).resolve()
        roots = approved_roots if approved_roots is not None else [
            self.data_root, *filter(None, os.environ.get(
                "PLATFORM_MODEL_APPROVED_ROOTS", "").split(os.pathsep))]
        self.roots = tuple(Path(p).resolve() for p in roots)
        if catalog_paths is None:
            configured = os.environ.get("PLATFORM_MODEL_CATALOGS")
            catalog_paths = json.loads(configured) if configured else [
                {"name": "baseline", "path": os.environ.get(
                    "PLATFORM_MODEL_BASELINE_CATALOG", str(self.data_root / "catalog.json"))},
                {"name": "lab", "path": os.environ.get(
                    "LAB_MODEL_CATALOG", str(self.data_root / LAB_CATALOG))},
                {"name": "architecture", "path": os.environ.get(
                    "PLATFORM_MODEL_ARCHITECTURE_CATALOG", str(self.data_root / ARCHITECTURE_CATALOG))}]
        if not isinstance(catalog_paths, (list, tuple)):
            raise ValueError("Catalog manifest must be a list")
        self.catalogs = [dict(name=str(p.get("name", "catalog")), path=str(p["path"]))
                         if isinstance(p, dict) else dict(name="catalog", path=str(p))
                         for p in catalog_paths]
        self.registry_path = str(registry_path or os.environ.get(
            "PLATFORM_MODEL_REGISTRY", self.data_root / "registry.json"))
        self.agent_spec_dir = str(agent_spec_dir or os.environ.get(
            "PLATFORM_AGENT_SPEC_DIR", self.data_root / "agent_defs"))
        self.service = unified_run_service
        self.deployed_shas = dict(deployed_shas or {})
        self._hash_cache: dict[tuple, str] = {}
        self._lock = threading.RLock()

    def _path(self, raw, *, relative_to=None) -> Path:
        text = str(raw)
        if "://" in text or text.startswith(("file:", "mlflow-artifacts:")):
            raise ValueError("Remote/URI paths are not approved local evidence")
        path = Path(text)
        for prefix in ("/home/jihua/quoridor-data", "/quoridor-data", "/data/quoridor"):
            if text == prefix or text.startswith(prefix + "/"):
                path = self.data_root / text[len(prefix):].lstrip("/")
                break
        if not path.is_absolute():
            path = (relative_to or self.data_root) / path
        path = path.resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError("Path outside explicitly approved roots")
        return path

    def _json(self, path):
        return json.loads(self._path(path).read_text())

    def _sources(self):
        entries, manifest, warnings = [], [], []
        sources = [*(dict(s, kind="catalog") for s in self.catalogs),
                   dict(name="registry", path=self.registry_path, kind="registry")]
        for source in sources:
            report = dict(source, status="unavailable", count=0)
            manifest.append(report)
            try:
                path = self._path(source["path"])
                report["resolved_path"] = str(path)
                data = self._json(path)
                models = data.get("models") if isinstance(data, dict) else None
                if not isinstance(models, list):
                    raise ValueError("Expected models list")
                report.update(status="available", count=len(models))
                for item in models:
                    if not isinstance(item, dict) or not _sha(item.get("sha256")):
                        warnings.append(f"Invalid model SHA in {path}")
                        continue
                    entries.append(dict(item=item, source=dict(source, path=str(path))))
            except (OSError, ValueError, TypeError) as exc:
                report["reason"] = str(exc)
                warnings.append(f"{source['name']}: {exc}")
        specs = []
        report = dict(name="agent_specs", kind="agent_specs", path=self.agent_spec_dir,
                      status="unavailable", count=0)
        manifest.append(report)
        try:
            directory = self._path(self.agent_spec_dir)
            if not directory.is_dir():
                raise ValueError("AgentSpec directory missing")
            for candidate in sorted(directory.iterdir()):
                if candidate.suffix.lower() not in (".json", ".yaml", ".yml"):
                    continue
                try:
                    path = self._path(candidate)
                    if path.suffix == ".json":
                        data = json.loads(path.read_text())
                    else:
                        import yaml
                        try:
                            data = yaml.safe_load(path.read_text())
                        except yaml.YAMLError as exc:
                            raise ValueError(f"Malformed AgentSpec YAML: {exc}") from exc
                    if not isinstance(data, dict):
                        raise ValueError("Expected AgentSpec object")
                    algo = data.get("algo", {})
                    params = algo.get("params", {})
                    digest = _sha(params.get("sha256"))
                    if digest:
                        spec = dict(data=data, path=str(path), sha256=digest)
                        specs.append(spec)
                        entries.append(dict(item=dict(id=data.get("id"), sha256=digest,
                            checkpoint=params.get("checkpoint")),
                            source=dict(name="agent_spec", kind="agent_spec", path=str(path))))
                except (OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
                    warnings.append(f"AgentSpec {candidate.name}: {exc}")
            report.update(status="available", count=len(specs))
        except (OSError, ValueError) as exc:
            report["reason"] = str(exc)
            warnings.append(f"AgentSpecs: {exc}")
        return entries, specs, manifest, warnings

    @staticmethod
    def _metadata(item):
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            return {}, ["metadata must be an object"]
        config = metadata.get("model_config", {})
        if not isinstance(config, dict):
            return {}, ["model_config must be an object"]
        result = {k: v for k, v in metadata.items() if k != "model_config" and v is not None}
        result.update({"model_config." + k: v for k, v in config.items() if v is not None})
        if "model_arch" not in config and "obs_size" in config:
            result["model_config.model_arch"] = "mlp"
        for key in ("architecture", "output_semantics", "observation_version", "action_space_version"):
            if item.get(key) is not None:
                result["model_config.model_arch" if key == "architecture" else key] = item[key]
        return result, []

    def _hash(self, path: Path) -> str:
        def stamp(stat):
            return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
                    stat.st_mtime_ns, stat.st_ctime_ns)
        before = stamp(path.stat())
        with self._lock:
            if before in self._hash_cache:
                return self._hash_cache[before]
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if stamp(path.stat()) != before:
                raise ValueError("Checkpoint changed while hashing")
            # Keep at most one stat identity per path, with a bounded total cache.
            self._hash_cache = {k: v for k, v in self._hash_cache.items() if k[0] != str(path)}
            if len(self._hash_cache) >= 1024:
                self._hash_cache.clear()
            self._hash_cache[before] = digest
            return digest

    @staticmethod
    def _ref(ref, evidence):
        if not isinstance(ref, dict) or not isinstance(ref.get("run_id"), str) or not ref["run_id"]:
            return None
        source = ref.get("source")
        if source not in ("mlflow", "legacy"):
            return None
        return dict(source=source, run_id=ref["run_id"], evidence=evidence)

    def _run_links(self, warnings):
        """Only checkpoint SHA evidence joins models to runs; never display names."""
        links = {}
        if self.service is None:
            return links
        try:
            listing = self.service.listing()
            warnings.extend(listing.get("warnings", []))
            for run in listing.get("runs", []):
                try:
                    ref = self._ref(run, "run_checkpoint_sha")
                    if not ref:
                        continue
                    provenance = run.get("provenance", {})
                    if run.get("source") == "mlflow" and "catalog_matches" in provenance:
                        # UnifiedRunService already exposes exact SHA catalog matches.
                        # Its detail() also computes benchmarks/replays, and may hash
                        # files: never invoke that expensive path just for lineage.
                        checkpoints = provenance["catalog_matches"]
                    else:
                        detail = self.service.provider_for(run["run_id"]).detail(run["run_id"])
                        checkpoints = detail.get("checkpoints", [])
                    for checkpoint in checkpoints:
                        digest = _sha(checkpoint.get("sha256"))
                        if digest:
                            links.setdefault(digest, []).append(dict(ref,
                                checkpoint=checkpoint.get("filename"), checkpoint_sha256=digest))
                except (KeyError, OSError, ValueError, TypeError, AttributeError) as exc:
                    warnings.append(f"Run lineage unavailable: {exc}")
        except (KeyError, OSError, ValueError, TypeError, AttributeError) as exc:
            warnings.append(f"Run service unavailable: {exc}")
        return links

    def listing(self, *, verify_sha: bool = False) -> dict:
        entries, specs, manifest, warnings = self._sources()
        groups, identities = {}, {}
        for entry in entries:
            item = entry["item"]
            digest = item["sha256"]
            groups.setdefault(digest, []).append(entry)
            if isinstance(item.get("id"), str):
                identities.setdefault(item["id"], set()).add(digest)
        links = self._run_links(warnings)
        rows = [self._model(digest, group, identities, specs, links.get(digest, []), verify_sha)
                for digest, group in sorted(groups.items())]
        return dict(models=rows, count=len(rows), sources=manifest, warnings=warnings,
                    verification_requested=verify_sha)

    def detail(self, sha256: str, *, verify_sha: bool = False) -> dict:
        if not _sha(sha256):
            raise KeyError("Model identity must be a full lowercase SHA-256")
        # Only hash the requested model, never all checkpoints on a detail request.
        entries, specs, manifest, warnings = self._sources()
        identities = {}
        for entry in entries:
            identifier = entry["item"].get("id")
            if isinstance(identifier, str):
                identities.setdefault(identifier, set()).add(entry["item"]["sha256"])
        group = [e for e in entries if e["item"]["sha256"] == sha256]
        if not group:
            raise KeyError(sha256)
        links = self._run_links(warnings)
        return dict(self._model(sha256, group, identities, specs, links.get(sha256, []), verify_sha),
                    sources=manifest, warnings=warnings)

    def _model(self, digest, group, identities, specs, run_refs, verify):
        aliases, evidence, checkpoints, conflicts, values, registries = set(), [], {}, [], {}, []
        run_refs = list(run_refs)
        for entry in group:
            item, source = entry["item"], entry["source"]
            identifier = item.get("id")
            if isinstance(identifier, str):
                aliases.add(identifier)
                if len(identities[identifier]) > 1:
                    conflicts.append(f"id_sha_conflict:{identifier}")
            evidence.append(dict(source=source, record=item))
            metadata, errors = self._metadata(item)
            conflicts.extend(errors)
            for key, value in metadata.items():
                encoded = json.dumps(value, sort_keys=True)
                values.setdefault(key, {})[encoded] = value
            if source["kind"] == "registry":
                registries.append(dict(item))
            ref = item.get("run_ref")
            if not ref and item.get("run_id"):
                ref = dict(source="mlflow" if source["kind"] == "registry" else item.get("source"),
                           run_id=item["run_id"])
            ref = self._ref(ref, source["kind"] + "_sha")
            if ref:
                run_refs.append(dict(ref, checkpoint_sha256=digest))
            if item.get("checkpoint"):
                raw = item["checkpoint"]
                try:
                    path = self._path(raw, relative_to=Path(source["path"]).parent)
                    state = dict(path=str(path), available=path.is_file(), sha_verified=False,
                                 reason="not_requested")
                    if state["available"] and verify:
                        actual = self._hash(path)
                        state.update(actual_sha256=actual, sha_verified=actual == digest,
                                     reason="verified" if actual == digest else "sha_mismatch")
                    elif not state["available"]:
                        state["reason"] = "checkpoint_missing"
                except (OSError, ValueError, TypeError) as exc:
                    state = dict(path=str(raw), available=False, sha_verified=False, reason=str(exc))
                checkpoints[state["path"]] = state
        conflicts += ["metadata_conflict:" + key for key, variants in values.items() if len(variants) > 1]
        metadata = {key: next(iter(variants.values())) for key, variants in values.items() if len(variants) == 1}
        architecture = metadata.get("model_config.model_arch")
        algorithm = metadata.get("model_config.algorithm")
        semantics = metadata.get("output_semantics")
        inferred = "policy_logits" if algorithm == "policy_distillation" else (
            "q_values" if algorithm in ("dqn", "double_dqn", "ddqn") else None)
        if semantics and inferred and semantics != inferred:
            conflicts.append("metadata_conflict:output_semantics")
        semantics = semantics or inferred
        paths = list(checkpoints.values())
        available = any(p["available"] for p in paths)
        verified = any(p["sha_verified"] for p in paths)
        if any(p["reason"] == "sha_mismatch" for p in paths):
            conflicts.append("checkpoint_sha_mismatch")
        conflicts = sorted(set(conflicts))
        blockers = (["checkpoint_missing_or_unapproved"] if not available else []) + conflicts
        analysis_blockers = list(blockers)
        if architecture not in ("cnn", "mlp"):
            analysis_blockers.append("unsupported_or_unknown_architecture")
        if metadata.get("observation_version") not in ("dqn_obs_v1", "dqn_obs_cnn_v1"):
            analysis_blockers.append("unsupported_or_unknown_observation")
        if metadata.get("action_space_version") != "dqn_action_v1" or metadata.get("action_count") != 209:
            analysis_blockers.append("unsupported_or_unknown_action_space")
        matching_specs = [s for s in specs if s["sha256"] == digest]
        def spec_checkpoint_available(spec):
            try:
                raw = spec["data"]["algo"]["params"]["checkpoint"]
                path = self._path(raw, relative_to=Path(spec["path"]).parent)
                return checkpoints.get(str(path), {}).get("available", False)
            except (KeyError, OSError, TypeError, ValueError):
                return False

        playable_specs = [s for s in matching_specs if
            s["data"].get("algo", {}).get("type") == "dqn" and
            s["data"].get("policy", {}).get("type") == "top_k" and
            s["data"].get("policy", {}).get("k") == 1 and spec_checkpoint_available(s)]
        play_blockers = list(analysis_blockers)
        if not playable_specs:
            play_blockers.append("no_pinned_top1_agent_spec")
        observed = self.deployed_shas.get(digest)
        deployment_blockers = list(blockers)
        if not observed:
            deployment_blockers.append("deployment_not_observed")
        analysis_notes = ["static_eligibility_only", "requires_isolated_research_catalog",
                          "actual_load_not_verified"]
        if not verified:
            analysis_notes.append("sha_not_verified")
        unique_refs = {json.dumps(ref, sort_keys=True): ref for ref in run_refs}
        reasons = dict(available=[] if available else ["checkpoint_missing_or_unapproved"],
            sha_verified=[] if verified else ["not_requested" if not verify else "no_matching_checkpoint_hash"],
            analyzable=analysis_blockers + analysis_notes,
            playable=play_blockers + ["static_eligibility_only", "actual_load_not_verified"] +
                ([] if verified else ["sha_not_verified"]),
            deployed=deployment_blockers)
        research_blockers = list(analysis_blockers)
        if architecture != "cnn":
            research_blockers.append("research_play_requires_cnn")
        if semantics not in ("policy_logits", "q_values"):
            research_blockers.append("unsupported_output_semantics")
        if not any(entry["source"]["kind"] == "catalog" for entry in group):
            research_blockers.append("not_in_approved_catalog")
        reasons["research_playable"] = research_blockers + ["static_only_runtime_load_required"]
        return dict(id=digest, sha256=digest, aliases=sorted(aliases), metadata=metadata,
            architecture=architecture, output_semantics=semantics, conflicts=conflicts,
            evidence=evidence, checkpoints=paths, registry=registries,
            agent_specs=[dict(path=s["path"], spec=s["data"]) for s in matching_specs],
            run_refs=list(unique_refs.values()), available=available, sha_verified=verified,
            analyzable=not analysis_blockers, playable=not play_blockers,
            research_playable=not research_blockers,
            deployed=not deployment_blockers, deployment_evidence=observed,
            actual_load_verified=False, requires_isolated_research_catalog=True, reasons=reasons)