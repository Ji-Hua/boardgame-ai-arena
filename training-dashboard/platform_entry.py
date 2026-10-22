"""Isolated unified platform: existing read-only API, built SPA and loopback Lab.

Uvicorn entry: platform_entry:create_app --factory --app-dir training-dashboard.
Importing this module neither imports a generic ``backend`` nor starts analysis.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette.routing import Route


ROOT = Path(__file__).resolve().parents[1]
BACKEND_PACKAGE = "_quoridor_platform_dashboard_backend"
MAX_BODY = 1024 * 1024
MAX_RESPONSE = 16 * 1024 * 1024
PROXY_SECONDS = 120
POST_ROUTES = frozenset({
    "/api/position", "/api/analyze", "/api/branch", "/api/apply",
    "/api/neural/tensor", "/api/neural/search", "/api/multi-compare",
})
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"]


def dashboard_module(name: str):
    """Load this worktree's dashboard under a private package, not backend-server.

    Relative imports stay private even when another service has already imported
    ``backend``. Never evict or replace modules owned by another application.
    """
    directory = Path(__file__).resolve().parent / "backend"
    if BACKEND_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            BACKEND_PACKAGE, directory / "__init__.py",
            submodule_search_locations=[str(directory)])
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load dashboard package")
        package = importlib.util.module_from_spec(spec)
        sys.modules[BACKEND_PACKAGE] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop(BACKEND_PACKAGE, None)
            raise
    return importlib.import_module(f"{BACKEND_PACKAGE}.{name}")


def research_catalog(directory: Path, *, index=None) -> Path:
    """Write deterministic external evidence; no hashing/loading of checkpoints.

    Full SHA IDs avoid cross-catalog alias collisions. Ambiguous and unavailable
    entries are excluded, not repaired. The platform keeps its original index.
    """
    if index is None:
        index = dashboard_module("platform_models").ModelIndex()
    directory = Path(directory).resolve()
    approved = (index.data_root / "platform-p1-p2").resolve()
    if not directory.is_relative_to(approved) or directory.is_relative_to(ROOT):
        raise ValueError("Research catalog must be outside Git under data_root/platform-p1-p2")
    listing = index.listing(verify_sha=False)
    models = []
    for row in listing["models"]:
        if not row["available"] or row["conflicts"]:
            continue
        evidence = [item for item in row["evidence"] if item["source"]["kind"] == "catalog"]
        if not evidence:
            continue
        paths = sorted(item["path"] for item in row["checkpoints"] if item["available"])
        record = copy.deepcopy(evidence[0]["record"])
        metadata, config = {}, {}
        for key, value in row["metadata"].items():
            if key.startswith("model_config."):
                config[key.removeprefix("model_config.")] = value
            else:
                metadata[key] = value
        if config:
            metadata["model_config"] = config
        record.update(id=row["sha256"], sha256=row["sha256"], checkpoint=paths[0],
                      metadata=metadata, aliases=row["aliases"], source_evidence=evidence,
                      static_analyzable=row["analyzable"],
                      static_analysis_reasons=row["reasons"]["analyzable"])
        models.append(record)
    document = dict(schema_version=1, purpose="isolated_research_only",
                    verification="static_only_no_checkpoint_load_or_hash",
                    sources=listing["sources"], warnings=listing["warnings"], models=models)
    payload = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"research-catalog-{digest}.json"
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("Existing content-addressed research catalog differs")
        return target
    # Atomic publication prevents the Lab from seeing a half-written catalog.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def loopback_upstream(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.port is None or not 1024 <= parsed.port <= 65535):
        raise ValueError("Lab upstream must be http://127.0.0.1:<unprivileged-port>")
    return f"http://127.0.0.1:{parsed.port}"


def create_app(*, backend_app=None, dist=None, lab_upstream=None, transport=None) -> FastAPI:
    """Factory with injectable backend/transport for isolated, network-free tests."""
    dist = Path(dist or os.environ.get(
        "PLATFORM_FRONTEND_DIST", ROOT / "training-dashboard/frontend/dist")).resolve()
    if not (dist / "index.html").is_file():
        raise RuntimeError(
            f"Built frontend missing: {dist}/index.html. "
            "Run npm run build in training-dashboard/frontend (install dependencies explicitly first).")
    upstream = loopback_upstream(lab_upstream or os.environ.get(
        "PLATFORM_LAB_UPSTREAM", "http://127.0.0.1:8754"))
    if backend_app is None:
        backend_app = dashboard_module("app").app
    slots = asyncio.Semaphore(8)

    @asynccontextmanager
    async def lifespan(app):
        async with backend_app.router.lifespan_context(backend_app):
            async with httpx.AsyncClient(
                transport=transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(PROXY_SECONDS, connect=3, write=10, pool=3),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            ) as client:
                app.state.lab_client = client
                yield

    app = FastAPI(title="Quoridor isolated research platform", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/research-lab", methods=METHODS)
    async def lab_slash(request: Request):
        if request.method not in ("GET", "HEAD"):
            raise HTTPException(405, "Only GET/HEAD are allowed here")
        query = request.scope.get("query_string", b"")
        if len(query) > 8192:
            raise HTTPException(414, "Query too long")
        suffix = ("?" + query.decode("ascii")) if query else ""
        # A fixed relative location cannot reflect an attacker-supplied Host/URL.
        return RedirectResponse("/research-lab/" + suffix, status_code=307)

    @app.api_route("/research-lab/{path:path}", methods=METHODS)
    async def lab_proxy(request: Request, path: str):
        target = "/" + path
        query = request.scope.get("query_string", b"")
        if len(path) + len(query) > 8192:
            raise HTTPException(414, "Request target too long")
        # Check the decoded path too: no normalization, traversal, double-encoded
        # paths, authority syntax, arbitrary URLs, or browser-selected upstreams.
        if (not re.fullmatch(r"[A-Za-z0-9_./-]*", path)
            or (path and any(part in (".", "..", "") for part in path.split("/")))):
            raise HTTPException(404, "Unknown Lab path")
        reads = target in ("/", "/health", "/api/meta", "/api/models") or target.startswith("/assets/")
        if not reads and target not in POST_ROUTES:
            raise HTTPException(404, "Unknown Lab path")
        allowed = ("GET", "HEAD") if reads else ("POST",)
        if request.method not in allowed:
            raise HTTPException(405, "Method not allowed for this Lab path", headers={"Allow": ", ".join(allowed)})
        if slots.locked():
            raise HTTPException(429, "Lab proxy is busy")
        async with slots:
            try:
                async with asyncio.timeout(PROXY_SECONDS):
                    length = request.headers.get("content-length")
                    if length is not None:
                        try:
                            size = int(length)
                        except ValueError:
                            raise HTTPException(400, "Invalid content length")
                        if size < 0 or size > MAX_BODY:
                            raise HTTPException(413, "Lab request body too large")
                    body = bytearray()
                    async with asyncio.timeout(10):
                        async for chunk in request.stream():
                            body.extend(chunk)
                            if len(body) > MAX_BODY:
                                raise HTTPException(413, "Lab request body too large")
                    if request.method in ("GET", "HEAD") and body:
                        raise HTTPException(400, "Read requests cannot carry a body")
                    headers = {name: request.headers[name] for name in ("accept", "content-type")
                               if name in request.headers}
                    headers["accept-encoding"] = "identity"
                    url = httpx.URL(upstream + target).copy_with(query=query or None)
                    async with app.state.lab_client.stream(
                        request.method, url, headers=headers, content=bytes(body)
                    ) as result:
                        # Lab's canonical allowlisted endpoints need no redirects.
                        # Never follow or reflect a Location from the upstream.
                        if 300 <= result.status_code < 400 and result.status_code != 304:
                            raise HTTPException(502, "Unexpected Lab redirect")
                        content = bytearray()
                        async for chunk in result.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > MAX_RESPONSE:
                                raise HTTPException(502, "Lab response too large")
                        response_headers = {name: result.headers[name] for name in (
                            "content-type", "cache-control", "etag", "last-modified") if name in result.headers}
                        return Response(bytes(content), status_code=result.status_code, headers=response_headers)
            except (TimeoutError, httpx.TimeoutException) as exc:
                raise HTTPException(504, "Lab request timed out") from exc
            except httpx.HTTPError as exc:
                raise HTTPException(502, "Isolated Lab unavailable") from exc

    # Route (not Mount) preserves /api in the ASGI scope for the existing app.
    # The API's own 404/405 responses must never become frontend index.html.
    app.router.routes.extend([
        Route("/api", backend_app, methods=METHODS),
        Route("/api/{path:path}", backend_app, methods=METHODS),
        Route("/health", backend_app, methods=METHODS),
    ])

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def frontend(request: Request, path: str):
        target = (dist / path).resolve()
        if not target.is_relative_to(dist):
            raise HTTPException(404, "Not found")
        if target.is_file():
            return FileResponse(target)
        # Only HTML navigation gets history fallback; missing JS/API/assets do not.
        if (path.startswith(("assets/", "api/", "research-lab/"))
                or Path(path).suffix or "text/html" not in request.headers.get("accept", "")):
            raise HTTPException(404, "Not found")
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-lab-catalog", type=Path, required=True)
    arguments = parser.parse_args()
    print(research_catalog(arguments.generate_lab_catalog))