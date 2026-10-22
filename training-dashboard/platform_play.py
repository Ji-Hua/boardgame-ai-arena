"""Fixed-upstream gateway for the existing Gameplay Backend research API."""
import asyncio
import os
import re
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response


def install_play_gateway(app, *, transport=None):
    upstream = os.environ.get("PLATFORM_GAMEPLAY_UPSTREAM", "")
    if not upstream:
        return
    parsed = urlsplit(upstream)
    if (parsed.scheme != "http" or parsed.hostname not in ("192.168.68.102", "127.0.0.1")
            or parsed.port not in (8774, 8776) or parsed.path not in ("", "/")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Play upstream must be a configured isolated Gameplay Backend")
    upstream = upstream.rstrip("/")
    slots = asyncio.Semaphore(8)

    @app.api_route("/play-api/games{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy(request: Request, path: str):
        if not re.fullmatch(r"(?:/[a-f0-9-]{36}(?:/(?:actions|retry|surrender))?)?", path):
            raise HTTPException(404, "Unknown Play path")
        allowed = ("POST",) if not path or path.endswith(("/actions", "/retry", "/surrender")) else ("GET", "DELETE")
        if request.method not in allowed or request.url.query:
            raise HTTPException(405, "Method or query not allowed")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin Play writes are not allowed")
        if slots.locked():
            raise HTTPException(429, "Play gateway busy")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 8192:
                raise HTTPException(413, "Play request too large")
        headers = {name: request.headers[name] for name in ("authorization", "content-type") if name in request.headers}
        headers["X-Research-Gateway"] = os.environ["QUORIDOR_RESEARCH_SERVICE_KEY"]
        async with slots:
            try:
                async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False, timeout=40) as client:
                    result = await client.request(request.method,
                        upstream + "/api/research/games" + path, content=bytes(body), headers=headers)
                if len(result.content) > 2 * 1024 * 1024:
                    raise HTTPException(502, "Play response too large")
                if 300 <= result.status_code < 400:
                    raise HTTPException(502, "Unexpected Gameplay redirect")
                return Response(result.content, status_code=result.status_code,
                    media_type="application/json", headers={"Cache-Control": "no-store"})
            except httpx.TimeoutException as exc:
                raise HTTPException(504, "Gameplay Backend timed out; reconnect before retrying") from exc
            except httpx.HTTPError as exc:
                raise HTTPException(502, "Isolated Dev Gameplay Backend unavailable") from exc