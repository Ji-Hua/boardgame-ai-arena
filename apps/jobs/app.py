"""HTTP surface of the Job controller, served behind the platform gateway at /api/v1/jobs.

Reads are open on the trusted LAN; submitting and cancelling require the shared
platform token (``X-Platform-Token``). Without a token file writes are disabled.
"""
from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path

from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from .docker import gpu_status
from .store import ACTIVE


class Promotion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: str = Field(max_length=64)
    season: str | None = Field(default=None, max_length=80)
    note: str = Field(max_length=2000)


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entrypoint: str = Field(max_length=80)
    config: str | dict = Field()
    name: str | None = Field(default=None, max_length=80)
    experiment: str | None = Field(default=None, max_length=80)
    source: str | None = Field(default=None, max_length=200)
    resources: dict[str, float] | None = None
    follow_up: dict[str, str] | None = None


def create_app(controller, token_file=None, stream_interval=2.0, heartbeat=15.0):
    app = FastAPI(title='Quoridor platform Jobs')
    router = APIRouter(prefix='/api/v1/jobs')

    def authorize(token):
        expected = Path(token_file).read_text().strip() if token_file and Path(token_file).is_file() else ''
        if not expected:
            raise HTTPException(403, 'Job writes are disabled: no platform token is configured')
        if not hmac.compare_digest(token.encode(), expected.encode()):
            raise HTTPException(401, 'Invalid platform token')

    def call(fn, *args):
        try:
            return fn(*args)
        except KeyError as exc:
            raise HTTPException(404, 'Job, entrypoint or template not found') from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get('')
    def listing(experiment: str | None = None):
        jobs = controller.store.listing(experiment=experiment)
        return dict(jobs=jobs, counts={s: sum(1 for j in jobs if j['status'] == s) for s in
                                       ('queued', 'preparing', 'running', 'succeeded', 'failed', 'cancelled')})

    @router.get('/system')
    def system():
        return dict(gpu=gpu_status(), image=controller.image, source_ref=controller.source_ref,
                    max_cpu_jobs=controller.max_cpu_jobs, writes_enabled=bool(token_file and Path(token_file).is_file()))

    @router.get('/entrypoints')
    def entrypoints(): return call(controller.entrypoints)

    @router.get('/templates/{entrypoint}/{name:path}', response_class=PlainTextResponse)
    def template(entrypoint: str, name: str): return call(controller.template, entrypoint, name)

    @router.post('')
    def submit(body: Submission, x_platform_token: str = Header(default='')):
        authorize(x_platform_token)
        return call(controller.submit, body.model_dump())

    @router.get('/{identifier}')
    def detail(identifier: str): return call(controller.detail, identifier)

    @router.get('/{identifier}/log', response_class=PlainTextResponse)
    def log(identifier: str, tail: int = 400): return call(controller.log_text, identifier, max(1, min(tail, 5000)))

    @router.get('/{identifier}/metrics')
    def metrics(identifier: str, after: int = 0): return call(controller.metrics, identifier, 5000, max(0, after))

    @router.get('/{identifier}/stream')
    async def stream(identifier: str):
        """Server-sent events: ``update`` carries new metric rows (from ``offset``), status, evaluations and alerts;
        ``end`` follows the last update of a finished Job. Comment lines keep idle proxies open."""
        if controller.store.get(identifier) is None:
            raise HTTPException(404, 'Job not found')

        async def events():
            after, last, idle = 0, None, 0.0
            while True:
                job = await run_in_threadpool(controller.store.get, identifier)
                data = await run_in_threadpool(controller.metrics, identifier, 5000, after)
                state = (job['status'], data['total'], len(data['evaluations']), json.dumps(data['alerts']))
                if state != last or data['metrics']:
                    last, idle, after = state, 0.0, data['total']
                    yield 'event: update\ndata: ' + json.dumps(dict(data, status=job['status'])) + '\n\n'
                elif idle >= heartbeat:
                    idle = 0.0
                    yield ': keep-alive\n\n'
                if job['status'] not in ACTIVE:
                    yield 'event: end\ndata: {}\n\n'
                    return
                await asyncio.sleep(stream_interval)
                idle += stream_interval
        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @router.post('/{identifier}/cancel')
    def cancel(identifier: str, x_platform_token: str = Header(default='')):
        authorize(x_platform_token)
        return call(controller.cancel, identifier)

    promotions = APIRouter(prefix='/api/v1/promotions')

    @promotions.get('')
    def promotion_log():
        return dict(deployed=list(controller.promotions.current().values()), history=controller.promotions.history())

    @promotions.get('/candidates/{sha}')
    def promotion_candidates(sha: str):
        return dict(sha256=sha, deployed=controller.promotions.current().get(sha), gates=call(controller.promotions.gates, sha))

    @promotions.post('')
    def promote(body: Promotion, x_platform_token: str = Header(default='')):
        authorize(x_platform_token)
        return call(controller.promotions.promote, body.sha256, body.season, body.note)

    @promotions.post('/withdraw')
    def withdraw(body: Promotion, x_platform_token: str = Header(default='')):
        authorize(x_platform_token)
        return call(controller.promotions.withdraw, body.sha256, body.note)

    @app.get('/health')
    def health(): return dict(status='ok', service='jobs')

    app.include_router(router)
    app.include_router(promotions)
    return app
