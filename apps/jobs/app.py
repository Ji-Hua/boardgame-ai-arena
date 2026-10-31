"""HTTP surface of the Job controller, served behind the platform gateway at /api/v1/jobs.

Reads are open on the trusted LAN; submitting and cancelling require the shared
platform token (``X-Platform-Token``). Without a token file writes are disabled.
"""
from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from .docker import gpu_status


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entrypoint: str = Field(max_length=80)
    config: str | dict = Field()
    name: str | None = Field(default=None, max_length=80)
    experiment: str | None = Field(default=None, max_length=80)
    source: str | None = Field(default=None, max_length=200)
    resources: dict[str, float] | None = None
    follow_up: dict[str, str] | None = None


def create_app(controller, token_file=None):
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
    def metrics(identifier: str): return call(controller.metrics, identifier)

    @router.post('/{identifier}/cancel')
    def cancel(identifier: str, x_platform_token: str = Header(default='')):
        authorize(x_platform_token)
        return call(controller.cancel, identifier)

    @app.get('/health')
    def health(): return dict(status='ok', service='jobs')

    app.include_router(router)
    return app
