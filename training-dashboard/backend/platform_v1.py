"""Single-entry /api/v1 surface: read-only model index and experiment index.

Capability states stay separate and are never inferred from each other:
available -> analyzable / playable -> ranked (appears in a complete season) -> deployed.
"""
from __future__ import annotations

import os
import threading
import time

from fastapi import APIRouter, HTTPException

from .experiments import Experiments

CAPABILITIES = ('available', 'sha_verified', 'analyzable', 'playable', 'research_playable', 'ranked', 'deployed')


class TimedCache:
    """Keeps one value for ``ttl`` seconds; the model listing joins MLflow lineage and is slow cold."""

    def __init__(self, ttl=60.0, clock=time.monotonic):
        self.ttl, self.clock, self.lock, self.value, self.stamp = ttl, clock, threading.Lock(), None, None

    def get(self, compute):
        with self.lock:
            if self.stamp is None or self.clock() - self.stamp >= self.ttl:
                self.value, self.stamp = compute(), self.clock()
            return self.value


def _ranked(row, rankings):
    seasons = rankings.get(row['sha256'], [])
    return dict(row, ranked=bool(seasons), seasons=seasons)


def model_listing(models, evidence):
    listing = models.listing()
    rankings, warnings = evidence.rankings()
    rows = [_ranked(row, rankings) for row in listing['models']]
    summary = dict(total=len(rows), **{k: sum(1 for row in rows if row.get(k)) for k in CAPABILITIES})
    return dict(listing, models=rows, summary=summary, warnings=listing['warnings'] + warnings)


def model_detail(models, evidence, sha):
    detail = models.detail(sha, verify_sha=True)
    rankings, warnings = evidence.rankings()
    return dict(_ranked(detail, rankings), warnings=detail['warnings'] + warnings)


def install(app):
    router = APIRouter(prefix='/api/v1')
    cache = TimedCache()
    experiments = Experiments(os.environ.get('QUORIDOR_DATA_ROOT', '/data/quoridor'))
    listing = lambda: cache.get(lambda: model_listing(app.state.models, app.state.evidence))

    def call(fn, *args):
        try:
            return fn(*args)
        except KeyError as exc:
            raise HTTPException(404, 'Model or experiment not found') from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError, ImportError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get('/models')
    def models(): return call(listing)

    @router.get('/models/{sha}')
    def model(sha: str): return call(model_detail, app.state.models, app.state.evidence, sha)

    @router.get('/experiments')
    def experiment_listing(): return call(experiments.listing)

    @router.get('/experiments/{slug}')
    def experiment(slug: str):
        def detail():
            # Validate the identity before paying for the model listing.
            experiments.directory(slug)
            return experiments.detail(slug, listing()['models'])
        return call(detail)

    app.include_router(router)
