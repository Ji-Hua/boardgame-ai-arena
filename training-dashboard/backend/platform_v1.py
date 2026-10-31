"""Single-entry /api/v1 surface. P0: one read-only model index keyed by checkpoint SHA.

Capability states stay separate and are never inferred from each other:
available -> analyzable / playable -> ranked (appears in a complete season) -> deployed.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

CAPABILITIES = ('available', 'sha_verified', 'analyzable', 'playable', 'research_playable', 'ranked', 'deployed')


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

    def call(fn, *args):
        try:
            return fn(*args)
        except KeyError as exc:
            raise HTTPException(404, 'Model not found') from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError, ImportError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get('/models')
    def models(): return call(model_listing, app.state.models, app.state.evidence)

    @router.get('/models/{sha}')
    def model(sha: str): return call(model_detail, app.state.models, app.state.evidence, sha)

    app.include_router(router)
