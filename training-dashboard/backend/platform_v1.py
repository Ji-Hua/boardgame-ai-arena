"""Single-entry /api/v1 surface: read-only model index and experiment index.

Capability states stay separate and are never inferred from each other:
available -> analyzable / playable -> ranked (appears in a complete season) -> deployed.
"""
from __future__ import annotations

import os
import threading
import time

import json
from pathlib import Path

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


DIAGNOSIS_FILES = ('experiments/**/diagnoses.json', 'experiments/**/search-diagnoses.json',
                   'jobs/runs/*/diagnoses.json', 'jobs/runs/*/experiment/diagnoses.json')


def diagnoses_index(data_root):
    """Checkpoint SHA -> every recorded diagnosis of exactly that SHA, with its source file."""
    root = Path(data_root).resolve()
    index, warnings = {}, []
    for pattern in DIAGNOSIS_FILES:
        for path in sorted(root.glob(pattern)):
            try:
                rows = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                warnings.append(f'{path.relative_to(root)}: {exc}')
                continue
            for row in rows if isinstance(rows, list) else []:
                sha = row.get('sha256') if isinstance(row, dict) else None
                if isinstance(sha, str) and len(sha) == 64:
                    counts = row.get('counts') or {}
                    index.setdefault(sha, []).append(dict(source=str(path.relative_to(root)), model_id=row.get('model_id'),
                        games=row.get('games'), wins=counts.get('wins'), losses=counts.get('losses'),
                        missed_immediate_win=counts.get('missed_immediate_win'), immediate_win_available=counts.get('immediate_win_available'),
                        tactical_wins=row.get('tactical_wins'), tactical_count=row.get('tactical_count'),
                        repetition_rate=row.get('repetition_rate'), mean_decision_ms=row.get('mean_decision_ms'),
                        output_semantics=row.get('output_semantics'), search=row.get('search'),
                        book_sha256=row.get('book_sha256'), unique_openings=row.get('unique_openings')))
    return index, warnings


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
    diagnosis_cache = TimedCache()
    data_root = os.environ.get('QUORIDOR_DATA_ROOT', '/data/quoridor')

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

    @router.get('/models/{sha}/diagnoses')
    def model_diagnoses(sha: str):
        def find():
            if not (len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)):
                raise ValueError('Model identity must be a full lowercase SHA-256')
            index, warnings = diagnosis_cache.get(lambda: diagnoses_index(data_root))
            return dict(sha256=sha, diagnoses=index.get(sha, []), warnings=warnings)
        return call(find)

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
