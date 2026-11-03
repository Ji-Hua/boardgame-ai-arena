"""HTTP surface of the gameplay service, served behind the platform gateway at /api/v1/play.

Creating a game is open on the trusted LAN; every later write needs the game's
player token (``X-Game-Token``). The watch token only reads.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, Header, HTTPException

from .games import Busy, Conflict, CreateGame, Denied, GameStep, Listing


def create_app(games, research_games=None):
    app = FastAPI(title='Quoridor platform gameplay', docs_url=None, redoc_url=None)
    router = APIRouter(prefix='/api/v1/play')

    def call(fn, *args):
        try:
            return fn(*args)
        except HTTPException:
            raise
        except Denied as exc:
            raise HTTPException(403, str(exc)) from exc
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except Busy as exc:
            raise HTTPException(429, str(exc)) from exc
        except (KeyError, StopIteration, FileNotFoundError, IndexError) as exc:
            raise HTTPException(404, 'Season or Agent not found') from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError, ImportError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get('')
    def listing(): return games.listing()

    @router.post('')
    def create(request: CreateGame): return call(games.create, request)

    @router.get('/{identifier}')
    def view(identifier: str, x_game_token: str = Header(default='')):
        return call(games.view, identifier, x_game_token)

    @router.post('/{identifier}/step')
    def step(identifier: str, request: GameStep, x_game_token: str = Header(default='')):
        return call(games.step, identifier, x_game_token, request)

    @router.put('/{identifier}/listing')
    def listed(identifier: str, request: Listing, x_game_token: str = Header(default='')):
        return call(games.set_listed, identifier, x_game_token, request.listed)

    @router.delete('/{identifier}')
    def close(identifier: str, x_game_token: str = Header(default='')):
        return call(games.close, identifier, x_game_token)

    @app.get('/health')
    def health(): return dict(ok=True, active=games.store.count('active'))

    if research_games is not None:
        from .research_api import router as research_router
        app.include_router(research_router(research_games))
    app.include_router(router)
    return app
