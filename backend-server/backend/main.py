"""Backend server entry point.

Usage:
    python -m backend.main
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.transport.rest.routes import router as rest_router
from backend.transport.websocket.handler import ws_router
from backend.application.room_manager import RoomManager
from backend.application.game_manager import GameManager
from backend.runtime.broadcast import BroadcastHub


def create_app() -> FastAPI:
    app = FastAPI(title="Quoridor Backend", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared application state
    app.state.room_manager = RoomManager()
    app.state.game_manager = GameManager()
    app.state.broadcast_hub = BroadcastHub()

    # REST routes
    app.include_router(rest_router)
    # WebSocket routes
    app.include_router(ws_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
