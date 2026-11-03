"""Run the gameplay service: ``python -m apps.gameplay`` from the repository (or platform image) root.

Environment: QUORIDOR_DATA_ROOT (season evidence and model catalogs, read only),
GAMEPLAY_DB (default <data root>/games/games.sqlite, the only file it writes),
GAMEPLAY_HOST/GAMEPLAY_PORT (default 127.0.0.1:18797), PLATFORM_MODEL_CATALOGS.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn

from .app import create_app
from .games import Games
from .store import Store


def build(env=os.environ):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'training-dashboard'))
    from platform_entry import dashboard_module
    data_root = env.get('QUORIDOR_DATA_ROOT', '/home/jihua/quoridor-data')
    models = dashboard_module('platform_models').ModelIndex()
    evidence = dashboard_module('agent_arena').Evidence(data_root, models)
    adapter = dashboard_module('replay')._dependencies()[0]
    store = Store(env.get('GAMEPLAY_DB', str(Path(data_root) / 'games' / 'games.sqlite')))
    from .research import ResearchGames
    research = ResearchGames(env.get('GAMEPLAY_RESEARCH_DB',
        str(Path(data_root) / 'games' / 'research.sqlite')),data_root)
    return create_app(Games(store, evidence, adapter._serialize_state),research)


def main():
    uvicorn.run(build(), host=os.environ.get('GAMEPLAY_HOST', '127.0.0.1'),
                port=int(os.environ.get('GAMEPLAY_PORT', '18797')), log_level='info')


if __name__ == '__main__':
    main()
