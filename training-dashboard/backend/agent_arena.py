"""Manifest-scoped research evidence and isolated, bounded preview games."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .replay import arena_replay, _stable_hash, _dependencies
from .platform_context import _trajectory


class Evidence:
    def __init__(self, root, models):
        self.root = Path(root).resolve()
        self.models = models
        self.cache = {}
        self.lock = threading.RLock()

    def path(self, path):
        result = Path(path).resolve(strict=True)
        if not result.is_relative_to(self.root):
            raise ValueError('Evidence outside data root')
        return result

    def read(self, path):
        return json.loads(self.path(path).read_text())

    def season(self, season):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', season):
            raise ValueError('Invalid season')
        base = self.root / 'arena' / season
        report, manifest = self.read(base / 'report.json'), self.read(base / 'manifest.json')
        digest = _stable_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        if digest != manifest.get('manifest_sha256') or digest != report.get('manifest_sha256'):
            raise ValueError('Manifest identity mismatch')
        roster = manifest['roster']['agents']
        if len({x['id'] for x in roster}) != len(roster):
            raise ValueError('Duplicate Agent identity')
        return base, report, manifest

    def seasons(self):
        rows, warnings = [], []
        for path in sorted((self.root / 'arena').glob('*/report.json')):
            try:
                _, report, manifest = self.season(path.parent.name)
                if report.get('status') != 'complete':
                    continue
                rows.append(dict(id=path.parent.name, run_id=report.get('run_id'),
                    updated_at=report.get('updated_at'), protocol=report.get('protocol'),
                    agents=len(manifest['roster']['agents'])))
            except (ValueError, OSError, KeyError) as exc:
                warnings.append(f'{path.parent.name}: {exc}')
        return dict(seasons=sorted(rows, key=lambda x: x['updated_at'] or '', reverse=True), warnings=warnings)

    def roster(self, season):
        _, report, manifest = self.season(season)
        ranks = {x['id']: x for x in report.get('ranking', [])}
        rows = []
        for spec in manifest['roster']['agents']:
            algo = spec['algo']['type']
            rows.append(dict(id=spec['id'], name=spec.get('display_name', spec['id']),
                kind='neural' if algo == 'dqn' else 'heuristic' if algo in ('minimax', 'greedy') else 'random',
                sha256=spec['algo'].get('params', {}).get('sha256'), spec=spec,
                identity=_stable_hash(spec), ranking=ranks.get(spec['id']),
                playable=algo in ('dqn', 'greedy', 'minimax', 'random')))
        return dict(season=season, agents=rows, matrix=report.get('matrix', {}),
                    protocol=report.get('protocol'), rating=report.get('rating'),
                    warning=report.get('ranking_warning'), manifest_sha256=manifest['manifest_sha256'])

    def rankings(self):
        """Exact checkpoint SHA -> every Agent appearance in complete seasons; never pooled."""
        listing = self.seasons()
        rows, warnings = {}, list(listing['warnings'])
        for season in listing['seasons']:
            try:
                roster = self.roster(season['id'])
            except (ValueError, OSError, KeyError) as exc:
                warnings.append(f"{season['id']}: {exc}"); continue
            for agent in roster['agents']:
                if not agent['sha256']:
                    continue
                rank = agent['ranking'] or {}
                rows.setdefault(agent['sha256'], []).append(dict(season=season['id'],
                    agent=agent['id'], name=agent['name'], protocol=roster['protocol'],
                    policy=agent['spec'].get('policy'), rank=rank.get('rank'), elo=rank.get('elo'),
                    wins=rank.get('wins'), draws=rank.get('draws'), losses=rank.get('losses')))
        return rows, warnings

    def spec(self, season, agent):
        rows = self.roster(season)['agents']
        return next(x for x in rows if x['id'] == agent)

    def training(self, season, agent):
        row = self.spec(season, agent)
        if row['kind'] != 'neural':
            return dict(runs=[], reason='该类型无需训练', model=None)
        model = self.models.detail(row['sha256'], verify_sha=False)
        return dict(runs=model.get('run_refs', []), model=model,
                    reason=None if model.get('run_refs') else '没有确切 SHA 关联的训练记录')

    def games(self, season, agent, opponent):
        base, _, manifest = self.season(season)
        self.spec(season, agent); self.spec(season, opponent)
        key = (manifest['manifest_sha256'], agent, opponent)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        rows, warnings = [], []
        for file in sorted((base / 'blocks').glob('*.json.gz')):
            with gzip.open(self.path(file), 'rt') as stream:
                block = json.load(stream)
            if {block.get('a'), block.get('b')} != {agent, opponent}:
                continue
            if block.get('manifest_sha256') != manifest['manifest_sha256']:
                warnings.append(f'{file.name}: manifest mismatch'); continue
            for i, game in enumerate(block.get('games', [])):
                if {game.get('p1'), game.get('p2')} != {agent, opponent}:
                    warnings.append(f'{file.name}: seat mismatch'); continue
                rows.append(dict(id=file.name.removesuffix('.json.gz') + ':' + str(i),
                    p1=game['p1'], p2=game['p2'], winner=game.get('winner'),
                    steps=game.get('steps'), termination=game.get('termination'),
                    control=block.get('control'), opening=block.get('opening_id'),
                    available=isinstance(game.get('actions'), list) and bool(game.get('trajectory_hash'))))
        result = dict(games=rows, warnings=warnings)
        with self.lock:
            self.cache[key] = result
        return result

    def replay(self, season, game_id):
        if not re.fullmatch(r'[a-f0-9]{16,64}:[0-9]{1,3}', game_id):
            raise ValueError('Invalid game identity')
        base, _, manifest = self.season(season)
        block_id, number = game_id.split(':')
        with gzip.open(self.path(base / 'blocks' / (block_id + '.json.gz')), 'rt') as stream:
            block = json.load(stream)
        game = block['games'][int(number)]
        spec = next(x for x in manifest['roster']['agents'] if x['id'] == game['p1'])
        ck = dict(sha256=spec['algo'].get('params', {}).get('sha256'), filename=None)
        replay = arena_replay(manifest, block, int(number), game['p1'], season, ck,
                              allow_non_neural=True)
        positions, actions = _trajectory(replay, 0)
        return dict(id=game_id, p1=game['p1'], p2=game['p2'], winner=game['winner'],
            termination=game['termination'], positions=positions, actions=actions,
            source=replay['source'], decision_milliseconds=game.get('decision_milliseconds', {}),
            diagnostics_scope='game_aggregate', decision_diagnostics=game.get('decision_diagnostics', {}))


class CreateGame(BaseModel):
    model_config = ConfigDict(extra='forbid')
    season: str = Field(max_length=120)
    p1: str = Field(max_length=160)
    p2: str = Field(max_length=160)


class GameStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=0)
    action: dict | None = None


class Sessions:
    def __init__(self, evidence, capacity=4):
        self.evidence, self.capacity = evidence, capacity
        self.games = {}
        self.lock = threading.RLock()

    def expire(self):
        for key in list(self.games):
            if self.games[key]['expires'] < time.monotonic():
                del self.games[key]

    def create(self, request):
        from agent_system.runtime.service.yaml_loader import parse_agent_definition, create_agent_from_definition
        from agent_system.analysis.state import ENGINE
        with self.lock:
            self.expire()
            if len(self.games) >= self.capacity:
                raise HTTPException(429, '预览对局已满，请先关闭旧对局')
            instances, identities = [], []
            for seat, identifier in enumerate((request.p1, request.p2), 1):
                if identifier == 'human':
                    instances.append(None); identities.append(dict(id='human', name='玩家', kind='human')); continue
                row = self.evidence.spec(request.season, identifier)
                spec = copy.deepcopy(row['spec'])
                if not row['playable']:
                    raise ValueError('Unsupported Agent')
                if row['kind'] == 'neural':
                    model = self.evidence.models.detail(row['sha256'], verify_sha=True)
                    if not model['available'] or not model['sha_verified'] or model.get('conflicts'):
                        raise ValueError('Unavailable or ambiguous exact checkpoint')
                    path = next(p['path'] for p in model['checkpoints'] if p.get('sha_verified'))
                    path = self.evidence.path(path)
                    if path.stat().st_size > 128 * 1024 * 1024:
                        raise ValueError('Checkpoint exceeds preview size limit')
                    spec['algo']['params'].update(checkpoint=str(path), device='cpu')
                if spec['algo']['type'] == 'minimax' and not 1 <= int(spec['algo'].get('params', {}).get('depth', 2)) <= 3:
                    raise ValueError('Preview supports Minimax depth 1–3')
                instance = create_agent_from_definition(parse_agent_definition(spec), context={'seed': 42})
                if row['kind'] == 'neural' and instance.inference.sha256 != row['sha256']:
                    raise ValueError('Loaded SHA mismatch')
                instances.append(instance); identities.append({**row, 'actual_sha256':row['sha256'] if row['kind']=='neural' else None})
            identifier, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            self.games[identifier] = dict(token=token, instances=instances, identities=identities,
                state=ENGINE.initial_state(), actions=[], positions=[], revision=0, expires=time.monotonic()+14400)
            return dict(id=identifier, token=token, **self.snapshot(self.games[identifier]))

    def get(self, identifier, token):
        self.expire()
        game = self.games.get(identifier)
        if game is None or not secrets.compare_digest(game['token'], token or ''):
            raise HTTPException(403, '对局不存在、已过期或无访问凭据')
        return game

    def snapshot(self, game):
        from agent_system.analysis.state import snapshot
        return {**snapshot(game['state']), 'revision':game['revision'], 'identities':game['identities'],
                'actions':game['actions'], 'positions':game['positions'], 'limit':game['revision']>=500}

    def step(self, identifier, token, request):
        from agent_system.analysis.state import ENGINE, Move, wire_action, to_position
        with self.lock:
            game = self.get(identifier, token)
            if request.revision != game['revision']:
                raise HTTPException(409, '对局已更新，请刷新')
            state = game['state']
            if ENGINE.is_game_over(state) or game['revision'] >= 500:
                raise HTTPException(409, '对局已结束')
            position = to_position(state)
            legal = [wire_action(a) for a in ENGINE.legal_actions(state)]
            instance = game['instances'][position['turn']-1]
            if instance is None:
                if request.action is None:
                    raise ValueError('请选择玩家动作')
                action = Move.model_validate(request.action).model_dump(mode='json')
            else:
                if request.action is not None:
                    raise ValueError('Agent 行动不能由客户端指定')
                adapter = _dependencies()[0]
                action = instance.make_action(adapter._serialize_state(state), legal)
            if action not in legal:
                raise ValueError('Engine rejected illegal action')
            new_state = ENGINE.apply_action(state, Move.model_validate(action).engine_action())
            game['positions'].append(position); game['actions'].append(action)
            game['state']=new_state; game['revision']+=1
            return self.snapshot(game)


def install(app):
    evidence = app.state.evidence = Evidence(os.environ.get('QUORIDOR_DATA_ROOT', '/data/quoridor'), app.state.models)
    sessions = Sessions(evidence)
    router = APIRouter(prefix='/api/agents')

    def call(fn, *args):
        try:
            return fn(*args)
        except HTTPException:
            raise
        except (KeyError, StopIteration, FileNotFoundError, IndexError) as exc:
            raise HTTPException(404, 'Evidence or Agent not found') from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError, ImportError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get('/seasons')
    def seasons(): return call(evidence.seasons)

    @router.get('/seasons/{season}')
    def roster(season: str): return call(evidence.roster, season)

    @router.get('/seasons/{season}/training/{agent}')
    def training(season: str, agent: str): return call(evidence.training, season, agent)

    @router.get('/seasons/{season}/games')
    def games(season: str, agent: str, opponent: str): return call(evidence.games, season, agent, opponent)

    @router.get('/seasons/{season}/replay/{game_id}')
    def replay(season: str, game_id: str): return call(evidence.replay, season, game_id)

    @router.post('/play')
    def create(request: CreateGame): return call(sessions.create, request)

    @router.get('/play/{identifier}')
    def snapshot(identifier: str, x_game_token: str = Header(default='')):
        with sessions.lock: return call(lambda: sessions.snapshot(sessions.get(identifier, x_game_token)))

    @router.post('/play/{identifier}/step')
    def step(identifier: str, request: GameStep, x_game_token: str = Header(default='')):
        return call(sessions.step, identifier, x_game_token, request)

    @router.delete('/play/{identifier}')
    def close(identifier: str, x_game_token: str = Header(default='')):
        with sessions.lock:
            call(sessions.get, identifier, x_game_token)
            del sessions.games[identifier]
        return {'closed':True}

    app.include_router(router)
