"""Durable two-seat games between people and exact season Agents.

The Rust engine is the rule authority: only actions are stored, and every
position is recomputed from them. A game has two capabilities: the player
token (moves, close, listing) and the watch token (read only). Only SHA-256
digests of tokens are stored. Agents are rebuilt from the recorded identity
after a restart; a model that no longer loads with the same SHA is an error,
never a substitute.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import hmac
import secrets
import threading
import time

from pydantic import BaseModel, ConfigDict, Field

MAX_PLIES = 500
IDLE_SECONDS = 14 * 24 * 3600
CHECKPOINT_LIMIT = 128 * 1024 * 1024


class CreateGame(BaseModel):
    model_config = ConfigDict(extra='forbid')
    season: str = Field(max_length=120)
    p1: str = Field(max_length=160)
    p2: str = Field(max_length=160)


class GameStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=0)
    action: dict | None = None


class Listing(BaseModel):
    model_config = ConfigDict(extra='forbid')
    listed: bool


class Denied(Exception):
    """Unknown game or a token without the needed capability."""


class Conflict(Exception):
    """Stale revision, finished or closed game."""


class Busy(Exception):
    """Too many active games."""


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Games:
    def __init__(self, store, evidence, serialize, *, loaded=4, active_limit=64, clock=time.time):
        self.store, self.evidence, self.serialize = store, evidence, serialize
        self.loaded, self.active_limit, self.clock = loaded, active_limit, clock
        self.cache = OrderedDict()  # id -> dict(state, positions, instances)
        self.lock = threading.RLock()

    # Agents -----------------------------------------------------------------
    def identity(self, season, identifier):
        if identifier == 'human':
            return dict(id='human', name='玩家', kind='human')
        row = self.evidence.spec(season, identifier)
        if not row['playable']:
            raise ValueError('Unsupported Agent')
        return {**row, 'actual_sha256': row['sha256'] if row['kind'] == 'neural' else None}

    def agent(self, row):
        if row['kind'] == 'human':
            return None
        from agent_system.runtime.service.yaml_loader import parse_agent_definition, create_agent_from_definition
        spec = copy.deepcopy(row['spec'])
        if row['kind'] == 'neural':
            model = self.evidence.models.detail(row['sha256'], verify_sha=True)
            if not model['available'] or not model['sha_verified'] or model.get('conflicts'):
                raise ValueError('Unavailable or ambiguous exact checkpoint')
            path = self.evidence.path(next(p['path'] for p in model['checkpoints'] if p.get('sha_verified')))
            if path.stat().st_size > CHECKPOINT_LIMIT:
                raise ValueError('Checkpoint exceeds preview size limit')
            spec['algo']['params'].update(checkpoint=str(path), device='cpu')
        if spec['algo']['type'] == 'minimax' and not 1 <= int(spec['algo'].get('params', {}).get('depth', 2)) <= 3:
            raise ValueError('Preview supports Minimax depth 1–3')
        instance = create_agent_from_definition(parse_agent_definition(spec), context={'seed': 42})
        if row['kind'] == 'neural' and instance.inference.sha256 != row['sha256']:
            raise ValueError('Loaded SHA mismatch')
        return instance

    # Positions --------------------------------------------------------------
    def _replay(self, actions):
        from agent_system.analysis.state import ENGINE, Move, to_position
        state, positions = ENGINE.initial_state(), []
        for action in actions:
            positions.append(to_position(state))
            state = ENGINE.apply_action(state, Move.model_validate(action).engine_action())
        return state, positions

    def _live(self, game, agents=False):
        entry = self.cache.get(game['id'])
        if entry is None or len(entry['positions']) != game['revision']:
            state, positions = self._replay(game['actions'])
            entry = dict(state=state, positions=positions, instances=None)
        if agents and entry['instances'] is None:
            entry['instances'] = [self.agent(row) for row in game['identities']]
        self.cache[game['id']] = entry
        self.cache.move_to_end(game['id'])
        while len(self.cache) > self.loaded:
            self.cache.popitem(last=False)
        return entry

    def snapshot(self, game, role):
        from agent_system.analysis.state import snapshot
        entry = self._live(game)
        view = {**snapshot(entry['state']), 'id': game['id'], 'revision': game['revision'],
                'identities': game['identities'], 'actions': game['actions'], 'positions': entry['positions'],
                'limit': game['revision'] >= MAX_PLIES, 'status': game['status'], 'listed': game['listed'],
                'season': game['season'], 'created': game['created'], 'updated': game['updated'], 'role': role}
        if role != 'player':
            view['legal_actions'] = []
        return view

    # Capabilities -----------------------------------------------------------
    def _game(self, identifier, token, need='watch'):
        game = self.store.get(identifier)
        if game is None or game['status'] == 'closed':
            raise Denied('对局不存在、已关闭或无访问凭据')
        hashed = digest(token or '')
        if hmac.compare_digest(game['play_hash'], hashed):
            return game, 'player'
        if need == 'watch' and hmac.compare_digest(game['watch_hash'], hashed):
            return game, 'watcher'
        if need == 'watch' and not token and game['listed'] and game['status'] == 'finished':
            return game, 'public'
        raise Denied('对局不存在、已关闭或无访问凭据')

    # Operations -------------------------------------------------------------
    def create(self, request):
        with self.lock:
            now = self.clock()
            self.store.abandon(now - IDLE_SECONDS, now)
            if self.store.count('active') >= self.active_limit:
                raise Busy('进行中的对局已满，请先关闭旧对局')
            identities = [self.identity(request.season, x) for x in (request.p1, request.p2)]
            instances = [self.agent(row) for row in identities]
            token, watch = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
            game = dict(id=secrets.token_hex(16), play_hash=digest(token), watch_hash=digest(watch),
                        season=request.season, identities=identities, created=now)
            self.store.insert(game)
            game = self.store.get(game['id'])
            self._live(game)['instances'] = instances
            return dict(token=token, watch_token=watch, **self.snapshot(game, 'player'))

    def view(self, identifier, token):
        with self.lock:
            game, role = self._game(identifier, token)
            return self.snapshot(game, role)

    def step(self, identifier, token, request):
        from agent_system.analysis.state import ENGINE, Move, wire_action, to_position
        with self.lock:
            game, _ = self._game(identifier, token, need='play')
            if request.revision != game['revision']:
                raise Conflict('对局已更新，请刷新')
            if game['status'] != 'active':
                raise Conflict('对局已结束')
            entry = self._live(game)
            state = entry['state']
            position = to_position(state)
            legal = [wire_action(a) for a in ENGINE.legal_actions(state)]
            seat = position['turn'] - 1
            if game['identities'][seat]['kind'] == 'human':
                if request.action is None:
                    raise ValueError('请选择玩家动作')
                action = Move.model_validate(request.action).model_dump(mode='json')
            else:
                if request.action is not None:
                    raise ValueError('Agent 行动不能由客户端指定')
                instance = self._live(game, agents=True)['instances'][seat]
                action = instance.make_action(self.serialize(state), legal)
            if action not in legal:
                raise ValueError('Engine rejected illegal action')
            new_state = ENGINE.apply_action(state, Move.model_validate(action).engine_action())
            winner = self._seat(ENGINE.winner(new_state))
            finished = winner is not None or game['revision'] + 1 >= MAX_PLIES
            actions = game['actions'] + [action]
            if not self.store.append(identifier, game['revision'], actions, 'finished' if finished else 'active',
                                     winner, self.clock()):
                raise Conflict('对局已更新，请刷新')
            entry.update(state=new_state, positions=entry['positions'] + [position])
            return self.snapshot(self.store.get(identifier), 'player')

    @staticmethod
    def _seat(winner):
        from quoridor_engine import Player
        return None if winner is None else (1 if winner == Player.P1 else 2)

    def close(self, identifier, token):
        with self.lock:
            self._game(identifier, token, need='play')
            self.store.update(identifier, self.clock(), status='closed', listed=0)
            self.cache.pop(identifier, None)
            return dict(closed=True)

    def set_listed(self, identifier, token, listed):
        with self.lock:
            game, _ = self._game(identifier, token, need='play')
            if listed and game['status'] != 'finished':
                raise Conflict('只有已结束的对局可以加入对局列表')
            self.store.update(identifier, self.clock(), listed=int(listed))
            return self.snapshot(self.store.get(identifier), 'player')

    def listing(self):
        rows = self.store.listing()
        return dict(games=[dict(id=g['id'], season=g['season'], winner=g['winner'], plies=g['revision'],
                                updated=g['updated'], players=[dict(id=i['id'], name=i.get('name'), kind=i['kind'],
                                                                    actual_sha256=i.get('actual_sha256'))
                                                               for i in g['identities']]) for g in rows])
