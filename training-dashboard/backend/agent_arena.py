"""Manifest-scoped research evidence. Games are owned by the gameplay service (apps/gameplay)."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from fastapi import APIRouter, HTTPException
from .replay import arena_replay, _stable_hash
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
                    agents=len(manifest['roster']['agents']), gate=manifest['roster'].get('gate')))
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

    FAILURE_KINDS = ('missed_win', 'loss', 'cycle', 'ply_limit')

    def failures(self, season, agent):
        """Recorded games of one Agent that it lost, drew by repetition or ply limit, or where it missed a win.

        Classification uses only recorded outcomes and the per-game decision diagnostics; nothing is inferred.
        """
        base, _, manifest = self.season(season)
        self.spec(season, agent)
        key = ('failures', manifest['manifest_sha256'], agent)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        rows, totals, games = [], dict.fromkeys(self.FAILURE_KINDS, 0), 0
        for file in sorted((base / 'blocks').glob('*.json.gz')):
            with gzip.open(self.path(file), 'rt') as stream:
                block = json.load(stream)
            if agent not in (block.get('a'), block.get('b')) or block.get('manifest_sha256') != manifest['manifest_sha256']:
                continue
            opponent = block['b'] if block['a'] == agent else block['a']
            for i, game in enumerate(block.get('games', [])):
                if agent not in (game.get('p1'), game.get('p2')):
                    continue
                games += 1
                missed = int(((game.get('decision_diagnostics') or {}).get(agent) or {}).get('missed_immediate_win') or 0)
                kinds = [k for k, hit in (('missed_win', missed > 0), ('loss', bool(game.get('winner')) and game['winner'] != agent),
                                          ('cycle', game.get('termination') == 'deterministic_cycle'),
                                          ('ply_limit', game.get('termination') == 'ply_limit')) if hit]
                for kind in kinds:
                    totals[kind] += 1
                if kinds:
                    rows.append(dict(id=file.name.removesuffix('.json.gz') + ':' + str(i), opponent=opponent,
                        seat=1 if game.get('p1') == agent else 2, kinds=kinds, missed_immediate_win=missed,
                        steps=game.get('steps'), termination=game.get('termination'), winner=game.get('winner'),
                        control=bool(block.get('control')), opening=block.get('opening_id'),
                        available=isinstance(game.get('actions'), list) and bool(game.get('trajectory_hash'))))
        result = dict(agent=agent, games=games, totals=totals, failures=rows)
        with self.lock:
            self.cache[key] = result
        return result

    def missed_wins(self, season, game_id, agent):
        """Plies of one recorded game where the Agent had a legal winning pawn move and played something else."""
        from agent_system.analysis.state import ENGINE, Position, build_state, wire_action
        replay = self.replay(season, game_id)
        if agent not in (replay['p1'], replay['p2']):
            raise ValueError('Agent did not play this game')
        plies = []
        for ply, (position, action) in enumerate(zip(replay['positions'], replay['actions'])):
            if (replay['p1'] if position['turn'] == 1 else replay['p2']) != agent:
                continue
            goal = 8 if position['turn'] == 1 else 0
            state = build_state(Position.model_validate(position))
            wins = [wire_action(a) for a in ENGINE.legal_actions(state) if a.kind == 'MovePawn' and a.target_y == goal]
            if wins and not (action['type'] == 'pawn' and action['target'][1] == goal):
                plies.append(dict(ply=ply, actual=action, winning=wins))
        return dict(game=game_id, agent=agent, plies=plies)

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


def install(app):
    evidence = app.state.evidence = Evidence(os.environ.get('QUORIDOR_DATA_ROOT', '/data/quoridor'), app.state.models)
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

    @router.get('/seasons/{season}/failures/{agent}')
    def failures(season: str, agent: str): return call(evidence.failures, season, agent)

    @router.get('/seasons/{season}/replay/{game_id}/missed-wins')
    def missed_wins(season: str, game_id: str, agent: str): return call(evidence.missed_wins, season, game_id, agent)

    @router.get('/seasons/{season}/replay/{game_id}')
    def replay(season: str, game_id: str): return call(evidence.replay, season, game_id)

    app.include_router(router)
