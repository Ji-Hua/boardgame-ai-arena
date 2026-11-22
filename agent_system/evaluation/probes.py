"""Versioned probe suites: fixed positions with an engine-checked expectation, kept as data.

A suite is published once to ``<data root>/probes/<name>/<version>.json`` together with the SHA-256
of its canonical JSON, and is never rewritten. Diagnoses record the name, version and SHA of every
suite they ran. The generators below exist to publish a suite and to give diagnoses run without a
data root the same positions; ``load`` checks that a published file still hashes to its recorded SHA.

Expectations are decided by the Rust engine only:
  immediate_win  the chosen action ends the game
  block          after the chosen action the opponent has no winning pawn move
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agent_system.analysis.state import ENGINE, Position, build_state, wire_action

GOAL = {1: 8, 2: 0}


def _winning_moves(state, player):
    return [a for a in ENGINE.legal_actions(state)
            if a.kind == 'MovePawn' and a.target_y == GOAL[player]]


def _opponent_wins_next(after, seat):
    return not ENGINE.is_game_over(after) and bool(_winning_moves(after, 3 - seat))


def _immediate_win_v1():
    """The 18 positions the CNN diagnosis has used since 2026-10-20: the mover one step from its goal row."""
    for seat in (1, 2):
        for x in range(9):
            yield f'seat{seat}-x{x}', Position(pawns=((x, 7), (4, 3)) if seat == 1 else ((4, 5), (x, 1)), turn=seat)


def _must_block_v1():
    """The opponent stands one step from its goal on an open board; only a wall can stop it. Kept only when the
    engine confirms the threat, that the mover cannot win first, and that at least one legal block exists."""
    for seat in (1, 2):
        for x in range(9):
            position = Position(pawns=((4, 4), (x, 1)) if seat == 1 else ((x, 7), (4, 4)), turn=seat)
            state = build_state(position)
            legal = ENGINE.legal_actions(state)
            if _winning_moves(state, seat) or not any(_opponent_wins_next(ENGINE.apply_action(state, a), seat) for a in legal):
                continue
            if any(not _opponent_wins_next(ENGINE.apply_action(state, a), seat) for a in legal):
                yield f'seat{seat}-x{x}', position


SUITES = {
    ('immediate-win', 'v1'): ('immediate_win', 'Mover one step from its goal row: does the agent take the win?', _immediate_win_v1),
    ('must-block', 'v1'): ('block', 'Opponent one step from its goal on an open board: does the agent block?', _must_block_v1),
}
DEFAULT = (('immediate-win', 'v1'), ('must-block', 'v1'))


def sha256(suite):
    body = {k: v for k, v in suite.items() if k != 'sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build(name, version):
    expectation, description, generate = SUITES[(name, version)]
    suite = dict(name=name, version=version, expectation=expectation, description=description,
                 positions=[dict(id=i, position=p.model_dump(mode='json')) for i, p in generate()])
    return dict(suite, sha256=sha256(suite))


def path(data_root, name, version):
    return Path(data_root) / 'probes' / name / f'{version}.json'


def load(data_root, name, version):
    suite = json.loads(path(data_root, name, version).read_text())
    if (suite.get('name'), suite.get('version')) != (name, version) or suite.get('sha256') != sha256(suite):
        raise ValueError(f'Probe suite {name}/{version} does not match its recorded SHA-256')
    return suite


def publish(data_root, name, version):
    """Write a suite once; an existing file must be identical."""
    suite, target = build(name, version), path(data_root, name, version)
    if target.exists():
        if load(data_root, name, version)['sha256'] != suite['sha256']:
            raise ValueError(f'{target} exists with different content; publish a new version instead')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.partial')
    partial.write_text(json.dumps(suite, indent=1, sort_keys=True) + '\n')
    partial.chmod(0o444)
    partial.rename(target)
    return target


def evaluate(suite, choose):
    """Run ``choose(state) -> engine action`` on every position and check the suite expectation."""
    items = []
    for row in suite['positions']:
        position = Position.model_validate(row['position'])
        state = build_state(position)
        action = choose(state)
        after = ENGINE.apply_action(state, action)
        ok = ENGINE.is_game_over(after) if suite['expectation'] == 'immediate_win' else not _opponent_wins_next(after, position.turn)
        items.append(dict(id=row['id'], position=row['position'], action=wire_action(action), ok=bool(ok)))
    return dict(name=suite['name'], version=suite['version'], sha256=suite['sha256'],
                passed=sum(i['ok'] for i in items), count=len(items), items=items)


def main():
    parser = argparse.ArgumentParser(description='Publish versioned probe suites under <data root>/probes.')
    parser.add_argument('--data-root', default='/home/jihua/quoridor-data')
    parser.add_argument('suites', nargs='*', help='name/version (default: every built-in suite)')
    args = parser.parse_args()
    for item in args.suites or [f'{n}/{v}' for n, v in SUITES]:
        name, version = item.split('/')
        target = publish(args.data_root, name, version)
        suite = load(args.data_root, name, version)
        print(f"{name}/{version} {len(suite['positions'])} positions sha256={suite['sha256']} {target}")


if __name__ == '__main__':
    main()
