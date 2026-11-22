"""Versioned probe suites: content, engine-checked expectations, publication and evaluation."""
import json
from pathlib import Path
import tempfile
import unittest

from agent_system.analysis.state import ENGINE, Position, build_state
from agent_system.evaluation import probes


class ProbeTests(unittest.TestCase):
    def test_immediate_win_v1_is_the_historical_diagnosis_set(self):
        suite = probes.build('immediate-win', 'v1')
        old = [Position(pawns=((x, 7), (4, 3)) if seat == 1 else ((4, 5), (x, 1)), turn=seat).model_dump(mode='json')
               for seat in (1, 2) for x in range(9)]
        self.assertEqual([p['position'] for p in suite['positions']], old)
        self.assertEqual(suite['sha256'], probes.build('immediate-win', 'v1')['sha256'])

    def test_must_block_positions_are_engine_checked_threats_with_a_block(self):
        suite = probes.build('must-block', 'v1')
        self.assertGreaterEqual(len(suite['positions']), 12)
        for row in suite['positions']:
            position = Position.model_validate(row['position'])
            state = build_state(position)
            outcomes = [probes._opponent_wins_next(ENGINE.apply_action(state, a), position.turn) for a in ENGINE.legal_actions(state)]
            self.assertTrue(any(outcomes) and not all(outcomes), row['id'])

    def test_evaluation_checks_the_expectation(self):
        def winner(state):
            return probes._winning_moves(state, 1 if str(state.current_player).endswith('1') else 2)[0]

        def blocker(state):
            seat = 1 if str(state.current_player).endswith('1') else 2
            return next(a for a in ENGINE.legal_actions(state) if not probes._opponent_wins_next(ENGINE.apply_action(state, a), seat))

        def first_pawn(state):
            return next(a for a in ENGINE.legal_actions(state) if a.kind == 'MovePawn')
        immediate, block = probes.build('immediate-win', 'v1'), probes.build('must-block', 'v1')
        self.assertEqual(probes.evaluate(immediate, winner)['passed'], 18)
        result = probes.evaluate(block, blocker)
        self.assertEqual((result['passed'], result['sha256']), (result['count'], block['sha256']))
        self.assertLess(probes.evaluate(block, first_pawn)['passed'], result['count'])

    def test_publication_is_write_once_and_load_checks_the_hash(self):
        with tempfile.TemporaryDirectory() as root:
            target = probes.publish(root, 'must-block', 'v1')
            self.assertEqual(probes.publish(root, 'must-block', 'v1'), target)
            self.assertEqual(probes.load(root, 'must-block', 'v1')['sha256'], probes.build('must-block', 'v1')['sha256'])
            data = json.loads(target.read_text())
            data['positions'] = data['positions'][1:]
            target.chmod(0o644)
            target.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                probes.load(root, 'must-block', 'v1')
            with self.assertRaises(ValueError):
                probes.publish(root, 'must-block', 'v1')


if __name__ == '__main__':
    unittest.main()
