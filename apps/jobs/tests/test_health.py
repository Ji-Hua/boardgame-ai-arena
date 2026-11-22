"""Training health alert rules."""
import unittest

from apps.jobs.health import assess


def codes(*args, **kw):
    return [a['code'] for a in assess(*args, **kw)]


class HealthTests(unittest.TestCase):
    def test_healthy_run_has_no_alerts(self):
        rows = [dict(type='train', episode=i, loss=0.01, truncation_rate=0.0, repeat_rate=0.02, avg_q_max=0.4) for i in range(10)]
        self.assertEqual(assess(rows, running=True, last_write=100, now=200), [])

    def test_behaviour_and_value_rules_read_the_latest_row(self):
        rows = [dict(episode=10, truncation_rate=0.5), dict(episode=20, truncation_rate=0.4, repeat_rate=0.3, avg_q_max=7.5)]
        alerts = assess(rows)
        self.assertEqual([a['code'] for a in alerts], ['truncation', 'repetition', 'q_divergence'])
        self.assertEqual({a['episode'] for a in alerts}, {20})
        self.assertIn('40%', alerts[0]['message'])

    def test_non_finite_and_loss_spike(self):
        base = [dict(episode=i, loss=0.01) for i in range(6)]
        self.assertEqual(codes(base + [dict(episode=7, loss=float('nan'), grad_norm=float('inf'))]), ['non_finite'])
        self.assertEqual(codes(base + [dict(episode=7, loss=0.5)]), ['loss_spike'])
        self.assertEqual(codes(base[:3] + [dict(episode=7, loss=0.5)]), [])  # too little history

    def test_eval_regression_is_per_opponent(self):
        evals = [dict(opponent_name='greedy', episode=100, win_rate=0.8), dict(opponent_name='minimax', episode=100, win_rate=0.1),
                 dict(opponent_name='greedy', episode=200, win_rate=0.55), dict(opponent_name='minimax', episode=200, win_rate=0.0)]
        alerts = assess([], evals)
        self.assertEqual([a['code'] for a in alerts], ['eval_regression'])
        self.assertIn('greedy', alerts[0]['message'])

    def test_stalled_only_while_running_and_eval_rows_are_ignored_in_metrics(self):
        self.assertEqual(codes([], running=True, last_write=0, now=601), ['stalled'])
        self.assertEqual(codes([], running=False, last_write=0, now=10_000), [])
        self.assertEqual(codes([dict(type='eval', truncation_rate=1.0)]), [])


if __name__ == '__main__':
    unittest.main()
