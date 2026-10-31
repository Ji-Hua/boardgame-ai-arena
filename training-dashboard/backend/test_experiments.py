"""Experiment index: safe identities, recorded reports and exact provenance joins."""
import json
from pathlib import Path
import tempfile
import unittest
from backend.experiments import Experiments, REPORT_LIMIT


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'experiments' / 'cnn-test-20261031'
        (self.base / '01-arm' / 'checkpoints').mkdir(parents=True)
        (self.base / 'holdout').mkdir()
        (self.base / 'documentation').mkdir()
        (self.base / 'benchmark-report.md').write_text('# Result\nA wins.')
        (self.base / 'documentation' / 'notes.md').write_text('notes')
        (self.base / 'assessment.json').write_text(json.dumps({'winner': 'A'}))
        (self.root / 'experiments' / 'other').mkdir()
        self.index = Experiments(self.root)

    def model(self, sha, **record):
        return dict(sha256=sha, aliases=[sha[:4]], evidence=[dict(record=record)], ranked=False, seasons=[])

    def test_listing_summarizes_arms_reports_and_assessment(self):
        rows = {r['id']: r for r in self.index.listing()['experiments']}
        self.assertEqual(set(rows), {'cnn-test-20261031', 'other'})
        row = rows['cnn-test-20261031']
        self.assertEqual(row['arms'], ['01-arm'])
        self.assertEqual(row['reports'], ['benchmark-report.md', 'documentation/notes.md'])
        self.assertTrue(row['has_assessment'])
        self.assertFalse(rows['other']['has_assessment'])

    def test_detail_joins_models_only_by_exact_experiment_path(self):
        inside = str(self.base / '01-arm')
        models = [self.model('a'*64, training_experiment=inside),
                  self.model('b'*64, source_path='/home/jihua/quoridor-data/experiments/cnn-test-20261031/01-arm/checkpoints/ep1.pt'),
                  self.model('c'*64, training_experiment=str(self.root / 'experiments' / 'cnn-test-20261031-copy' / '01-arm')),
                  self.model('d'*64, id='cnn-test-20261031')]
        detail = self.index.detail('cnn-test-20261031', models)
        linked = {m['sha256']: m['arms'] for m in detail['models']}
        self.assertEqual(linked['a'*64], ['01-arm'])
        self.assertEqual(linked['b'*64], ['01-arm'])
        self.assertNotIn('c'*64, linked)
        self.assertNotIn('d'*64, linked)
        job = self.model('e'*64, experiment='cnn-test-20261031', training_job='job-1', source_path='/elsewhere/x.pt')
        other = self.model('f'*64, experiment='other', training_job='job-2')
        linked = {m['sha256']: m['arms'] for m in self.index.detail('cnn-test-20261031', [job, other])['models']}
        self.assertEqual(linked, {'e'*64: ['job:job-1']})
        self.assertEqual(detail['assessment'], {'winner': 'A'})
        self.assertEqual(detail['reports'][0]['text'], '# Result\nA wins.')
        self.assertIn('holdout', [f['name'] for f in detail['files'] if f['directory']])

    def test_production_prefix_maps_to_data_root(self):
        index = Experiments(self.root)
        record = dict(source_path='/home/jihua/quoridor-data/experiments/cnn-test-20261031/01-arm/checkpoints/ep1.pt')
        self.assertEqual(index._arm(self.base, record), '01-arm')
        nested = dict(training_experiment=str(self.base / 'matrix' / 'A-baseline_seed-42'))
        self.assertEqual(index._arm(self.base, nested), 'matrix/A-baseline_seed-42')

    def test_large_report_is_truncated_and_bad_identity_rejected(self):
        (self.base / 'big.md').write_bytes(b'x' * (REPORT_LIMIT + 10))
        report = next(r for r in self.index.detail('cnn-test-20261031')['reports'] if r['path'] == 'big.md')
        self.assertTrue(report['truncated'])
        self.assertEqual(len(report['text']), REPORT_LIMIT)
        for slug in ('../experiments', '.hidden', 'a/b', ''):
            with self.assertRaises(ValueError):
                self.index.directory(slug)
        with self.assertRaises(KeyError):
            self.index.directory('missing')


if __name__ == '__main__':
    unittest.main()
