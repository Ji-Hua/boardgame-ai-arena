"""Verification ledger of the immutable model store."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from apps.jobs.model_store import ModelStore


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'models').mkdir()
        self.store = ModelStore(self.root, clock=lambda: 5.0)

    def add(self, data, name=None):
        path = self.root / 'models' / f'{name or hashlib.sha256(data).hexdigest()}.pt'
        path.write_bytes(data)
        return path

    def test_ledger_hashes_only_new_or_changed_files_and_reports_mismatches(self):
        good = self.add(b'one')
        self.add(b'two', name='f' * 64)
        self.assertEqual(self.store.verify(), (1, 2, ['f' * 64 + '.pt']))
        ledger = json.loads((self.root / 'models' / 'verified.json').read_text())
        self.assertEqual(ledger['models'][good.stem]['size'], 3)
        self.assertEqual(self.store.verify()[:2], (1, 1))  # only the mismatched file is hashed again
        os.utime(good, ns=(1, 1))
        self.assertEqual(self.store.verify()[:2], (1, 2))


if __name__ == '__main__':
    unittest.main()
