"""Read-only index of external experiment directories under ``<data root>/experiments``.

Models join an experiment only through exact catalog provenance
(``training_experiment`` or ``source_path`` inside the experiment directory),
never through names. Reports and assessments are returned as recorded.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

SLUG = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z')
REPORT_LIMIT = 256 * 1024
ASSESSMENT_LIMIT = 1024 * 1024


class Experiments:
    def __init__(self, data_root):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root / 'experiments'

    def directory(self, slug):
        if not isinstance(slug, str) or not SLUG.fullmatch(slug):
            raise ValueError('Invalid experiment identity')
        path = (self.root / slug).resolve()
        if path.parent != self.root.resolve() or not path.is_dir():
            raise KeyError(slug)
        return path

    def _reports(self, path):
        candidates = sorted(path.glob('*.md')) + sorted(path.glob('documentation/*.md'))
        return [p for p in candidates if p.is_file() and p.resolve().is_relative_to(path)]

    def _summary(self, path):
        entries = list(path.iterdir())
        # A training arm is any run directory (up to three levels deep) holding checkpoints/.
        arms = sorted(str(c.parent.relative_to(path)) for pattern in
                      ('*/checkpoints', '*/*/checkpoints', '*/*/*/checkpoints')
                      for c in path.glob(pattern) if c.is_dir())
        modified = max((p.stat().st_mtime for p in entries), default=path.stat().st_mtime)
        return dict(id=path.name, modified=modified, arms=arms,
                    reports=[str(p.relative_to(path)) for p in self._reports(path)],
                    has_assessment=(path / 'assessment.json').is_file(),
                    catalogs=sorted(p.name for p in path.glob('*catalog*.json')))

    def listing(self):
        rows, warnings = [], []
        if not self.root.is_dir():
            return dict(experiments=rows, warnings=['Experiment directory missing'])
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not SLUG.fullmatch(path.name):
                continue
            try:
                rows.append(self._summary(path))
            except OSError as exc:
                warnings.append(f'{path.name}: {exc}')
        rows.sort(key=lambda r: r['modified'], reverse=True)
        return dict(experiments=rows, warnings=warnings)

    def _arm(self, path, record):
        """Experiment-relative arm of a catalog record, or None when it belongs elsewhere."""
        for key in ('training_experiment', 'source_path'):
            raw = record.get(key)
            if not isinstance(raw, str) or not raw:
                continue
            for prefix in ('/home/jihua/quoridor-data', '/quoridor-data', '/data/quoridor'):
                if raw == prefix or raw.startswith(prefix + '/'):
                    raw = str(self.data_root) + raw[len(prefix):]
                    break
            candidate = Path(raw)
            if candidate.is_absolute() and candidate.is_relative_to(path):
                parts = list(candidate.relative_to(path).parts)
                if key == 'source_path':
                    # A checkpoint file: its arm is the run directory holding checkpoints/.
                    parts = parts[:-1]
                    if parts and parts[-1] == 'checkpoints':
                        parts = parts[:-1]
                return '/'.join(parts)
        return None

    def detail(self, slug, models=()):
        path = self.directory(slug)
        result = self._summary(path)
        reports, warnings = [], []
        for report in self._reports(path):
            size = report.stat().st_size
            text = report.read_bytes()[:REPORT_LIMIT].decode('utf-8', 'replace')
            reports.append(dict(path=str(report.relative_to(path)), size=size,
                                truncated=size > REPORT_LIMIT, text=text))
        assessment = None
        file = path / 'assessment.json'
        if file.is_file():
            if file.stat().st_size > ASSESSMENT_LIMIT:
                warnings.append('assessment.json exceeds display limit')
            else:
                try:
                    assessment = json.loads(file.read_text())
                except ValueError as exc:
                    warnings.append(f'assessment.json: {exc}')
        files = [dict(name=p.name, directory=p.is_dir(), size=None if p.is_dir() else p.stat().st_size)
                 for p in sorted(path.iterdir())]
        linked = []
        for model in models:
            arms = sorted({arm for e in model.get('evidence', [])
                           if isinstance(e.get('record'), dict)
                           for arm in [self._arm(path, e['record'])] if arm is not None})
            if arms:
                linked.append(dict(sha256=model['sha256'], aliases=model.get('aliases', []),
                    arms=arms, architecture=model.get('architecture'),
                    output_semantics=model.get('output_semantics'), ranked=model.get('ranked', False),
                    seasons=model.get('seasons', [])))
        return dict(result, reports=reports, assessment=assessment, files=files,
                    models=linked, warnings=warnings)
