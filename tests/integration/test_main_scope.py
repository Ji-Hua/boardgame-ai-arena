"""Intent AC-01/04: the supported registry is Quoridor only."""
import unittest
from agent_system.games.registry import DEFINITIONS, validate_ref

class MainScopeTests(unittest.TestCase):
    def test_only_quoridor_is_advertised(self):
        self.assertEqual([row['ref']['game_id'] for row in DEFINITIONS], ['quoridor'])

    def test_unsupported_game_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            validate_ref({'game_id':'mr_jack','ruleset':'london-2016-base','version':1})

    def test_unsupported_history_is_preserved_but_not_advertised(self):
        import tempfile, json
        from pathlib import Path
        from apps.jobs.projects import Projects
        with tempfile.TemporaryDirectory() as root:
            projects = Projects(Path(root)/"projects.sqlite")
            projects.db.execute("insert into projects values(?,?,?,?)", ("old", "Historical", "Preserve", json.dumps(dict(game_id="retired",ruleset="old",version=1))))
            projects.publish()
            self.assertEqual([p["id"] for p in projects.listing()], ["quoridor-legacy"])
            self.assertEqual(projects.db.execute("select count(*) from projects").fetchone()[0],2)
