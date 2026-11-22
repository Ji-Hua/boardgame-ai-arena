"""Deployment paths must not reference archived release directories (AC-03/08)."""
import pathlib,subprocess,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
class ReleasePathsTests(unittest.TestCase):
 def test_release_resolves_only_current_locations(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td);(p/'release.env').write_text('COMMIT='+'a'*40+'\n')
   (p/'lib.sh').write_text((ROOT/'deploy/release/lib.sh').read_text())
   (p/'check.sh').write_text('. "'+str(p/'lib.sh')+'"\nprintf "%s\\n" "$DEV_DIR" "$R" "$SRC"\n')
   values=subprocess.check_output(['sh',str(p/'check.sh')],text=True).splitlines()
   self.assertEqual(values[:2],['/home/jihua/quoridor-releases/main-current']*2)
   self.assertEqual(values[2],str(pathlib.Path.home()/'quoridor-runtime/releases'/('a'*40)))
 def test_operation_templates_have_no_retired_runtime_paths(self):
  for p in (ROOT/'deploy').rglob('*'):
   if p.is_file() and p.suffix in ('.sh','.service'):
    text=p.read_text()
    for retired in ('/workspace/boardgame-ai-arena/.venv/', '/quoridor-releases/p5-dev/'):
     self.assertNotIn(retired,text,str(p))
