"""Platform Job entrypoints, executed inside a Job container.

``python -m agent_system.training.jobs /job/job.json`` runs one registered
entrypoint with the Job's source snapshot as the working directory and writes
everything to the Job run directory (``/job/run``):

* ``events.jsonl`` - started / checkpoint / finished / failed events;
* the entrypoint's own artifacts (for training: ``config.yaml``,
  ``metrics.jsonl``, ``eval_results.jsonl``, ``checkpoints/``), which keep the
  existing training artifact contract so the Training workspace reads them;
* ``manifest.json`` - every produced checkpoint with its SHA-256, metadata and
  output semantics, used by the controller to register immutable copies.

``ENTRYPOINTS`` is plain metadata and imports nothing heavy, so the controller
can validate submissions without torch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ENTRYPOINTS = {
    'train_dqn': dict(
        kind='train', gpu=True, format='yaml', title='DQN / CNN-DQN 训练',
        templates='agent_system/training/configs/dqn', template_pattern=r'[^/]+\.ya?ml',
        description='scripts/train_dqn.py with the submitted YAML. checkpoint_dir and device are set by the Job.'),
    'train_cnn_experiment': dict(
        kind='train', gpu=True, format='json', title='CNN 研究实验（蒸馏 / PPO / DAgger / DDQN / AlphaZero-lite）',
        templates='agent_system/training/configs', template_pattern=r'cnn_[^/]+/[^/]+\.json',
        description='scripts/train_cnn_repair.py with the submitted experiment JSON, including per-checkpoint '
                    'diagnosis. output, checkpoint_dir and device are set by the Job; input paths must be under the data root.'),
    'run_arena_matrix': dict(
        kind='evaluate', gpu=False, format='json', title='门槛评测（Arena 评测批次）',
        template={'candidates': [], 'names': {}, 'gate': 'gate-v1', 'workers': 4},
        description='Registered checkpoints (full SHA) against a versioned gate protocol with scripts/run_benchmark_matrix.py. '
                    'A complete result is published as a new ladder season named by the Job id.'),
    'diagnose_checkpoint': dict(
        kind='evaluate', gpu=False, format='json', title='Checkpoint 诊断（开发对局 + 战术题）',
        template={'checkpoints': [], 'book': '/home/jihua/quoridor-data/arena/cnn-architecture-20261021-213018/manifest.json',
                  'max_plies': 300},
        description='agent_system.evaluation.cnn_diagnostics.diagnose_checkpoint on registered checkpoints (full SHA): '
                    'paired development games on a locked book plus the tactical probes. Development results are not holdout.'),
    'evaluate_checkpoints': dict(
        kind='evaluate', gpu=False, format='json', title='Checkpoint 评测',
        template={'checkpoints': [], 'opponents': [{'name': 'greedy', 'type': 'greedy'},
                                                   {'name': 'minimax_d1', 'type': 'minimax', 'depth': 1}],
                  'games_per_opponent': 20, 'max_steps': 3000, 'seed': 42},
        description='Registered checkpoints (full SHA) against fixed opponents; DQN checkpoint agents, argmax.'),
}
CHECKPOINT_METADATA = ('checkpoint_id', 'agent_id', 'episode_count', 'training_step', 'observation_version',
                       'action_space_version', 'action_count', 'created_at', 'model_config')


class Events:
    def __init__(self, run: Path):
        self.path = run / 'events.jsonl'

    def emit(self, event: str, **fields):
        with self.path.open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), event=event, **fields)) + '\n')


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def infer_semantics(metadata: dict) -> str | None:
    algorithm = str((metadata.get('model_config') or {}).get('algorithm') or '')
    if not algorithm:
        return None
    return 'q_values' if 'dqn' in algorithm else 'policy_logits'


def write_manifest(run: Path, semantics: str | None, directory: Path | None = None, known: dict | None = None) -> list[dict]:
    """List checkpoints; semantics come from the caller, a diagnosis of the same SHA, or the recorded algorithm."""
    import torch
    rows = []
    for path in sorted((directory or run / 'checkpoints').glob('*.pt')):
        raw = torch.load(path, map_location='cpu', weights_only=True)
        metadata = {k: raw.get(k) for k in CHECKPOINT_METADATA}
        digest = sha256(path)
        rows.append(dict(file=str(path.relative_to(run)), sha256=digest, metadata=metadata,
                         output_semantics=semantics or (known or {}).get(digest) or infer_semantics(metadata)))
    (run / 'manifest.json').write_text(json.dumps(dict(checkpoints=rows), indent=2))
    return rows


def train_dqn(job: dict, run: Path, events: Events):
    import yaml
    config = yaml.safe_load(job['config'])
    if not isinstance(config, dict):
        raise ValueError('train_dqn config must be a YAML mapping')
    device = 'cuda' if job['resources'].get('gpu') else 'cpu'
    config.update(checkpoint_dir=str(run / 'checkpoints'), device=device)
    config.setdefault('agent_id', job['id'])
    path = run / 'job-config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    events.emit('process', command='scripts/train_dqn.py', device=device)
    subprocess.run([sys.executable, '-u', 'scripts/train_dqn.py', '--config', str(path),
                    '--checkpoint-dir', config['checkpoint_dir'], '--device', device], check=True)
    algorithm = str(config.get('algorithm', 'dqn'))
    return write_manifest(run, 'q_values' if 'dqn' in algorithm else None)


def train_cnn_experiment(job: dict, run: Path, events: Events):
    experiment = json.loads(job['config']) if isinstance(job['config'], str) else dict(job['config'])
    if not isinstance(experiment.get('training'), dict):
        raise ValueError('CNN experiment config needs a training object')
    output = run / 'experiment'
    experiment['output'] = str(output)
    experiment['training'].update(checkpoint_dir=str(output / 'checkpoints'), device='cuda')
    experiment['training'].setdefault('agent_id', job['id'])
    path = run / 'job-experiment.json'
    path.write_text(json.dumps(experiment, indent=2))
    events.emit('process', command='scripts/train_cnn_repair.py', method=next(
        (k for k in ('alphazero', 'research', 'distillation', 'teacher') if k in experiment), 'cnn_dqn'))
    subprocess.run([sys.executable, '-u', 'scripts/train_cnn_repair.py', '--config', str(path)], check=True)
    diagnoses = output / 'diagnoses.json'
    known = {d['sha256']: d.get('output_semantics') for d in json.loads(diagnoses.read_text())} if diagnoses.exists() else {}
    return write_manifest(run, None, output / 'checkpoints', known)


def gate_roster(config: dict, data_root: str) -> tuple[dict, dict]:
    """Candidate AgentSpecs plus the gate opponents, restricted to candidate-versus-gate matchups."""
    gate_id = config.get('gate', 'gate-v1')
    if not isinstance(gate_id, str) or not gate_id.replace('-', '').isalnum():
        raise ValueError('Invalid gate protocol id')
    gate = json.loads(Path(f'agent_system/evaluation/arena/gates/{gate_id}.json').read_text())
    shas = config.get('candidates') or []
    if not shas or any(not isinstance(s, str) or len(s) != 64 for s in shas):
        raise ValueError('candidates must be a non-empty list of full SHA-256 identities')
    names = config.get('names') or {}
    taken = {a['id'] for a in gate['agents']}
    candidates = []
    for sha in dict.fromkeys(shas):
        identifier = names.get(sha) or f'cand-{sha[:12]}'
        if identifier in taken:
            raise ValueError(f'Candidate name collides with another agent: {identifier}')
        taken.add(identifier)
        candidates.append(dict(id=identifier, display_name=identifier, deterministic=True, policy=dict(type='top_k', k=1),
                               algo=dict(type='dqn', params=dict(checkpoint=f'{data_root}/models/{sha}.pt', sha256=sha, device='cpu'))))
    matchups = [[c['id'], g['id']] for c in candidates for g in gate['agents']]
    return dict(agents=candidates + gate['agents'], matchups=matchups, gate=gate_id), gate


def run_arena_matrix(job: dict, run: Path, events: Events):
    config = json.loads(job['config']) if isinstance(job['config'], str) else job['config']
    roster, gate = gate_roster(config, job['data_root'])
    path = run / 'arena-roster.json'
    path.write_text(json.dumps(roster, indent=2))
    workers = max(1, min(int(config.get('workers', 4)), int(job['resources'].get('cpus', 4))))
    events.emit('process', command='scripts/run_benchmark_matrix.py', gate=gate['id'],
                candidates=[a['id'] for a in roster['agents'][:len(roster['agents']) - len(gate['agents'])]])
    subprocess.run([sys.executable, '-u', 'scripts/run_benchmark_matrix.py', '--roster', str(path), '--output', str(run / 'arena'),
                    '--pairs', str(gate['pairs']), '--seed', str(gate['seed']), '--max-plies', str(gate['max_plies']),
                    '--workers', str(workers), '--book', gate['book']], check=True)
    report = json.loads((run / 'arena' / 'report.json').read_text())
    events.emit('arena', status=report.get('status'), ranking=[(r.get('id'), r.get('rank'), r.get('elo')) for r in report.get('ranking', [])])
    return []


def diagnose_checkpoints(job: dict, run: Path, events: Events):
    from agent_system.evaluation.cnn_diagnostics import diagnose_checkpoint
    config = json.loads(job['config']) if isinstance(job['config'], str) else job['config']
    shas = config.get('checkpoints') or []
    if not shas or any(not isinstance(s, str) or len(s) != 64 for s in shas):
        raise ValueError('checkpoints must be a non-empty list of full SHA-256 identities')
    book_path = Path(config.get('book') or ENTRYPOINTS['diagnose_checkpoint']['template']['book'])
    if not book_path.resolve().is_relative_to(Path(job['data_root'])):
        raise ValueError('The book must be under the data root')
    loaded = json.loads(book_path.read_text())
    book = loaded if isinstance(loaded, list) else loaded['book']
    rows = []
    for sha in dict.fromkeys(shas):
        path = Path(job['data_root']) / 'models' / f'{sha}.pt'
        if not path.is_file() or sha256(path) != sha:
            raise ValueError(f'Checkpoint {sha} is not a verified registered model')
        rows.append(diagnose_checkpoint(path, f'diag-{sha[:12]}', book, run / 'diagnostics' / sha[:12],
                                        max_plies=int(config.get('max_plies', 300))))
        (run / 'diagnoses.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        events.emit('diagnosis', sha256=sha, wins=rows[-1]['counts']['wins'], games=rows[-1]['games'])
    return []


def load_train_script(path='scripts/train_dqn.py'):
    """Import scripts/train_dqn.py as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location('train_dqn_script', path)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(script)
    return script


def evaluate_checkpoints(job: dict, run: Path, events: Events):
    config = json.loads(job['config']) if isinstance(job['config'], str) else job['config']
    shas = config.get('checkpoints') or []
    if not shas or any(not isinstance(s, str) or len(s) != 64 for s in shas):
        raise ValueError('checkpoints must be a non-empty list of full SHA-256 identities')
    script = load_train_script()
    from agent_system.training.dqn.observation import encode_observation
    from agent_system.training.dqn.observation_cnn import encode_observation_cnn
    import torch
    models = Path(job['data_root']) / 'models'
    cfg = script.TrainConfig(eval_games=int(config.get('games_per_opponent', 20)),
                             eval_max_steps=int(config.get('max_steps', 3000)), seed=int(config.get('seed', 42)))
    rows = []
    with (run / 'eval_results.jsonl').open('a') as out:
        for sha in shas:
            path = models / f'{sha}.pt'
            if not path.is_file() or sha256(path) != sha:
                raise ValueError(f'Checkpoint {sha} is not a verified registered model')
            version = torch.load(path, map_location='cpu', weights_only=True).get('observation_version')
            if version == 'dqn_obs_cnn_v1':
                encoder = encode_observation_cnn
            elif version == 'dqn_obs_v2':
                from agent_system.training.dqn.observation_v2 import encode_observation_v2 as encoder
            else:
                encoder = encode_observation
            for opponent in config.get('opponents', []):
                result = script.run_evaluation(str(path), cfg, encoder=encoder, eval_opp_cfg=opponent)
                row = dict(checkpoint_sha=sha, opponent=opponent.get('name') or opponent.get('type'),
                           opponent_depth=opponent.get('depth'), games=result['num_games'], wins=result['wins'],
                           draws=result['draws'], losses=result['losses'], win_rate=result['win_rate'],
                           avg_game_length=result['avg_game_length'],
                           illegal_action_count=result['illegal_action_count'], seed=cfg.seed)
                out.write(json.dumps(row) + '\n'); out.flush()
                rows.append(row)
                events.emit('evaluation', **row)
    (run / 'evaluation.json').write_text(json.dumps(dict(config=config, results=rows), indent=2))
    return []


RUNNERS = {'train_dqn': train_dqn, 'train_cnn_experiment': train_cnn_experiment, 'run_arena_matrix': run_arena_matrix,
           'diagnose_checkpoint': diagnose_checkpoints, 'evaluate_checkpoints': evaluate_checkpoints}


def main(path: str) -> int:
    job = json.loads(Path(path).read_text())
    run = Path(job['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    events = Events(run)
    events.emit('started', entrypoint=job['entrypoint'], source_commit=job['source']['commit'])
    try:
        checkpoints = RUNNERS[job['entrypoint']](job, run, events)
    except Exception as exc:
        events.emit('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    events.emit('finished', checkpoints=[c['sha256'] for c in checkpoints])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
