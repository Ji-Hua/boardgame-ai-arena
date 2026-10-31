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
        templates='agent_system/training/configs/dqn',
        description='scripts/train_dqn.py with the submitted YAML. checkpoint_dir and device are set by the Job.'),
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


def write_manifest(run: Path, semantics: str | None) -> list[dict]:
    import torch
    rows = []
    for path in sorted((run / 'checkpoints').glob('*.pt')):
        raw = torch.load(path, map_location='cpu', weights_only=True)
        rows.append(dict(file=str(path.relative_to(run)), sha256=sha256(path), output_semantics=semantics,
                         metadata={k: raw.get(k) for k in CHECKPOINT_METADATA}))
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


def evaluate_checkpoints(job: dict, run: Path, events: Events):
    config = json.loads(job['config']) if isinstance(job['config'], str) else job['config']
    shas = config.get('checkpoints') or []
    if not shas or any(not isinstance(s, str) or len(s) != 64 for s in shas):
        raise ValueError('checkpoints must be a non-empty list of full SHA-256 identities')
    spec = importlib.util.spec_from_file_location('train_dqn_script', 'scripts/train_dqn.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
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


RUNNERS = {'train_dqn': train_dqn, 'evaluate_checkpoints': evaluate_checkpoints}


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
