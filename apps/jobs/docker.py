"""Docker CLI and host tooling used by the Job controller (kept small so tests can replace it)."""
from __future__ import annotations

import json
import subprocess


def _run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=kwargs.pop('timeout', 120), **kwargs)


class Docker:
    def run(self, args):
        result = _run(['docker', 'run', '-d', *args])
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'docker run failed')
        return result.stdout.strip()

    def state(self, name):
        """Container state dict, or None when the container no longer exists."""
        result = _run(['docker', 'inspect', '--format', '{{json .State}}', name])
        return json.loads(result.stdout) if result.returncode == 0 else None

    def logs(self, name, tail=None):
        args = ['docker', 'logs'] + (['--tail', str(tail)] if tail else []) + [name]
        result = _run(args, timeout=60)
        return result.stdout + result.stderr

    def stop(self, name, grace=30):
        _run(['docker', 'stop', '-t', str(grace), name], timeout=grace + 30)

    def remove(self, name):
        _run(['docker', 'rm', '-f', name])

    def image_id(self, image):
        result = _run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image])
        return result.stdout.strip() if result.returncode == 0 else None


def gpu_status():
    try:
        result = _run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                       '--format=csv,noheader,nounits'], timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(available=False, reason=str(exc))
    if result.returncode:
        return dict(available=False, reason=result.stderr.strip())
    rows = []
    for line in result.stdout.strip().splitlines():
        name, total, used, utilization = [x.strip() for x in line.split(',')]
        rows.append(dict(name=name, memory_total_mib=int(total), memory_used_mib=int(used), utilization=int(utilization)))
    return dict(available=True, gpus=rows)


class Git:
    def __init__(self, repo):
        self.repo = str(repo)

    def resolve(self, ref):
        result = _run(['git', '-C', self.repo, 'rev-parse', '--verify', f'{ref}^{{commit}}'])
        if result.returncode:
            raise ValueError(f'Unknown source commit or ref: {ref}')
        return result.stdout.strip()

    def export(self, commit, target):
        """Write the committed tree (without tracked historical artifacts) into target."""
        archive = subprocess.Popen(['git', '-C', self.repo, 'archive', '--format=tar', commit, '--', '.',
                                    ':(exclude)agent_system/training/artifacts'], stdout=subprocess.PIPE)
        extract = _run(['tar', '-x', '-C', str(target)], stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() or extract.returncode:
            raise RuntimeError(f'Source export failed for {commit}: {extract.stderr.strip()}')

    def files(self, commit, directory):
        result = _run(['git', '-C', self.repo, 'ls-tree', '--name-only', f'{commit}:{directory}'])
        return sorted(result.stdout.split()) if result.returncode == 0 else []

    def show(self, commit, path):
        result = _run(['git', '-C', self.repo, 'show', f'{commit}:{path}'])
        if result.returncode:
            raise KeyError(path)
        return result.stdout
