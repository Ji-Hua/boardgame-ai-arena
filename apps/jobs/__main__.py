"""Run the Job controller: ``python -m apps.jobs`` from a repository snapshot on the Training host.

Environment: JOBS_DATA_ROOT (default /home/jihua/quoridor-data), JOBS_REPO (git repository
holding the submitted commits), JOBS_IMAGE (GPU training image), JOBS_SOURCE_REF (default
ref for new Jobs), JOBS_TOKEN_FILE (shared write token), JOBS_HOST/JOBS_PORT, JOBS_MAX_CPU_JOBS.
"""
from __future__ import annotations

import logging
import os
import threading

import uvicorn

from .app import create_app
from .controller import Controller
from .docker import Docker, Git


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    env = os.environ.get
    controller = Controller(data_root=env('JOBS_DATA_ROOT', '/home/jihua/quoridor-data'),
                            git=Git(env('JOBS_REPO', '/home/jihua/workspace/boardgame-ai-arena')), docker=Docker(),
                            image=env('JOBS_IMAGE', 'quoridor-train-gpu:latest'),
                            source_ref=env('JOBS_SOURCE_REF', 'feature/unified-rl-platform'),
                            uid_gid=f'{os.getuid()}:{os.getgid()}', max_cpu_jobs=int(env('JOBS_MAX_CPU_JOBS', '2')))
    stop = threading.Event()
    threading.Thread(target=controller.loop, args=(stop,), daemon=True, name='scheduler').start()
    try:
        uvicorn.run(create_app(controller, env('JOBS_TOKEN_FILE')), host=env('JOBS_HOST', '127.0.0.1'),
                    port=int(env('JOBS_PORT', '18796')), log_level='info')
    finally:
        stop.set()


if __name__ == '__main__':
    main()
