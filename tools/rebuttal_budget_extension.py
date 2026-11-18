"""Run the BREC pair and triple encoding stages sequentially."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rebuttal_experiments.common import atomic_json
from rebuttal_experiments.scheduling import add_device_arguments, selected_devices

ROOT = Path(__file__).resolve().parents[1]
STAGES = (('brec', 'brec_p8.json', 'brec'),
          ('triple-encodings', 'triple_encodings_q8_q16.json', 'triple_encodings'))
FULL_STAGES = (('brec', 'brec_full.json', 'brec'),
               ('triple-encodings', 'triple_encodings_full.json', 'triple_encodings'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, help='Override the worker count of both stages')
    parser.add_argument('--memory-gib', type=float,
                        help='Override the total RSS budget for each sequential stage in GiB')
    parser.add_argument('--reserve-gib', type=float, default=8)
    parser.add_argument('--pair-ids', type=int, nargs='+')
    add_device_arguments(parser)
    parser.add_argument('--full', action='store_true', help='Run the union of the default and supplemental grids')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    devices = selected_devices(args, parser)
    device_arguments = ['--devices', *devices] if args.devices else ['--device', args.device]
    if ((args.workers is not None and args.workers < 1)
            or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0
            or args.memory_gib is not None and
            (not math.isfinite(args.memory_gib) or args.memory_gib <= 0)):
        parser.error('Positive workers and memory budget, and finite nonnegative reserve required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.budget-extension.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'补充实验已在运行：{out}；本次未启动新任务。', file=sys.stderr)
            return 0 if args.resume else 2
        active = None
        pending_signal = None
        previous = {}
        state = {'status': 'running', 'supervisor_pid': os.getpid(),
                 'started_unix_seconds': time.time(), 'completed_stages': [],
                 'workers': args.workers, 'memory_gib_override': args.memory_gib,
                 'reserve_gib': args.reserve_gib, 'devices': devices,
                 'scope': 'full' if args.full else 'supplement'}

        def save(**updates):
            state.update(updates, updated_unix_seconds=time.time())
            atomic_json(out/'extension.json', state)

        def forward(signum, frame):
            nonlocal pending_signal
            pending_signal = signum
            if active is not None and active.poll() is None:
                try:
                    os.killpg(active.pid, signum)
                except ProcessLookupError:
                    pass

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, forward)
            stages = FULL_STAGES if args.full else STAGES
            for experiment, config, directory in stages:
                if pending_signal is not None:
                    save(status='interrupted')
                    return 128+pending_signal
                stage_out = out/directory
                scientific = [sys.executable, '-u', '-m', 'rebuttal_experiments', experiment,
                    '--config', str(ROOT/'configs/rebuttal'/config),
                    '--brec-file', str(args.brec_file.resolve()), '--out', str(stage_out),
                    '--bank-cache', str(out/'polynomial_banks'),
                    '--reserve-gib', str(args.reserve_gib), *device_arguments]
                if args.workers is not None:
                    scientific += ['--workers', str(args.workers)]
                if args.memory_gib is not None:
                    scientific += ['--memory-gib', str(args.memory_gib)]
                if args.pair_ids is not None:
                    scientific += ['--pair-ids', *map(str, args.pair_ids)]
                if args.resume:
                    scientific += ['--resume']
                command = [sys.executable, '-u', '-m', 'tools.rebuttal_progress']
                if os.environ.get('REBUTTAL_PROGRESS', '1') == '0':
                    command += ['--no-progress']
                command += ['--', *scientific]
                save(stage=experiment, command=command)
                print(f'BREC 阶段 {experiment} ({config})', flush=True)
                active = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
                save(stage_launcher_pid=active.pid)
                if pending_signal is not None:
                    forward(pending_signal, None)
                code = active.wait()
                active = None
                if code:
                    save(status='interrupted' if pending_signal else 'failed', returncode=code)
                    return code if code > 0 else 128-code
                # A duplicate --resume can report an existing writer and return
                # success. Never start the next stage while that writer is live.
                manifest = json.loads((stage_out/'manifest.json').read_text())
                if manifest['status'] != 'execution_finished':
                    save(status='failed', reason='previous_stage_still_running_or_incomplete')
                    print('前一阶段仍在运行或尚未完成；后续阶段未启动。', file=sys.stderr)
                    return 2
                state['completed_stages'].append(experiment)
                save()
            save(status='execution_finished')
            return 0
        except (OSError, ValueError, KeyError) as exc:
            save(status='failed', reason=str(exc))
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            if active is not None and active.poll() is None:
                forward(signal.SIGTERM, None)
                active.wait()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
