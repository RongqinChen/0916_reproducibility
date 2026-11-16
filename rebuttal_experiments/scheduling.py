"""Parallel BREC conditions with the original scheduler's CPU and RAM policy."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import tempfile
import time
from collections import Counter
from pathlib import Path

import psutil

from brec.determinism import configure_numpy
from brec.schedule import ordered_cpus
from .common import ResourceLimit, atomic_json
from .runtime import report_phase, report_to

GIB = 2**30
HEARTBEAT_SECONDS = 30


def estimate_bytes(nodes, width, q, matching_edges=0):
    """Conservative admission reservation; live RSS is monitored separately."""
    features_and_banks = nodes**2 * (4 * width + 2 * q + 12) * 256
    matching = min(matching_edges, nodes**4) * 512
    return 192 * 2**20 + features_and_banks + matching


def process_rss(pid):
    try:
        return psutil.Process(pid).memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def _execute(worker, task, target, cpu, seed, progress_path):
    # The parent owns cancellation; children must not inherit its SIGTERM handler.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    configure_numpy(seed)
    with report_to(progress_path):
        report_phase('starting')
        result = worker(task)
        report_phase('saving_result')
        atomic_json(target, result)


def run_tasks(tasks, worker, out, *, workers, memory_gib, reserve_gib, seed,
              record_directory='records', on_result=None, reservation_scale=1.0):
    """Run independent tasks, retaining one atomic record per completed condition.

    Linux fork shares the read-only dataset mapping. CPU ordering, reservations,
    system reserve and the RSS watchdog follow brec/schedule.py. An unexpectedly
    large worker is stopped and recorded, never retried on another numeric path.
    """
    if not math.isfinite(reservation_scale) or not 0 < reservation_scale <= 1:
        raise ValueError('Admission reservation scale must be in (0,1]')
    context = multiprocessing.get_context('fork')
    cpus = ordered_cpus()
    slots = min(workers, len(cpus)) if cpus else workers
    allowance = int(memory_gib * GIB)
    pending = sorted(tasks, key=lambda task: (-task['work'], task['stem']))
    active = {}
    records = out / record_directory
    records.mkdir(parents=True, exist_ok=True)
    progress_directory = tempfile.TemporaryDirectory(prefix='.brec-progress-', dir=out)
    completed = 0
    total = len(pending)
    scheduler_pid = os.getpid()
    started = last_result = time.monotonic()
    last_heartbeat = float('-inf')
    outcome = 'interrupted'
    previous = signal.getsignal(signal.SIGTERM)

    def estimate(task):
        return max(1, math.ceil(task['estimated_bytes']*reservation_scale))

    def stop(signum, frame):
        # A just-forked child can receive SIGTERM before _execute resets handlers.
        if os.getpid() != scheduler_pid:
            os._exit(128 + signum)
        raise KeyboardInterrupt('BREC scheduler interrupted')

    def save_limited(task, reason):
        atomic_json(records / (task['stem'] + '.json'),
                    {**task['record'], 'status': 'resource_limited', 'reason': reason})

    def heartbeat(force=False, status='running'):
        nonlocal last_heartbeat
        now = time.monotonic()
        if not force and now - last_heartbeat < HEARTBEAT_SECONDS:
            return
        rows = []
        for pid, entry in active.items():
            try:
                phase = json.loads(entry['progress_path'].read_text())
            except (OSError, ValueError):
                phase = {'phase': 'starting', 'started_monotonic': entry['started'],
                         'details': {}}
            rows.append({'pid': pid, 'cpu': entry['cpu'], 'condition': entry['task']['stem'],
                         'rss_bytes': process_rss(pid), 'elapsed_seconds': now - entry['started'],
                         'phase': phase['phase'], 'details': phase['details'],
                         'phase_elapsed_seconds': max(0, now - phase['started_monotonic'])})
        rss = process_rss(scheduler_pid) + sum(row['rss_bytes'] for row in rows)
        atomic_json(out / 'progress.json', {
            'status': status, 'scheduler_pid': scheduler_pid, 'updated_unix_seconds': time.time(),
            'elapsed_seconds': now - started, 'completed_this_session': completed,
            'total_this_session': total, 'pending': len(pending),
            'seconds_since_last_result': now - last_result,
            'rss_bytes': rss, 'memory_budget_bytes': allowance,
            'reserved_bytes': sum(entry['reservation'] for entry in active.values()),
            'memory_reservation_scale': reservation_scale, 'cpu_slots': slots, 'workers': rows})
        if rows and status == 'running':
            counts = ','.join(f'{key}={value}' for key, value in sorted(
                Counter(row['phase'] for row in rows).items()))
            oldest = max(rows, key=lambda row: row['phase_elapsed_seconds'])
            print(f'BREC heartbeat: active={len(rows)} pending={len(pending)}; '
                  f'no_result={now - last_result:.0f}s; RSS={rss / GIB:.2f} GiB; '
                  f'phases {counts}; oldest={oldest["condition"]} '
                  f'{oldest["phase"]} {oldest["phase_elapsed_seconds"]:.0f}s', flush=True)
        last_heartbeat = now

    signal.signal(signal.SIGTERM, stop)
    print(f'BREC: {slots} CPU slots; total RSS budget {memory_gib:g} GiB; '
          f'system reserve {reserve_gib:g} GiB; admission scale {reservation_scale:g}; '
          f'{total} pending conditions', flush=True)
    try:
        while pending or active:
            for pid, entry in list(active.items()):
                process = entry['process']
                if process.exitcode is None:
                    continue
                process.join()
                task = entry['task']
                target = records / (task['stem'] + '.json')
                if not target.exists():
                    if entry.get('memory_killed'):
                        save_limited(task, 'scheduler_memory_limit')
                    else:
                        raise RuntimeError(f'{task["stem"]}: worker exited '
                                           f'{process.exitcode} without a result')
                result = json.loads(target.read_text())
                for key, value in task['record'].items():
                    if result.get(key) != value:
                        raise ValueError(f'{task["stem"]}: worker result identity mismatch')
                del active[pid]
                entry['progress_path'].unlink(missing_ok=True)
                process.close()
                completed += 1
                last_result = time.monotonic()
                print(f'[{completed}/{total}] {task["stem"]}: {result["status"]}', flush=True)
                if on_result:
                    on_result(task, result)
                if result['status'] == 'implementation_error':
                    raise RuntimeError(f'{task["stem"]}: {result["reason"]}')

            parent_rss = process_rss(os.getpid())
            for pid, entry in active.items():
                entry['rss'] = process_rss(pid)
                entry['reservation'] = max(entry['reservation'], int(entry['rss'] * 1.15))
            live = sum(entry['rss'] for entry in active.values())
            available = psutil.virtual_memory().available
            if active and (parent_rss + live > allowance or available < reserve_gib * GIB):
                candidates = [entry for entry in active.values()
                              if not entry.get('memory_killed')
                              and not (records / (entry['task']['stem'] + '.json')).exists()]
                if candidates:
                    victim = max(candidates, key=lambda entry: entry['rss'])
                    victim['memory_killed'] = True
                    victim['process'].kill()
                heartbeat()
                time.sleep(0.1)
                continue

            launched = False
            while pending and len(active) < slots:
                reserved = sum(entry['reservation'] for entry in active.values())
                outstanding = sum(max(0, entry['reservation'] - entry['rss'])
                                  for entry in active.values())
                room = min(allowance - parent_rss - reserved,
                           psutil.virtual_memory().available - reserve_gib * GIB - outstanding)
                index = next((i for i, task in enumerate(pending)
                              if estimate(task) <= room), None)
                if index is None:
                    break
                task = pending.pop(index)
                used_cpus = {entry['cpu'] for entry in active.values()}
                cpu = next((cpu for cpu in cpus if cpu not in used_cpus), None)
                target = records / (task['stem'] + '.json')
                progress_path = Path(progress_directory.name) / (task['stem'] + '.json')
                process = context.Process(target=_execute,
                                          args=(worker, task, target, cpu, seed, progress_path))
                process.start()
                active[process.pid] = {'process': process, 'task': task, 'cpu': cpu,
                                       'rss': 0, 'reservation': estimate(task),
                                       'started': time.monotonic(), 'progress_path': progress_path}
                launched = True

            if pending and not active:
                oversized = [task for task in pending
                             if estimate(task) + parent_rss > allowance]
                if oversized:
                    for task in oversized:
                        save_limited(task, 'estimated_memory_limit')
                        pending.remove(task)
                        completed += 1
                        last_result = time.monotonic()
                        print(f'[{completed}/{total}] {task["stem"]}: estimated_memory_limit', flush=True)
                        if on_result:
                            on_result(task, json.loads((records / (task['stem'] + '.json')).read_text()))
                    continue
                raise ResourceLimit('scheduler_memory_unavailable: system reserve prevents admission; '
                                    'completed conditions are retained for --resume')
            heartbeat()
            if active and not launched:
                time.sleep(0.1)
        outcome = 'finished'
    except KeyboardInterrupt:
        raise
    except BaseException:
        outcome = 'failed'
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        # Stop every child before releasing the parent's output-directory lock.
        for entry in active.values():
            if entry['process'].is_alive():
                entry['process'].terminate()
        for entry in active.values():
            process = entry['process']
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        active.clear()
        try:
            heartbeat(force=True, status=outcome)
        finally:
            progress_directory.cleanup()
