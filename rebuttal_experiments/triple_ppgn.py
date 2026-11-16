"""Three paired legacy 3-PPGN conditions on BREC, with frozen worker sources."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from brec.determinism import prepare_environment, child_environment, ensure_startup, start_brec
prepare_environment()
import psutil

from .brec import graph_at, load_brec_data
from .common import (ResourceLimit, atomic_json, configuration, run_directory, sha256)
from .scheduling import ordered_cpus
from .triple_ppgn_inputs import VARIANTS


def validate_config(config):
    if (config['variants'] != list(VARIANTS) or config['basis'] != 'chebyshev'
            or config['triple_q'] != 1 or config['dtype'] != 'float32'
            or config['physical_batch'] != 1 or config['gradient_cache'] is not True):
        raise ValueError('Freeze the three Q=1 augmentations, float32 and single-graph gradient caching')
    for key in ('seed_q', 'hidden_dim', 'layers', 'output_dim', 'epochs', 'effective_batch', 'input_width'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Positive integer required: {key}')
    if config['effective_batch'] % 2 or 64 % config['effective_batch']:
        raise ValueError('Effective batch must be even and divide 64')
    if config['input_width'] != 1 + 6*(config['seed_q']+2) + 27:
        raise ValueError('Input width does not match the frozen legacy+augmentation dictionary')
    seeds = config['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError('Distinct nonnegative seeds required')
    for key in ('lr', 'rpc_threshold', 'memory_gib'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Positive finite value required: {key}')
    for key in ('weight_decay', 'early_stop_loss', 'input_cache_gib', 'reserve_gib'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Nonnegative finite value required: {key}')
    if config['seconds_per_condition'] is not None:
        raise ValueError('This protocol has no per-condition time limit')


def freeze_sources(out):
    destination = out/'frozen'
    source = Path(__file__).resolve().parents[1]
    if destination.exists():
        for package in ('rebuttal_experiments', 'brec', 'supplementary'):
            for path in (source/package).glob('*.py'):
                frozen = destination/package/path.name
                if not frozen.is_file() or sha256(path) != sha256(frozen):
                    raise ValueError('Frozen source identity mismatch')
        return destination
    destination.mkdir()
    for package in ('rebuttal_experiments', 'brec', 'supplementary'):
        target = destination/package
        target.mkdir()
        for path in (source/package).glob('*.py'):
            shutil.copy2(path, target/path.name)
    return destination


def record_path(out, pid, seed, variant):
    return out/'records'/f'pair{pid:03d}_seed{seed}_{variant}.json'


def valid_record(path, identity, pid, seed, variant, check_artifacts=False):
    row = json.loads(path.read_text())
    expected = {'identity_sha256': identity, 'pair_id': pid, 'seed': seed, 'variant': variant, 'order': 3}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Record identity mismatch: {path}')
    if row['status'] != 'complete':
        raise ValueError(f'Inspect failed condition before resuming: {path}')
    if row['evaluation']['outcome'] not in ('separated', 'not_separated', 'unresolved'):
        raise ValueError(f'Unknown RPC outcome: {path}')
    if check_artifacts:
        for name, digest in row['artifacts'].items():
            artifact = path.parent.parent/name
            if not artifact.is_file() or sha256(artifact) != digest:
                raise ValueError(f'Incomplete or changed artifact: {artifact}')
    return row


def summarize(out, config, identity):
    found = {}
    settings = []
    for seed in config['seeds']:
        for variant in config['variants']:
            counts = Counter()
            for pid in config['executed_pair_ids']:
                path = record_path(out, pid, seed, variant)
                if not path.exists():
                    counts['not_executed'] += 1
                    continue
                row = json.loads(path.read_text())
                if row.get('identity_sha256') != identity:
                    raise ValueError('Summary encountered a record with the wrong identity')
                found[pid, seed, variant] = row
                if row['status'] != 'complete':
                    counts[row['status']] += 1
                elif row.get('reference_upper_bound_violation'):
                    counts['reference_conflict'] += 1
                else:
                    counts[row['evaluation']['outcome']] += 1
            counts['not_requested'] += 400-len(config['executed_pair_ids'])
            settings.append({'seed': seed, 'variant': variant, 'denominator': 400,
                             'counts': dict(counts), 'separated_lower_bound': counts['separated']})
    paired = []
    for seed in config['seeds']:
        for first, second in (('original', 'baseline'), ('baseline', 'joint'), ('original', 'joint')):
            outcomes = {key: [] for key in ('gained', 'lost', 'both', 'neither', 'unresolved')}
            for pid in config['executed_pair_ids']:
                rows = [found.get((pid, seed, variant)) for variant in (first, second)]
                if any(row is None or row['status'] != 'complete'
                       or row.get('reference_upper_bound_violation')
                       or row['evaluation']['outcome'] == 'unresolved' for row in rows):
                    outcomes['unresolved'].append(pid)
                    continue
                a, b = [row['evaluation']['outcome'] == 'separated' for row in rows]
                outcomes['both' if a and b else 'lost' if a else 'gained' if b else 'neither'].append(pid)
            paired.append({'seed': seed, 'first': first, 'second': second,
                           'counts': {key: len(ids) for key, ids in outcomes.items()}, 'pair_ids': outcomes})
    atomic_json(out/'summary.json', {'settings': settings, 'paired': paired,
        'scope': 'Empirical neural RPC decisions; unresolved and missing measurements stay in the denominator. Input width and initialization are paired; graph sizes are not omitted.'})


def process_rss_tree(pid):
    try:
        parent = psutil.Process(pid)
        return sum(process.memory_info().rss for process in [parent, *parent.children(recursive=True)])
    except psutil.NoSuchProcess:
        return 0


def schedule(out, config, identity, frozen, devices):
    data = load_brec_data(config['brec_file'], config['expected_brec_sha256'])
    tasks = []
    for pid in config['executed_pair_ids']:
        n = max(len(graph_at(data, 64*(pid+offset)+i)) for offset in (0,400) for i in (0,1))
        for seed in config['seeds']:
            complete = True
            for variant in config['variants']:
                path = record_path(out, pid, seed, variant)
                if path.exists():
                    valid_record(path, identity, pid, seed, variant, True)
                else:
                    complete = False
            if not complete:
                # Up to four input/feature buffers plus the bounded CPU cache.
                reservation = int(4*n**3*config['input_width']*4 + config['input_cache_gib']*2**30 + 2*2**30)
                tasks.append({'pid': pid, 'seed': seed, 'nodes': n, 'reserved': reservation})
    tasks.sort(key=lambda row: (row['nodes'], row['pid'], row['seed']))
    slots, cpus, active = list(devices), ordered_cpus(), {}
    started, last_heartbeat = time.monotonic(), float('-inf')
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    final_status = 'interrupted'

    def heartbeat(force=False, status='running'):
        nonlocal last_heartbeat
        now = time.monotonic()
        if not force and now-last_heartbeat < 30:
            return
        workers = []
        for device, entry in active.items():
            task = entry['task']
            phase_path = out/'workers'/f'pair{task["pid"]:03d}_seed{task["seed"]}.json'
            phase = json.loads(phase_path.read_text()) if phase_path.exists() else {'phase': 'starting'}
            workers.append({'device': device, 'pid': entry['process'].pid,
                            'pair_id': task['pid'], 'seed': task['seed'], 'nodes': task['nodes'],
                            'elapsed_seconds': now-entry['started'], 'progress': phase})
        completed = len(list((out/'records').glob('*.json')))
        atomic_json(out/'progress.json', {'status': status, 'completed_records': completed,
            'total_records': len(config['executed_pair_ids'])*len(config['seeds'])*len(config['variants']),
            'pending_jobs': len(tasks), 'elapsed_seconds': now-started,
            'rss_bytes': process_rss_tree(os.getpid()), 'workers': workers, 'updated_unix_seconds': time.time()})
        summarize(out, config, identity)
        print(f'3-PPGN: {completed} records; {len(active)} workers; {len(tasks)} pending jobs; '
              f'elapsed {now-started:.0f}s', flush=True)
        last_heartbeat = now

    try:
        while tasks or active:
            for device, entry in list(active.items()):
                code = entry['process'].poll()
                if code is None:
                    continue
                entry['log'].close()
                task = entry['task']
                if code != 0:
                    raise RuntimeError(f'pair{task["pid"]:03d} seed{task["seed"]} on {device} exited {code}; inspect {entry["log_path"]}')
                for variant in config['variants']:
                    valid_record(record_path(out, task['pid'], task['seed'], variant), identity,
                                 task['pid'], task['seed'], variant, True)
                del active[device]
            if process_rss_tree(os.getpid()) > config['memory_gib']*2**30:
                raise ResourceLimit('Total scheduler+worker RSS exceeds the frozen budget')
            for device in slots:
                if device in active or not tasks:
                    continue
                reserved = sum(entry['task']['reserved'] for entry in active.values())
                room = min(config['memory_gib']*2**30 - psutil.Process().memory_info().rss-reserved,
                           psutil.virtual_memory().available-config['reserve_gib']*2**30-reserved)
                index = next((i for i, task in enumerate(tasks) if task['reserved'] <= room), None)
                if index is None:
                    continue
                task = tasks.pop(index)
                stem = f'pair{task["pid"]:03d}_seed{task["seed"]}'
                log_path = out/'logs'/f'{stem}.log'
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open('a')
                cpu = cpus[slots.index(device) % len(cpus)] if cpus else -1
                command = [sys.executable, '-u', '-m', 'rebuttal_experiments.triple_ppgn',
                           '--worker', '--out', str(out), '--pair-id', str(task['pid']),
                           '--seed', str(task['seed']), '--device', device, '--cpu', str(cpu)]
                process = subprocess.Popen(command, cwd=frozen, env=child_environment(),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[device] = {'process': process, 'task': task, 'log': log,
                                  'log_path': log_path, 'started': time.monotonic()}
            if tasks and not active:
                raise ResourceLimit('No pending job fits the total RAM budget/system reserve')
            heartbeat()
            if active:
                time.sleep(.5)
        final_status = 'finished'
    except KeyboardInterrupt:
        raise
    except BaseException:
        final_status = 'failed'
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        for entry in active.values():
            if entry['process'].poll() is None:
                os.killpg(entry['process'].pid, signal.SIGTERM)
        for entry in active.values():
            try:
                entry['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(entry['process'].pid, signal.SIGKILL)
                entry['process'].wait()
            entry['log'].close()
        active.clear()
        heartbeat(True, final_status)


def main(argv=None):
    ensure_startup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/rebuttal/triple_ppgn.json'))
    parser.add_argument('--brec-file', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--pair-ids', nargs='+', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--pair-id', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--device', help=argparse.SUPPRESS)
    parser.add_argument('--cpu', type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.cpu >= 0:
            os.sched_setaffinity(0, {args.cpu})
        from .triple_ppgn_training import worker
        worker(args.out.resolve(), args.pair_id, args.seed, args.device)
        return
    if args.brec_file is None:
        parser.error('--brec-file is required')
    config = configuration(args.config)
    if args.seeds is not None:
        config['seeds'] = args.seeds
    validate_config(config)
    ids = args.pair_ids if args.pair_ids is not None else list(range(400))
    if not ids or len(set(ids)) != len(ids) or any(pid < 0 or pid >= 400 for pid in ids):
        parser.error('Distinct pair IDs in 0..399 required')
    if len(set(args.devices)) != len(args.devices) or not args.devices:
        parser.error('Distinct devices required')
    import torch
    determinism = start_brec(config['seeds'][0], use_torch=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_metadata = {}
    for device in args.devices:
        if device == 'cpu':
            continue
        if not device.startswith('cuda:') or not torch.cuda.is_available():
            parser.error(f'Unavailable device: {device}')
        props = torch.cuda.get_device_properties(device)
        gpu_metadata[device] = {'name': props.name, 'total_memory': props.total_memory}
    legacy_path = Path(__file__).with_name('legacy_triple_ppgn.py')
    assignments = ast.parse(legacy_path.read_text()).body
    legacy_sha = next(ast.literal_eval(node.value) for node in assignments
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                          t.id == 'LEGACY_SOURCE_SHA256' for t in node.targets))
    config.update(executed_pair_ids=ids, brec_file=str(args.brec_file.resolve()),
                  devices=args.devices, gpu_metadata=gpu_metadata, determinism=determinism,
                  legacy_backbone_source_sha256=legacy_sha,
                  input_policy='legacy-chebyshev-Q8-plus-zero-padded-Q1-ordered-triple-channels',
                  gradient_policy='embedding-chain-rule-cache; no batch-dependent or stochastic layers')
    out = args.out.resolve()
    with run_directory(out, config, [args.config.resolve(), args.brec_file.resolve()], args.resume) as (out, identity):
        (out/'records').mkdir(exist_ok=True)
        frozen = freeze_sources(out)
        schedule(out, config, identity, frozen, args.devices)


if __name__ == '__main__':
    main()
