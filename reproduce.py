#!/usr/bin/env python3
"""Repository entry point for exact BREC experiments and the paper's other protocols."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def config(name):
    return json.loads((ROOT / 'configs' / f'{name}.json').read_text())


def flags(settings):
    result = []
    for key, value in settings.items():
        if value is None or value is False:
            continue
        result.append('--' + key.replace('_', '-'))
        if value is not True:
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def launch(args, command, threads=1):
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    if args.action.startswith('brec-') or args.action == 'ppgn':
        from brec.determinism import child_environment
        env = child_environment(env)
        threads = 1
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        env[key] = str(threads)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


BREC_ACTIONS = ('brec-plan', 'brec-run', 'brec-report', 'brec-rescore', 'brec-worker')


def main():
    if len(sys.argv) > 1 and sys.argv[1] in BREC_ACTIONS:
        # Establish the process policy before importing numerical libraries.
        from brec.determinism import ensure_startup, prepare_environment
        prepare_environment()
        ensure_startup()
        from brec.suite_cli import main as brec_main
        return brec_main(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    datasets = config('object_datasets')
    sub.add_parser('list', help='list object datasets, domains, and evidence status')
    for command, description in (
        ('brec-plan', 'prepare the complete exact BREC campaign without executing it'),
        ('brec-run', 'run/resume BREC with exact aggregates and memory-bounded parallelism'),
        ('brec-report', 'summarize BREC results and unresolved/failed cases'),
        ('brec-rescore', 'recompute numerical neural scores from saved embeddings')):
        sub.add_parser(command, help=description)
    for name in ('prepare', 'objects', 'ppgn'):
        p = sub.add_parser(name)
        p.add_argument('--dry-run', action='store_true', help='print commands without executing')
        if name in ('prepare', 'objects'):
            p.add_argument('--dataset', choices=sorted(datasets), required=True)
            p.add_argument('--topology', type=Path, required=True)
        if name == 'prepare':
            p.add_argument('--data-root', type=Path, required=True)
        if name == 'objects':
            p.add_argument('--out', type=Path, required=True)
            p.add_argument('--workers', type=int, default=12)
            p.add_argument('--memory-gib', type=float, default=100)
            p.add_argument('--resume', action='store_true')
        if name == 'ppgn':
            p.add_argument('--brec-file', type=Path, required=True)
            p.add_argument('--trust-pickle', action='store_true',
                           help='allow loading the official object-array BREC file')
        if name == 'ppgn':
            p.add_argument('--out', type=Path, required=True)
            p.add_argument('--gpus', nargs='+', default=['0', '1'])
            p.add_argument('--physical-batch', type=int, choices=[2, 4, 8, 16, 32], default=32)
            p.add_argument('--cpu-workers', type=int, default=1)
            p.add_argument('--threads', type=int, default=1)
            p.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'ppgn' and args.threads != 1:
        parser.error('BREC deterministic runs require --threads 1')
    for name in ('topology', 'data_root', 'out', 'root', 'brec_file', 'scratch_root'):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    for name in ('workers', 'memory_gib', 'memory_factor', 'cpu_workers', 'threads'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(name.replace('_', '-') + ' must be positive')
    py = sys.executable
    if args.action == 'list':
        for key, entry in datasets.items():
            print(f"{key:20} {entry['graphs']:7}  {','.join(entry['domains']):18} "
                  f"node_gate={entry['max_compare_distance']}  {entry['evidence']}")
        return
    if args.action == 'prepare':
        entry = datasets[args.dataset]
        if args.topology.exists():
            parser.error('Topology already exists; reuse it or choose a new filename')
        if not args.dry_run:
            args.topology.parent.mkdir(parents=True, exist_ok=True)
        command = [py, '-m', 'discriminability.dump_topology',
                   '--format', entry['format'], '--name', entry['name'],
                   '--root', str(args.data_root), '--out', str(args.topology),
                   '--max-nodes', '0', '--limit', '0']
        if 'slic_compactness' in entry:
            command += ['--slic-compactness', str(entry['slic_compactness'])]
        launch(args, command)
    elif args.action == 'objects':
        if not args.topology.is_file() and not args.dry_run:
            parser.error('Topology is missing; run prepare first or supply an existing NPZ')
        entry = datasets[args.dataset]
        settings = config('object_protocol')
        settings.update(dataset='npz', root=str(args.topology), out=str(args.out),
                        tag='rerun', domains=entry['domains'],
                        max_compare_distance=entry['max_compare_distance'],
                        workers=args.workers, memory_gb=args.memory_gib,
                        resume=args.resume)
        launch(args, [py, '-m', 'discriminability.run', *flags(settings)])
    elif args.action == 'ppgn':
        settings = config('ppgn_protocol')
        settings.update(brec_file=str(args.brec_file), out_dir=str(args.out),
                        gpus=args.gpus, trust_pickle=args.trust_pickle,
                        physical_batch=args.physical_batch, threads=args.threads,
                        cpu_workers=args.cpu_workers, resume=args.resume)
        launch(args, [py, '-m', 'brec.ppgn_schedule', *flags(settings)], args.threads)


if __name__ == '__main__':
    main()
