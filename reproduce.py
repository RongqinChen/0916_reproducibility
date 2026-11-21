#!/usr/bin/env python3
"""Repository entry point for float32 BREC and object discrimination experiments."""
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
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        env[key] = str(threads)
    if args.action == 'brec':
        env.update(MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_brec(args):
    orders = (2, 3, 4) if args.order == 'all' else (int(args.order),)
    for order in orders:
        settings = {'brec_file': args.brec_file, 'out': args.out/f'fwl{order}'}
        if order in (2, 3):
            settings['workers'] = args.workers
        else:
            settings.update(memory_gib=args.memory_gib, devices=args.devices, ids=args.ids)
        if args.order == 'all' and order > 2:
            settings['lower_order_out'] = args.out/f'fwl{order-1}'
        launch(args, [sys.executable, '-u', '-m', 'brec.float32', str(order), *flags(settings)],
               args.threads if order == 4 else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    datasets = config('object_datasets')
    sub.add_parser('list', help='list object datasets, domains, and evidence status')
    p = sub.add_parser('brec', help='run/resume the float32 2-, 3-, and 4-FWL experiments')
    p.add_argument('--brec-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--order', choices=('2', '3', '4', 'all'), default='all')
    p.add_argument('--workers', type=int, default=1, help='CPU workers for 2- and 3-FWL')
    p.add_argument('--threads', type=int, default=1, help='OpenMP threads for CPU 4-FWL')
    p.add_argument('--memory-gib', type=float, default=58, help='host memory limit for 4-FWL')
    p.add_argument('--devices', type=int, nargs=2, help='CUDA device indices for 4-FWL')
    p.add_argument('--ids', type=int, nargs='+', help='subset of the 4-FWL target cohort')
    p.add_argument('--dry-run', action='store_true', help='print commands without executing')
    for name in ('prepare', 'objects'):
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
    args = parser.parse_args(argv)
    if args.action == 'brec' and args.ids and args.order != '4':
        parser.error('--ids requires --order 4')
    for name in ('topology', 'data_root', 'out', 'brec_file'):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    for name in ('workers', 'memory_gib', 'threads'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(name.replace('_', '-') + ' must be positive')
    py = sys.executable
    if args.action == 'brec':
        return run_brec(args)
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


if __name__ == '__main__':
    main()
