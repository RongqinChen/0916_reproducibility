#!/usr/bin/env python3
"""Plan or run equal-width matched GNNPlus/GINE encoding conditions on one dataset."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# config, dataset label, task metric, direction, default injection domain
DATASETS = {
    'zinc': ('zinc', 'ZINC-subset', 'mae', 'min', 'edge'),
    'molhiv': ('hiv', 'ogbg-molhiv', 'auc', 'max', 'edge'),
    'pcba': ('pcba', 'ogbg-molpcba', 'ap', 'max', 'edge'),
    'peptides-functional': ('peptides-func', 'peptides-functional', 'ap', 'max', 'edge'),
    'peptides-structural': ('peptides-struct', 'peptides-structural', 'mae', 'min', 'edge'),
    'mnist': ('mnist', 'MNIST', 'accuracy', 'max', 'node'),
    'cifar10': ('cifar10', 'CIFAR10', 'accuracy', 'max', 'node'),
    'cluster': ('cluster', 'CLUSTER', 'accuracy-SBM', 'max', 'node'),
    'pattern': ('pattern', 'PATTERN', 'accuracy-SBM', 'max', 'node'),
    'voc': ('voc', 'VOC', 'f1', 'max', 'node'),
    'coco': ('coco', 'COCO', 'f1', 'max', 'node'),
    'ppa': ('ppa', 'ogbg-ppa', 'accuracy', 'max', 'edge'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--domain', choices=['node', 'edge'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--widths', type=int, nargs='+', default=[20])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--degree-budget', type=int, default=14)
    parser.add_argument('--encodings', nargs='+', choices=['powers', 'endpoints', 'mixed', 'wider_powers'],
                        default=['powers', 'endpoints', 'mixed'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config, label, metric, direction, domain = DATASETS[args.dataset]
    command = [sys.executable, '-m', 'downstream.grid', '--engine', 'gnnplus',
               '--config', str(ROOT / 'gnnplus_official/configs/gine' / f'{config}.yaml'),
               '--dataset', label, '--metric', metric, '--direction', direction,
               '--selection-metric', metric, '--selection-direction', direction,
               '--split-id', 'official-0', '--split-index', '0',
               '--domain', args.domain or domain, '--degree-budget', str(args.degree_budget),
               '--widths', *map(str, args.widths), '--seeds', *map(str, args.seeds),
               '--encodings', *args.encodings, '--data-root', str(args.data_root.resolve()),
               '--out-dir', str(args.out_dir.resolve()), '--device', args.device]
    if args.execute:
        command.append('--execute')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
