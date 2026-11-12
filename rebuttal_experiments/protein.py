"""Protein experiment commands: prepare, train, predict and report."""
from __future__ import annotations

import argparse
from pathlib import Path

from .protein_data import VARIANTS, prepare
from .protein_results import predict, report
from .protein_training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--config', default='configs/rebuttal/protein.json')
    p.add_argument('--dataset', choices=['ENZYMES', 'PROTEINS', 'PROTEINS_full'])
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--resume', action='store_true')
    p.set_defaults(function=prepare)
    p = commands.add_parser('train')
    p.add_argument('--config', default='configs/rebuttal/protein.json')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--folds', nargs='+', type=int)
    p.add_argument('--variants', nargs='+', choices=VARIANTS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-incomplete', action='store_true',
                   help='Discard this run\'s incomplete job directories and restart those jobs')
    p.set_defaults(function=train)
    p = commands.add_parser('predict')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.set_defaults(function=predict)
    p = commands.add_parser('report')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(function=report)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == '__main__':
    main()
