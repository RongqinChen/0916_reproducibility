"""Command-line definition shared by direct and registry-backed runs."""

import argparse
import os

import numpy as np


def add_protocol_arguments(parser: argparse.ArgumentParser, include_data=True) -> None:
    if include_data:
        parser.add_argument('--dataset', choices=('zinc', 'npz'), default='zinc')
        parser.add_argument('--root', default='data/ZINC')
        parser.add_argument('--full', action='store_true')
        parser.add_argument('--limit', type=int, default=0)
        parser.add_argument('--out', default='')
        parser.add_argument('--tag', default='')
    parser.add_argument('--domains', nargs='+', choices=('nodes','edges','pairs'),
                        default=['nodes','edges','pairs'], help='Object domains actually evaluated')
    parser.add_argument('--dims', type=int, nargs='+', default=[6, 8, 10, 12, 14],
                        help='branch widths (use --degree-budgets for complete powers by maximum degree)')
    parser.add_argument('--degree-budgets', type=int, nargs='+', default=None,
                        help='evaluate complete strict-Q Powers, Endpoints, and Mixed dictionaries')
    parser.add_argument('--routes', nargs='+',
                        choices=('powers', 'endpoints', 'mixed',
                                 'powers_2d', 'endpoints_2d'),
                        default=['powers', 'endpoints', 'mixed',
                                 'powers_2d', 'endpoints_2d'],
                        help='encoding routes to evaluate')
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--max-compare-distance', type=int, default=None,
                        help='omit node comparisons whose shortest-path distance exceeds this value; edge/pair domains stay ungated')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--memory-gb', type=float, default=8,
                        help='estimated concurrent worker-memory budget, not a hard RSS limit')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summaries-only', action='store_true',
                        help='write only dataset summaries; skip per-graph '
                             'checkpoints and per-graph witness outputs')
    parser.add_argument('--stream-results', action='store_true',
                        help='append per-graph result rows to disk as graphs '
                             'finish, without witness or checkpoint data, so an '
                             'interrupted run can continue where it stopped')
    parser.add_argument('--budget', type=int, default=200000)
    parser.add_argument('--no-mis', action='store_true',
                        help='skip Representative Capacity (maximum independent set)')
    parser.add_argument('--permutation-checks', type=int, default=3)
    parser.add_argument('--paired-permutations', type=int, default=10000)
    parser.add_argument('--sesoi', type=float, default=0.01)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--pair-details', choices=('none', 'differences', 'all'),
                        default='differences',
                        help='write route-difference object identities; all can be large')
    parser.add_argument('--seed', type=int, default=0)


def validate_protocol_args(args, parser: argparse.ArgumentParser) -> None:
    if args.threads_per_worker < 1 or not np.isfinite(args.memory_gb) or args.memory_gb <= 0:
        parser.error('threads-per-worker and memory-gb must be positive')
    if args.resume and not args.out:
        parser.error('--resume requires --out')
    if args.resume and args.summaries_only:
        parser.error('--summaries-only cannot be used with --resume')
    if args.stream_results and not args.out:
        parser.error('--stream-results requires --out')
    if min(args.dims) < 1:
        parser.error('dims must be positive')
    if args.max_compare_distance is not None and args.max_compare_distance < 0:
        parser.error('max-compare-distance must be nonnegative')
    if args.degree_budgets is not None:
        if not args.degree_budgets or min(args.degree_budgets) < 2:
            parser.error('degree budgets must be >=2')
        if len(set(args.degree_budgets)) != len(args.degree_budgets):
            parser.error('degree budgets must be distinct')
        if (not args.routes or any(route not in ('powers', 'endpoints', 'mixed')
                                   for route in args.routes)):
            parser.error('degree-budget mode supports only powers, endpoints, and mixed routes')
        if args.standardize:
            parser.error('degree-budget mode uses raw features; do not standardize')

    if (args.chunk_size < 1 or args.workers < 1 or args.budget < 1
            or args.permutation_checks < 0 or getattr(args, 'limit', 0) < 0):
        parser.error('chunk-size, workers and budget must be positive; '
                     'limit and permutation-checks must be nonnegative')
    if args.paired_permutations < 1:
        parser.error('paired-permutations must be positive')
    if not np.isfinite(args.sesoi) or args.sesoi <= 0:
        parser.error('sesoi must be finite and strictly positive')
    if not all(np.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error('tolerances must be finite and nonnegative')
