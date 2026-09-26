#!/usr/bin/env python
r"""Run the within-graph discrimination protocol for one dataset.

The runner accepts either a named dataset loadable by GNNPlus/PyG/OGB or an
existing topology npz.  Both sources feed the same protocol; there is no
separate "built-in" versus "other dataset" path.

For a named dataset the command runs two stages:

1. `discriminability.dump_topology` reads standard datasets directly through PyG and lazily
   uses GNNPlus adapters for project-specific datasets, then writes the
   unweighted topology to a compact npz;
2. `discriminability.run --dataset npz` runs the protocol on that npz.

Two interpreters are involved because the two halves have disjoint dependencies:

* the dump needs `torch` + `torch_geometric` (GNNPlus's loader imports PyG
  datasets), which is `requirements-grit.txt`;
* the protocol needs `numpy` + `networkx` + `pynauty`, which is
  `requirements.txt`.

Point `--dump-python` / `--run-python` at the two environments. If a single
environment satisfies both, the defaults (`sys.executable`) are enough.

    python -m discriminability.datasets \
        --format PyG-TUDataset --name ENZYMES \
        --data-root data --max-nodes 60 \
        --out discriminability_results/object_coverage_v8/ENZYMES

An existing topology dump uses the same entry point without dummy dataset
arguments:

    python -m discriminability.datasets \
        --npz data/ENZYMES_topology.npz --skip-dump \
        --out discriminability_results/object_coverage_v8/ENZYMES

The protocol keeps one row per (graph, domain, encoding, branch width), so a
run over `--dims 6 8 10 12 14` and 5 routes costs `3 * 5 * 5` rows per graph.
Large graphs dominate the cost: the all-pairs domain holds `n(n-1)/2` objects
and its similarity matrix is quadratic in that, so `--max-nodes` is the main
runtime lever. It is applied at dump time and is recorded in the run manifest,
because dropping graphs changes the dataset the numbers describe.

Every independent run uses a fresh output directory. Resume requires the same
saved topology NPZ with `--skip-dump --npz --resume`; streamed runs also verify
the exact input, source, and scientific configuration before continuing. The
output directory gets `run_<tag>.log` and `manifest_<tag>.json`, whose status
is `complete` only when both stages exited zero.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--format', default='',
                        help='GNNPlus format id: PyG-TUDataset, PyG-ZINC, '
                             'PyG-AQSOL, PyG-GNNBenchmarkDataset, PyG-MalNetTiny, '
                             'PyG-VOCSuperpixels, PyG-COSuperpixels, OGB')
    parser.add_argument('--name', default='',
                        help='GNNPlus dataset name, e.g. NCI1, CSL, ogbg-molhiv')
    parser.add_argument('--data-root', default='data',
                        help='dataset cache directory GNNPlus reads/writes')
    parser.add_argument('--max-nodes', type=int, default=0,
                        help='drop graphs with more nodes (0 = keep all); the '
                             'dominant runtime lever')
    parser.add_argument('--limit', type=int, default=0,
                        help='keep at most this many graphs (0 = all)')
    parser.add_argument('--slic-compactness', type=int, default=10,
                        help='only used by the Superpixels formats')

    parser.add_argument('--out', required=True,
                        help='output directory for this run')
    parser.add_argument('--tag', default='',
                        help='filename component; defaults to dataset name or npz stem')
    parser.add_argument('--npz', default='',
                        help='topology npz path; defaults to <out>/topology.npz')
    parser.add_argument('--skip-dump', action='store_true',
                        help='reuse --npz instead of re-dumping')
    parser.add_argument('--resume', action='store_true',
                        help='resume discriminability.run from its checkpoints; '
                             'requires --skip-dump and the original output directory')

    parser.add_argument('--dump-python', default=sys.executable,
                        help='interpreter with torch + torch_geometric')
    parser.add_argument('--run-python', default=sys.executable,
                        help='interpreter with numpy + networkx + pynauty')
    parser.add_argument('--gnnplus', default=str(REPO / 'gnnplus_official'),
                        help='path to the GNNPlus checkout')

    parser.add_argument('--domains', nargs='+', choices=('nodes','edges','pairs'),
                        default=['nodes','edges','pairs'], help='Object domains actually evaluated')
    parser.add_argument('--dims', type=int, nargs='+', default=[6, 8, 10, 12, 14],
                        help='branch widths')
    parser.add_argument('--degree-budgets', type=int, nargs='+', default=None,
                        help='complete Powers dictionaries through each degree Q')
    parser.add_argument('--routes', nargs='+',
                        choices=('powers', 'endpoints', 'mixed',
                                 'powers_2d', 'endpoints_2d'),
                        default=['powers', 'endpoints', 'mixed',
                                 'powers_2d', 'endpoints_2d'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--memory-gb', type=float, default=8)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--budget', type=int, default=200000,
                        help='recursion-node budget per MIS solve')
    parser.add_argument('--no-mis', action='store_true',
                        help='skip the legacy auxiliary MIS search')
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--max-compare-distance', type=int, default=None,
                        help='omit node comparisons beyond this shortest-path distance; edge/pair domains remain ungated')
    parser.add_argument('--sesoi', type=float, default=0.01)
    parser.add_argument('--permutation-checks', type=int, default=1)
    parser.add_argument('--paired-permutations', type=int, default=10000)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--summaries-only', action='store_true')
    parser.add_argument('--stream-results', action='store_true')
    parser.add_argument('--pair-details', choices=('none', 'differences', 'all'),
                        default='differences')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def resolve_source(args, parser: argparse.ArgumentParser):
    """Validate and normalize the named-dataset or existing-npz source."""
    if bool(args.format) != bool(args.name):
        parser.error('--format and --name must be provided together')
    if args.skip_dump:
        if args.format:
            parser.error('choose one source: --format/--name or --skip-dump --npz')
        if not args.npz:
            parser.error('--skip-dump requires --npz')
        npz = Path(args.npz).resolve()
        if not npz.is_file():
            parser.error(f'topology npz does not exist: {npz}')
        return {'kind': 'npz', 'label': npz.stem, 'npz': npz}
    if not args.format:
        parser.error('provide --format and --name, or use --skip-dump --npz')
    output = Path(args.out).resolve()
    npz = Path(args.npz).resolve() if args.npz else output / 'topology.npz'
    return {'kind': 'named_dataset', 'label': args.name, 'npz': npz}


def build_dump_command(args, npz: Path):
    return [args.dump_python, '-m', 'discriminability.dump_topology',
            '--format', args.format, '--name', args.name,
            '--root', args.data_root, '--gnnplus', args.gnnplus,
            '--slic-compactness', str(args.slic_compactness),
            '--max-nodes', str(args.max_nodes),
            '--limit', str(args.limit), '--out', str(npz)]


def build_protocol_command(args, npz: Path, tag: str):
    command = [
        args.run_python, '-m', 'discriminability.run',
        '--dataset', 'npz', '--root', str(npz),
        '--dims', *[str(width) for width in sorted(set(args.dims))],
        '--routes', *args.routes, '--domains', *args.domains,
        '--workers', str(args.workers), '--chunk-size', str(args.chunk_size),
        '--threads-per-worker', str(args.threads_per_worker),
        '--memory-gb', str(args.memory_gb),
        '--budget', str(args.budget), '--atol', str(args.atol),
        '--rtol', str(args.rtol), '--sesoi', str(args.sesoi),
        '--pair-details', args.pair_details,
        '--permutation-checks', str(args.permutation_checks),
        '--paired-permutations', str(args.paired_permutations),
        '--seed', str(args.seed), '--out', str(Path(args.out).resolve()),
        '--tag', tag,
    ]
    if args.max_compare_distance is not None:
        command.extend(['--max-compare-distance', str(args.max_compare_distance)])
    if args.degree_budgets is not None:
        command.extend(['--degree-budgets', *map(str, sorted(set(args.degree_budgets)))])
    if args.standardize:
        command.append('--standardize')
    if args.summaries_only:
        command.append('--summaries-only')
    if args.stream_results:
        command.append('--stream-results')
    if args.resume:
        command.append('--resume')
    if args.no_mis:
        command.append('--no-mis')
    return command


def run(command, label: str, log, env=None) -> None:
    """Run one stage, echoing it to the log and to stdout."""
    line = '$ ' + ' '.join(command)
    print(line, flush=True)
    log.write(line + '\n')
    log.flush()
    started = time.perf_counter()
    process = subprocess.run(command, stdout=subprocess.PIPE, env=env,
                             stderr=subprocess.STDOUT, text=True)
    log.write(process.stdout)
    log.flush()
    print(process.stdout[-4000:], flush=True)
    elapsed = time.perf_counter() - started
    print(f'[{label}] exit={process.returncode} in {elapsed:.1f}s', flush=True)
    if process.returncode != 0:
        raise SystemExit(f'{label} failed with exit code {process.returncode}')


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.resume and (not args.skip_dump or not args.npz):
        parser.error('--resume requires --skip-dump --npz to preserve the original topology snapshot')
    if args.resume and args.summaries_only:
        parser.error('--resume cannot be combined with --summaries-only')
    if args.degree_budgets is not None and any(
            route not in ('powers', 'endpoints', 'mixed') for route in args.routes):
        parser.error('degree-budget mode supports only powers, endpoints, and mixed routes')
    source = resolve_source(args, parser)
    tag = args.tag or source['label']
    if Path(tag).name != tag:
        parser.error('tag must be a filename component')
    out = Path(args.out).resolve()
    npz = source['npz']

    existing = ([path for path in out.iterdir() if path.resolve() != npz]
                if out.exists() else [])
    if existing and not args.resume:
        raise SystemExit(
            f'{out} is not empty; pick a fresh directory or pass --resume with the original NPZ')

    # Both stages run in their own interpreter, so this process's BLAS pool is
    # not inherited; set the variables anyway so torch does not oversubscribe
    # during the dump and the protocol's worker pool stays single-threaded.
    env = dict(os.environ)
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH']
                                    if env.get('PYTHONPATH') else '')

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f'run_{tag}.log'

    # A manifest next to the results, so a result directory explains itself.
    # It is written even when a stage fails: the repository convention is that
    # an interrupted run leaves records behind and is never reported complete.
    manifest = {
        'source_kind': source['kind'],
        'format': args.format or None, 'name': args.name or None, 'tag': tag,
        'data_root': str(args.data_root), 'max_nodes': args.max_nodes,
        'limit': args.limit, 'topology_npz': str(npz),
        'dims': sorted(set(args.dims)), 'degree_budgets': args.degree_budgets,
        'routes': args.routes,
        'workers': args.workers, 'no_mis': args.no_mis,
        'budget': args.budget, 'sesoi': args.sesoi,
        'atol': args.atol, 'rtol': args.rtol,
        'max_compare_distance': args.max_compare_distance,
        'standardize': args.standardize, 'pair_details': args.pair_details,
        'summaries_only': args.summaries_only,
        'stream_results': args.stream_results,
        'resume': args.resume,
        'permutation_checks': args.permutation_checks,
        'paired_permutations': args.paired_permutations, 'seed': args.seed,
        'dump_python': args.dump_python, 'run_python': args.run_python,
        'note': ('max_nodes drops larger graphs; the protocol numbers describe '
                 'the kept subset, not the full dataset' if args.max_nodes else
                 'no node cap was applied'),
    }

    def write_manifest(status: str) -> None:
        manifest['status'] = status
        (out / f'manifest_{tag}.json').write_text(
            json.dumps(manifest, indent=2) + '\n')

    write_manifest('running')
    try:
        with open(log_path, 'a' if args.resume else 'w') as log:
            if not args.skip_dump:
                dump = build_dump_command(args, npz)
                # The dump prints a summary of what it kept and dropped; run()
                # captures it so it lands in the log as well as on stdout.
                run(dump, 'dump', log, env=env)
            else:
                print(f'[dump] skipped, reusing {npz}', flush=True)

            protocol = build_protocol_command(args, npz, tag)
            run(protocol, 'protocol', log, env=env)
    except BaseException:
        write_manifest('failed')
        raise
    write_manifest('complete')
    print(f'[done] results and manifest in {out}', flush=True)


if __name__ == '__main__':
    main()
