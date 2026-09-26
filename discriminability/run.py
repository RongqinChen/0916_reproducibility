#!/usr/bin/env python
"""Measure within-graph pairwise resolution and representative capacity.

The public CLI remains argument-compatible with v8. The implementation lives in the
``discriminability`` package so each protocol layer can be tested independently.
"""

import argparse
import csv
import gzip
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from discriminability import *  # noqa: F401,F403 - compatibility re-exports
from discriminability.cli import add_protocol_arguments, validate_protocol_args
from discriminability.pipeline import (init_worker, load_graphs, load_npz, load_zinc,
                                        topology_provenance, worker)
from discriminability.reporting import (build_payload, print_diagnostics,
                                         print_paired, print_summary, write_csv)
from discriminability.stream_stats import (DIFFERENCE_FIELDS, ROW_FIELDS,
                                           finalize_stream)
from discriminability.statistics import summarize_differences
from discriminability.scheduling import scheduled_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_protocol_arguments(parser, include_data=True)
    return parser


def validate(args, parser: argparse.ArgumentParser) -> None:
    validate_protocol_args(args, parser)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(args, parser)
    try:
        import pynauty  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f'The orbit reference needs pynauty ({exc}); '
                         'install it with: pip install pynauty') from exc

    tag = args.tag or (Path(args.root).stem if args.dataset == 'npz'
                       else ('full' if args.full else 'subset'))
    if Path(tag).name != tag:
        parser.error('tag must be a filename component')
    config = vars(args).copy()
    config['feature_dtype'] = 'float32'
    config['dims'] = sorted(set(args.dims))
    if args.degree_budgets is not None:
        config['degree_budgets'] = sorted(set(args.degree_budgets))
    def checksum(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    source_root = Path(__file__).resolve().parent
    config['source_sha256'] = {str(path.relative_to(source_root)): checksum(path)
                              for path in sorted(source_root.glob('*.py'))}
    config['runtime'] = {'python': platform.python_version()}
    for package in ('numpy', 'pynauty', 'torch', 'torch-geometric'):
        try:
            config['runtime'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config['runtime'][package] = None
    stream_path = None
    stream_continuation = False
    if args.out:
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        # Held until process exit, including resume; never run two writers.
        run_lock = (output / f'.lock_{tag}').open('a')
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another process is already using this output tag')
        if args.stream_results:
            stream_path = output / f'stream_{tag}.jsonl'
        wrapper_files = {f'manifest_{tag}.json', f'run_{tag}.log'}
        existing = [path for path in output.glob(f'*_{tag}.*')
                    if path.name not in wrapper_files]
        stream_outputs = {
            f'stream_{tag}.jsonl', f'run_{tag}.json', f'identity_{tag}.json',
            f'results_{tag}.json', f'results_{tag}.csv', f'paired_{tag}.csv',
            f'per_graph_{tag}.csv', f'route_differences_{tag}.csv',
            f'degree_budget_{tag}.csv',
        }
        stream_continuation = bool(
            args.stream_results and stream_path.exists() and
            all(path.name in stream_outputs or path.name.startswith('.')
                for path in existing))
        if existing and not args.resume and not stream_continuation:
            parser.error('Outputs already exist for this tag; choose a new directory or tag')
        # Exclusive reservation also prevents concurrent writers using one tag.
        if not args.resume and not stream_continuation:
            with (output / f'run_{tag}.json').open('x') as reservation:
                json.dump({'status': 'started', 'config': config}, reservation, indent=2)
        elif stream_continuation:
            (output / f'run_{tag}.json').write_text(
                json.dumps({'status': 'continuing', 'config': config}, indent=2))
    if args.dataset == 'npz':
        config['input_file_sha256'] = checksum(args.root)
        provenance = topology_provenance(args.root)
        if provenance:
            config['topology_source'] = provenance

    rows, differences, timings = [], [], []
    started = time.perf_counter()

    def collect(results, detail_handle=None, stream_handle=None):
        detail_writer = None
        for graph_rows, graph_differences, graph_details, timing in results:
            if stream_handle is not None:
                clean_rows = [{key: value for key, value in row.items()
                               if key in ROW_FIELDS} for row in graph_rows]
                clean_differences = [
                    {key: value for key, value in row.items()
                     if key in DIFFERENCE_FIELDS} for row in graph_differences]
                graph_index = (graph_rows[0]['graph_index'] if graph_rows
                               else graph_differences[0]['graph_index'])
                split = (graph_rows[0].get('split') if graph_rows
                         else graph_differences[0].get('split'))
                stream_handle.write(json.dumps({
                    'graph_index': graph_index, 'split': split,
                    'rows': clean_rows,
                    'differences': clean_differences,
                    'timing': timing,
                }) + '\n')
            else:
                rows.extend(graph_rows)
                differences.extend(graph_differences)
            if graph_details and detail_handle:
                if detail_writer is None:
                    detail_writer = csv.DictWriter(detail_handle, fieldnames=list(graph_details[0]))
                    detail_writer.writeheader()
                detail_writer.writerows(graph_details)
            timings.append(timing)
            if len(timings) % 200 == 0:
                print(f'[run] {len(timings)} graphs', flush=True)

    topology_digest = hashlib.sha256()
    evaluated_graphs = []
    def audited_tasks():
        for index, split, edges, n in load_graphs(args.dataset, args.root, args.full, args.limit):
            graph_digest = hashlib.sha256()
            graph_digest.update(str(n).encode('ascii') + b'\0')
            graph_digest.update(edges.astype('<i8', copy=False).tobytes())
            identity = dict(graph_index=index, split=split, nodes=n,
                            topology_sha256=graph_digest.hexdigest())
            topology_digest.update(json.dumps(identity, sort_keys=True).encode() + b'\n')
            evaluated_graphs.append(identity)
            yield index, split, edges, n
    tasks = list(audited_tasks())
    identity_config = {k: v for k, v in config.items()
                       if k not in ('resume', 'workers', 'memory_gb', 'out')}
    identity = {'config': identity_config, 'graphs': evaluated_graphs}
    completed_graphs = set()
    if args.stream_results:
        identity_path = output / f'identity_{tag}.json'
        continuation = args.resume or stream_continuation
        if continuation:
            if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
                parser.error('Stream resume identity mismatch: input, code, or scientific configuration changed')
        else:
            identity_path.write_text(json.dumps(identity, indent=2))
    if stream_path is not None and stream_path.exists():
        with stream_path.open() as stream_in:
            for line in stream_in:
                try:
                    completed_graphs.add(json.loads(line)['graph_index'])
                except (json.JSONDecodeError, KeyError, TypeError):
                    parser.error('Stream result file is malformed')
    checkpoint_dir = None
    if args.out and not args.summaries_only and not args.stream_results:
        checkpoint_dir = output / f'checkpoints_{tag}'
        identity_path = output / f'identity_{tag}.json'
        if args.resume:
            if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
                parser.error('Resume identity mismatch: input, code, or scientific configuration changed')
        else:
            identity_path.write_text(json.dumps(identity, indent=2))
    if args.stream_results:
        tasks = [task for task in tasks if task[0] not in completed_graphs]
    results = scheduled_results(tasks, config, checkpoint_dir)
    if args.stream_results:
        with stream_path.open('a') as stream_handle:
            collect(results, stream_handle=stream_handle)
    elif args.out and args.pair_details != 'none':
        with gzip.open(output / f'pair_differences_{tag}.csv.gz', 'wt', newline='') as handle:
            collect(results, handle)
    else:
        collect(results)

    if args.stream_results:
        summary, difference_summary, paired, diagnostics, identical, total, \
            timings = finalize_stream(stream_path, args.paired_permutations,
                                      args.seed, args.sesoi)
    else:
        rows.sort(key=lambda row: (row['graph_index'], row['items'], row['channels'],
                                   row['encoding']))
        differences.sort(key=lambda row: (row['graph_index'], row['items'],
                                          row['branch_dim'], row['comparison']))
        summary = summarize(rows)
        difference_summary = summarize_differences(differences)
        paired = paired_comparisons(rows, args.paired_permutations,
                                    args.seed, args.sesoi)

    if args.degree_budgets is not None:
        for row in summary:
            row['degree_budget'] = row['branch_dim']
        summary.sort(key=lambda row: (row['items'], row['degree_budget']))
    config['evaluated_graphs'] = evaluated_graphs
    config['evaluated_topology_sha256'] = topology_digest.hexdigest()

    print_summary(summary)
    if args.stream_results:
        print(f"[reliability] same-orbit separated pairs: "
              f"{diagnostics['orbit_violations']}")
        print(f"[reliability] same-WL separated pairs: "
              f"{diagnostics['wl_violations']}")
        print(f"[reliability] relabelled pair-mask changes: "
              f"{diagnostics['permutation_changes']}")
        print(f"[reliability] augmentation monotonicity violations: "
              f"{diagnostics['monotonicity']}")
        print(f"[capacity] budget-limited MIS rows (lower bounds): "
              f"{diagnostics['budget_limited']}")
        print(f"[reliability] rows excluded from primary summaries: "
              f"{diagnostics['unreliable']}")
        print(f"[reliability] rows with relabelling checks: "
              f"{diagnostics['checked']} / {diagnostics['rows']}")
        print(f"[diagnostics] powers and powers_2d cover equal counts in "
              f"{identical}/{total} graph-conditions")
    else:
        print_diagnostics(rows, summary)
    if timings:
        orbit_seconds = sum(row['orbit_seconds'] for row in timings)
        print(f'[timing] orbit reference: {orbit_seconds:.3f}s total, '
              f'{1000 * orbit_seconds / len(timings):.1f} ms/graph')
    print_paired(paired)

    if args.out:
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        payload = build_payload(config, summary, paired, difference_summary,
                                timings, started)
        (output / f'results_{tag}.json').write_text(
            json.dumps(payload, indent=2, allow_nan=False) + '\n')
        write_csv(output / f'results_{tag}.csv', summary)
        if args.degree_budgets is not None:
            budget_rows = [row for row in summary if row['encoding'] == 'powers']
            write_csv(output / f'degree_budget_{tag}.csv', budget_rows)
        write_csv(output / f'paired_{tag}.csv', paired)
        if not args.summaries_only and not args.stream_results:
            write_csv(output / f'per_graph_{tag}.csv', rows)
            write_csv(output / f'route_differences_{tag}.csv', differences)
        elif args.stream_results:
            # Keep graph-level values available for paper tables and paired
            # association analyses without holding the full dataset in RAM.
            def export_stream(field, destination, columns):
                with destination.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns,
                                            extrasaction='ignore')
                    writer.writeheader()
                    with stream_path.open() as source:
                        for line in source:
                            record = json.loads(line)
                            for row in record[field]:
                                writer.writerow({**row, 'split': record.get('split')})
            export_stream('rows', output / f'per_graph_{tag}.csv',
                          ['graph_index', 'split', *sorted(ROW_FIELDS)])
            export_stream('differences', output / f'route_differences_{tag}.csv',
                          ['graph_index', 'split', *sorted(DIFFERENCE_FIELDS)])
        (output / f'run_{tag}.json').write_text(json.dumps(
            {'status': 'complete', 'config': config}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
