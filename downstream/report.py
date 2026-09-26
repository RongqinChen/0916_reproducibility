"""Read completed GraphGym runs and report validation-selected, paired effects.

Run directories and completion status must be declared in a manifest. This
collector never starts training, selects by test score, or infers completion
from a stats file. Each output is a new, immutable collection snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from supplementary.common import atomic_json, checksum, write_csv


def read_snapshot(path):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during collection: ' + str(path))
    return data


def stats_rows(data, metric):
    rows = {}
    for line in data.decode('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        epoch = row.get('epoch')
        if type(epoch) is not int or epoch < 0:
            raise ValueError('Each stats row must have a nonnegative integer epoch')
        value = row.get(metric)
        if type(value) not in (float, int) or not np.isfinite(value):
            raise ValueError(f'Missing or nonfinite metric {metric} at epoch {epoch}')
        if epoch in rows and rows[epoch] != row:
            raise ValueError(f'Conflicting duplicate stats at epoch {epoch}')
        rows[epoch] = row
    if not rows:
        raise ValueError('Empty stats file')
    return rows


def final_gnnplus(data, group, run):
    """Read the current fork's final validation-selected table, not a checkpoint."""
    lines = data.decode('utf-8').splitlines()
    header = re.compile(r'FINAL_RESULT \| seed=(\d+) \| split=(\d+) \| best epoch=(\d+)\s*$')
    candidates = []
    for index, line in enumerate(lines):
        match = header.search(line)
        if match is None:
            continue
        seed, split, epoch = map(int, match.groups())
        if seed != run['seed'] or split != group['split_index']:
            raise ValueError('FINAL_RESULT seed/split differs from manifest')
        rows, opened = {}, False
        for line in lines[index + 1:]:
            line = line.strip()
            if line and set(line) == {'-'}:
                if opened:
                    break
                continue
            if line.split() == ['Metric', 'Validation', 'Test']:
                opened = True
                continue
            if not opened:
                continue
            columns = line.split()
            if len(columns) != 3:
                raise ValueError('Malformed FINAL_RESULT table')
            metric, a, b = columns
            a, b = float(a), float(b)
            if not np.isfinite([a, b]).all() or metric in rows:
                raise ValueError('Nonfinite or repeated FINAL_RESULT metric')
            rows[metric] = (a, b)
        else:
            raise ValueError('Truncated FINAL_RESULT table')
        if group['metric'] not in rows or group['selection']['metric'] not in rows:
            raise ValueError('Requested metric absent from FINAL_RESULT; cannot reconstruct it')
        candidates.append({'selected_epoch': epoch,
                           'validation_score': rows[group['selection']['metric']][0],
                           'test_score': rows[group['metric']][1]})
    if not candidates:
        raise ValueError('No FINAL_RESULT; a checkpoint alone does not certify a completed run')
    if any(c != candidates[0] for c in candidates[1:]):
        raise ValueError('Conflicting FINAL_RESULT blocks; provide a log for one completed run')
    return {**candidates[0], 'selection_basis': 'trainer final validation-selected result',
            'source_precision': 'six decimal places printed by the current GNNPlus fork'}


def parameter_count(log_data):
    text = log_data.decode('utf-8', errors='replace')
    for pattern in (r'Num parameters:\s*([\d,]+)',
                    r'Parameters:\s*([\d,]+)'):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1).replace(',', ''))
    return None


def runtime_metadata(log_data):
    text = log_data.decode('utf-8', errors='replace')
    seconds = re.findall(r'RUN_TRAIN_SECONDS=([0-9.eE+-]+)', text)
    memory = re.findall(r'RUN_PEAK_CUDA_ALLOCATED_BYTES=(\d+)', text)
    preprocessing = re.findall(r'RUN_ENCODING_PREPROCESS_SECONDS=([0-9.eE+-]+)', text)
    storage = re.findall(r'RUN_ENCODING_STORAGE_BYTES=(\d+)', text)
    return (float(seconds[-1]) if seconds else None,
            int(memory[-1]) if memory else None,
            float(preprocessing[-1]) if preprocessing else None,
            int(storage[-1]) if storage else None)


def validate_manifest(manifest):
    if manifest.get('schema_version') != 1 or not manifest.get('groups'):
        raise ValueError('Expected schema_version=1 and nonempty groups')
    ids = set()
    for group in manifest['groups']:
        for key in ('id', 'dataset', 'backbone', 'split_id', 'protocol_id', 'metric', 'baseline'):
            if not isinstance(group.get(key), str) or not group[key].strip():
                raise ValueError('Missing group field: ' + key)
        if group['id'] in ids:
            raise ValueError('Duplicate group ID')
        ids.add(group['id'])
        if group.get('direction') not in ('min', 'max'):
            raise ValueError('Test metric direction must be min or max')
        selection = group.get('selection', {})
        if not isinstance(selection.get('metric'), str) or not selection['metric']:
            raise ValueError('Declare a validation selection metric')
        if selection.get('direction') not in ('min', 'max'):
            raise ValueError('Validation direction must be min or max')
        if group.get('format', 'graphgym_jsonl') not in ('graphgym_jsonl', 'gnnplus_final_log'):
            raise ValueError('Unknown result format')
        if group.get('format') == 'gnnplus_final_log' and (
                type(group.get('split_index')) is not int or group['split_index'] < 0):
            raise ValueError('GNNPlus final logs require the actual integer split_index')
        seeds, encodings = group.get('expected_seeds', []), group.get('encodings', [])
        if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError('Declare distinct nonnegative expected_seeds')
        if (not encodings or any(not isinstance(e, str) or not e for e in encodings)
                or len(set(encodings)) != len(encodings) or group['baseline'] not in encodings):
            raise ValueError('Declare distinct encodings including the baseline')
        margin = group.get('equivalence_margin')
        if margin is not None and (type(margin) not in (int, float) or not np.isfinite(margin) or margin <= 0):
            raise ValueError('equivalence_margin must be positive, in test-metric units')
        seen, paths = set(), set()
        for run in group.get('runs', []):
            if type(run.get('seed')) is not int:
                raise ValueError('Run seed must be an integer')
            key = (run.get('encoding'), run['seed'])
            if key[0] not in encodings or key[1] not in seeds or key in seen:
                raise ValueError('Unknown or duplicate encoding/seed condition')
            seen.add(key)
            if run.get('status') not in ('complete', 'running', 'queued', 'failed'):
                raise ValueError('Run status must be complete, running, queued, or failed')
            if run['status'] == 'complete' and not isinstance(run.get('run_dir'), str):
                raise ValueError('Completed run requires run_dir')
            if run.get('run_dir'):
                # Resolve against the manifest directory before this check.
                if run['run_dir'] in paths:
                    raise ValueError('Two conditions point to the same run directory')
                paths.add(run['run_dir'])


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
            'sample_std': float(a.std(ddof=1)) if len(a) >= 2 else None}


def paired_interval(differences, replicates, seed):
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 1000):
        draw = rng.integers(0, len(d), size=(min(1000, replicates - start), len(d)))
        means.extend(d[draw].mean(axis=1).tolist())
    return np.quantile(means, [0.025, 0.975]).tolist()


def paired_t_interval(differences):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return None
    half = float(student_t.ppf(0.975, len(values) - 1) *
                 values.std(ddof=1) / np.sqrt(len(values)))
    mean = float(values.mean())
    return [mean - half, mean + half]


def collect(group, source_root):
    records = []
    declared = {(r['encoding'], r['seed']): r for r in group.get('runs', [])}
    for encoding in group['encodings']:
        for seed in group['expected_seeds']:
            record = {'group_id': group['id'], 'dataset': group['dataset'],
                      'backbone': group['backbone'], 'split_id': group['split_id'],
                      'protocol_id': group['protocol_id'], 'encoding': encoding,
                      'seed': seed, 'metric': group['metric']}
            run = declared.get((encoding, seed))
            if run is None or run['status'] != 'complete':
                record['status'] = 'not_declared' if run is None else run['status']
                records.append(record)
                continue
            record['run_dir'] = run['run_dir']
            try:
                folder = Path(run['run_dir'])
                if group.get('format') == 'gnnplus_final_log':
                    raw = {'final_log': read_snapshot(Path(run.get('log_file', folder / 'logging.log')))}
                    selected = final_gnnplus(raw['final_log'], group, run)
                else:
                    raw = {name: read_snapshot(folder / name / 'stats.json') for name in ('val', 'test')}
                    val = stats_rows(raw['val'], group['selection']['metric'])
                    test = stats_rows(raw['test'], group['metric'])
                    sign = 1 if group['selection']['direction'] == 'min' else -1
                    # Earliest epoch breaks validation ties, independently of test performance.
                    epoch = min(val, key=lambda e: (sign * val[e][group['selection']['metric']], e))
                    if epoch not in test:
                        raise ValueError(f'Test metric absent at validation-selected epoch {epoch}')
                    selected = {'selected_epoch': epoch,
                                'validation_score': float(val[epoch][group['selection']['metric']]),
                                'test_score': float(test[epoch][group['metric']]),
                                'selection_basis': 'reselected from all supplied validation epochs',
                                'source_precision': 'as stored in JSONL'}
                stdout_log = run.get('stdout_log')
                if stdout_log:
                    raw['stdout_log'] = read_snapshot(Path(stdout_log))
                runtime_log = raw.get('stdout_log', raw.get('final_log', b''))
                train_seconds, peak_cuda_bytes, preprocessing_seconds, storage_bytes = runtime_metadata(runtime_log)
                if train_seconds is None and group.get('format') == 'gnnplus_final_log':
                    (train_seconds, peak_cuda_bytes,
                     preprocessing_seconds, storage_bytes) = runtime_metadata(
                         raw.get('final_log', b''))
                counts = [value for value in (
                    parameter_count(raw.get('final_log', b'')),
                    parameter_count(raw.get('stdout_log', b''))) if value is not None]
                if group.get('format') == 'graphgym_jsonl':
                    epoch_count = val[epoch].get('params')
                    if type(epoch_count) in (int, float):
                        counts.append(int(epoch_count))
                selected['parameter_count'] = counts[0] if counts and len(set(counts)) == 1 else None
                selected['train_seconds'] = train_seconds
                selected['peak_cuda_allocated_bytes'] = peak_cuda_bytes
                selected['encoding_preprocess_seconds'] = preprocessing_seconds
                selected['encoding_storage_bytes'] = storage_bytes
                selected['job_wall_seconds'] = run.get('elapsed_seconds')
                per_graph = folder / 'per_graph_losses.csv'
                if per_graph.is_file():
                    raw['per_graph_losses'] = read_snapshot(per_graph)
                if run.get('config_file'):
                    raw['config'] = read_snapshot(Path(run['config_file']))
                archive = source_root / f'run-{len(records):04d}'
                archive.mkdir(parents=True)
                hashes = {}
                for name, data in raw.items():
                    suffix = ('.jsonl' if name in ('val', 'test') else
                              '.csv' if name == 'per_graph_losses' else '.yaml'
                              if name == 'config' else '.txt')
                    destination = archive / (name + suffix)
                    destination.write_bytes(data)
                    hashes[str(destination.relative_to(source_root.parent.parent))] = hashlib.sha256(data).hexdigest()
                record.update(status='complete', **selected,
                              selection_metric=group['selection']['metric'],
                              source_sha256=hashes,
                              completion_basis='operator-declared in manifest')
                if 'per_graph_losses' in raw:
                    record['per_graph_losses_archive'] = str(
                        (archive / 'per_graph_losses.csv').relative_to(source_root.parent.parent))
            except (OSError, ValueError, TypeError, AttributeError, UnicodeError) as exc:
                record.update(status='invalid', error=str(exc))
            records.append(record)
    return records


def report_group(group, records, args):
    scores = {(r['encoding'], r['seed']): r['test_score']
              for r in records if r['status'] == 'complete'}
    parameters = {(r['encoding'], r['seed']): r.get('parameter_count')
                  for r in records if r['status'] == 'complete'}
    by_encoding, paired = [], []
    for encoding in group['encodings']:
        seeds = [s for s in group['expected_seeds'] if (encoding, s) in scores]
        completed_records = [r for r in records if r['encoding'] == encoding
                             and r['status'] == 'complete']
        by_encoding.append({'encoding': encoding, **summarize([scores[encoding, s] for s in seeds]),
                            'completed_seeds': seeds,
                            'missing_seeds': [s for s in group['expected_seeds'] if s not in seeds],
                            'parameter_counts': {str(s): parameters.get((encoding, s)) for s in seeds},
                            'train_seconds': summarize([r['train_seconds'] for r in completed_records
                                                        if r.get('train_seconds') is not None]),
                            'job_wall_seconds': summarize([r['job_wall_seconds'] for r in completed_records
                                                           if r.get('job_wall_seconds') is not None]),
                            'encoding_preprocess_seconds': summarize(
                                [r['encoding_preprocess_seconds'] for r in completed_records
                                 if r.get('encoding_preprocess_seconds') is not None]),
                            'encoding_storage_bytes': {
                                str(r['seed']): r['encoding_storage_bytes']
                                for r in completed_records
                                if r.get('encoding_storage_bytes') is not None},
                            'peak_cuda_allocated_bytes': max(
                                (r['peak_cuda_allocated_bytes'] for r in completed_records
                                 if r.get('peak_cuda_allocated_bytes') is not None), default=None)})
        if encoding == group['baseline']:
            continue
        common = [s for s in seeds if (group['baseline'], s) in scores]
        sign = 1 if group['direction'] == 'max' else -1
        differences = [sign * (scores[encoding, s] - scores[group['baseline'], s]) for s in common]
        paired_counts = {str(s): [parameters.get((group['baseline'], s)),
                                  parameters.get((encoding, s))] for s in common}
        parameter_match = (all(a is not None and a == b for a, b in paired_counts.values())
                           if common else None)
        token = json.dumps([args.seed, group['id'], encoding]).encode()
        stable_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], 'little')
        interval = paired_interval(differences, args.bootstrap, stable_seed)
        margin = group.get('equivalence_margin')
        paired.append({'encoding': encoding, 'baseline': group['baseline'],
                       'matched_seeds': common, 'missing_paired_seeds': [s for s in group['expected_seeds'] if s not in common],
                       'positive_difference_means': 'improvement over baseline',
                       'per_seed_differences': dict(zip(map(str, common), differences)),
                       **summarize(differences), 'paired_bootstrap_95_ci': interval,
                       'paired_student_t_95_ci': paired_t_interval(differences),
                       'bootstrap_replicates': args.bootstrap,
                       'wins': sum(d > 0 for d in differences), 'ties': sum(d == 0 for d in differences),
                       'losses': sum(d < 0 for d in differences), 'equivalence_margin': margin,
                       'parameter_counts_by_seed': paired_counts,
                       'parameter_count_matched': parameter_match,
                       'ci_inside_declared_margin': (None if margin is None or interval is None
                                                     or len(common) != len(group['expected_seeds'])
                                                     else interval[0] >= -margin and interval[1] <= margin),
                       'all_expected_pairs_complete': len(common) == len(group['expected_seeds'])})
    return {'group_id': group['id'], 'dataset': group['dataset'],
            'domain': group.get('domain'), 'degree_budget': group.get('degree_budget'),
            'metric': group['metric'], 'direction': group['direction'],
            'parameter_count_matches_across_all_conditions': (
                all(parameters.get((encoding, seed)) is not None
                    and parameters.get((encoding, seed)) == parameters.get((group['baseline'], seed))
                    for encoding in group['encodings'] for seed in group['expected_seeds'])
                if all((encoding, seed) in scores for encoding in group['encodings']
                       for seed in group['expected_seeds']) else None),
            'by_encoding': by_encoding, 'paired_comparisons': paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260924)
    args = parser.parse_args()
    if args.bootstrap < 1000 or args.seed < 0:
        parser.error('Use at least 1000 bootstrap replicates and a nonnegative seed')
    data = read_snapshot(args.manifest)
    manifest = json.loads(data)
    base = args.manifest.resolve().parent
    for group in manifest.get('groups', []):
        for run in group.get('runs', []):
            for key in ('run_dir', 'config_file', 'log_file'):
                if run.get(key):
                    run[key] = str((base / run[key]).resolve())
            if run.get('stdout_log'):
                run['stdout_log'] = str((base / run['stdout_log']).resolve())
    validate_manifest(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / 'input_manifest.json').write_bytes(data)
    atomic_json(args.out_dir / 'resolved_manifest.json', manifest)
    all_records, summaries = [], []
    for i, group in enumerate(manifest['groups']):
        records = collect(group, args.out_dir / 'sources' / f'group-{i:03d}')
        all_records.extend(records)
        summaries.append(report_group(group, records, args))
    atomic_json(args.out_dir / 'runs.json', all_records)
    write_csv(args.out_dir / 'runs.csv', [{k: v for k, v in r.items() if k != 'source_sha256'} for r in all_records])
    atomic_json(args.out_dir / 'summary.json', summaries)
    atomic_json(args.out_dir / 'collection.json', {
        'protocol': 'downstream_paired_report_v1',
        'status': ('complete' if all(r['status'] == 'complete' for r in all_records)
                   else 'finished_with_missing_measurements'),
        'manifest_sha256': hashlib.sha256(data).hexdigest(),
        'source_sha256': {str(Path(__file__).name): checksum(__file__),
                          'supplementary/common.py': checksum(
                              Path(__file__).resolve().parents[1] / 'supplementary' / 'common.py')},
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': __import__('scipy').__version__,
        'bootstrap_seed': args.seed, 'bootstrap_replicates': args.bootstrap,
        'pairing': 'same declared group, split and training seed; no pooling across groups',
        'completion_and_matching': 'operator declarations, not automatically verified',
        'interval_scope': 'training-seed variation conditional on the declared fixed split',
    })
    print(f'Collected {len(all_records)} conditions into {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
