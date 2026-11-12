#!/usr/bin/env python3
"""Extract paper-ready aggregate results without copying per-graph metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def compact(value, include_graph_ids=False):
    if isinstance(value, dict):
        return {k: compact(v, include_graph_ids) for k, v in value.items()
                if k not in {'evaluated_graphs', 'timings'}
                and (include_graph_ids or not k.endswith('graph_ids'))}
    if isinstance(value, list):
        return [compact(v, include_graph_ids) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Run directory or results_<tag>.json')
    parser.add_argument('--tag', default='rerun')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-graph-ids', action='store_true',
                        help='retain per-condition eligible graph-ID arrays')
    args = parser.parse_args()
    source = args.source / f'results_{args.tag}.json' if args.source.is_dir() else args.source
    if source.resolve() == args.out.resolve():
        parser.error('Output must differ from input')
    with source.open() as handle:
        data = json.load(handle)
    rows = data.get('rows', data.get('summary'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('No aggregate rows/summary found; use results_<tag>.json')
    if 'paired_comparisons' not in data:
        raise ValueError('Input lacks paired_comparisons; do not infer paired CIs from route means')
    result = compact({k: v for k, v in data.items() if k not in {'rows', 'summary', 'timings', 'run'}},
                     args.include_graph_ids)
    result['summary'] = compact(rows, args.include_graph_ids)
    result['source'] = str(source)
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    result['source_sha256'] = digest.hexdigest()
    timings = data.get('timings', [])
    result['cost'] = data.get('cost', {}) or {}
    for dest, field, operation in [
        ('sum_graph_seconds', 'total_seconds', sum),
        ('orbit_seconds_total', 'orbit_seconds', sum),
        ('max_worker_high_water_rss_bytes', 'worker_peak_rss_bytes', max),
    ]:
        values = [r[field] for r in timings if r.get(field) is not None]
        if values:
            result['cost'][dest] = operation(values)
    if 'elapsed_seconds' in data:
        result['cost']['reported_elapsed_seconds'] = data['elapsed_seconds']
    result['cost']['note'] = 'Worker high-water RSS is not aggregate RAM; resumed elapsed time is not full campaign duration.'
    run_path = source.with_name(source.name.replace('results_', 'run_', 1))
    if run_path != source and run_path.exists():
        with run_path.open() as handle:
            result['run'] = compact(json.load(handle))
    elif 'run' in data:
        result['run'] = compact(data['run'])
    result['extraction_note'] = ('Existing macro means, paired intervals and eligible counts are preserved; '
                                 'no re-averaging of summaries and no replacement of missing values with zero.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f"Saved {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")
    print(f"Graphs: {data.get('graph_count', 'unknown')}; configurations: {len(rows)}")
    print('domain  d  route  RC  valid_RC  unreliable  unresolved')
    max_dim = max(r['branch_dim'] for r in rows)
    for r in rows:
        if r['branch_dim'] == max_dim:
            rc = r.get('representative_capacity_mean')
            value = 'NA' if rc is None else f'{rc:.9f}'
            print(r['items'], r['branch_dim'], r['encoding'], value,
                  r.get('representative_capacity_valid_graphs'),
                  r.get('unreliable_graphs'), r.get('unresolved_graphs'))


if __name__ == '__main__':
    main()
