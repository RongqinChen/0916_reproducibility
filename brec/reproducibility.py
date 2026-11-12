"""Compare result records from two completed supplementary BREC pair runs."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from rebuttal_experiments.common import atomic_json, identity_hash


def snapshot(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['identity']['configuration']
    if manifest['status'] != 'execution_finished':
        raise ValueError(f'Incomplete run: {root}')
    if config['determinism']['policy'] != 'brec-determinism-v1':
        raise ValueError('Run does not record the strict BREC policy')
    identity = manifest['identity_sha256']
    if identity != identity_hash(manifest['identity']):
        raise ValueError('Manifest identity checksum mismatch')
    dimensions = [config[key] for key in ('bases', 'variants', 'q_values', 'mean_p_values',
                                          'executed_pair_ids')]
    if any(not values or len(set(values)) != len(values) for values in dimensions):
        raise ValueError('Empty or duplicate configured conditions')
    expected = {}
    grouped = {}
    for basis, variant, q in itertools.product(config['bases'], config['variants'], config['q_values']):
        for readout, p in [('multiset', 1), *(('mean', p) for p in config['mean_p_values'])]:
            stem = f'{basis}_{variant}_q{q}_{readout}_p{p}'
            grouped[stem] = Counter()
            for pid in config['executed_pair_ids']:
                expected[f'{stem}_pair{pid:03d}.json'] = {
                    'basis': basis, 'variant': variant, 'q': q, 'readout': readout,
                    'p': p, 'pair_id': pid, 'order': 2, 'identity_sha256': identity}
    records = {}
    for path in sorted((root / 'records').glob('*.json')):
        record = json.loads(path.read_text())
        if path.name not in expected or any(record.get(key) != value
                                           for key, value in expected[path.name].items()):
            raise ValueError(f'Unexpected condition or record identity: {path.name}')
        if record['status'] not in ('certified_different', 'certified_equal', 'unresolved'):
            raise ValueError(f'Failed or resource-limited condition: {path.name}')
        records[path.name] = {key: value for key, value in record.items() if key != 'elapsed_seconds'}
        grouped[path.name.rsplit('_pair', 1)[0]][record['status']] += 1
    if records.keys() != expected.keys():
        raise ValueError('Result records do not cover the requested BREC conditions')
    summary = json.loads((root / 'summary.json').read_text())
    settings = summary['settings']
    if len(settings) != len(grouped) or {row['setting'] for row in settings} != grouped.keys():
        raise ValueError('Summary does not cover the requested settings')
    for row in settings:
        counts = grouped[row['setting']].copy()
        missing = 400 - sum(counts.values())
        if missing:
            counts['not_executed'] = missing
        if (row['denominator'] != 400 or row['counts'] != dict(counts)
                or row['certified_detection_lower_bound'] != counts['certified_different']
                or row['detection_upper_bound'] != 400 - counts['certified_equal']):
            raise ValueError(f'Summary differs from result records: {row["setting"]}')
    return {'identity': manifest['identity'], 'records': records,
            'summary': summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    first, second, out = args.first.resolve(), args.second.resolve(), args.out.resolve()
    if first == second or first.is_relative_to(second) or second.is_relative_to(first):
        parser.error('Use two independent run directories')
    if out.is_relative_to(first) or out.is_relative_to(second):
        parser.error('Write the report outside both run directories')
    try:
        a, b = snapshot(first), snapshot(second)
        missing = sorted(a['records'].keys() - b['records'].keys())
        extra = sorted(b['records'].keys() - a['records'].keys())
        changed = sorted(key for key in a['records'].keys() & b['records'].keys()
                         if a['records'][key] != b['records'][key])
        metadata_match = a['identity'] == b['identity'] and a['summary'] == b['summary']
        result = {'passed': metadata_match and not (missing or extra or changed),
                  'metadata_match': metadata_match, 'compared_conditions': len(a['records']),
                  'missing_in_second': missing, 'extra_in_second': extra, 'changed_records': changed}
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'passed': False, 'error': str(error)}
    result.update(first=str(first), second=str(second),
                  scope='Supplementary order-2 BREC result records and evidence summaries; excludes elapsed time. Does not compare neural weights or prove relabeling invariance.')
    atomic_json(out, result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
