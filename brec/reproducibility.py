"""Compare result records from two completed supplementary BREC pair runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rebuttal_experiments.common import atomic_json


def snapshot(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['identity']['configuration']
    if manifest['status'] != 'execution_finished':
        raise ValueError(f'Incomplete run: {root}')
    if config['determinism']['policy'] != 'brec-determinism-v1':
        raise ValueError('Run does not record the strict BREC policy')
    records = {}
    for path in sorted((root / 'records').glob('*.json')):
        record = json.loads(path.read_text())
        if record['status'] not in ('certified_different', 'certified_equal', 'unresolved'):
            raise ValueError(f'Failed or resource-limited condition: {path.name}')
        records[path.name] = {key: value for key, value in record.items() if key != 'elapsed_seconds'}
    expected = (len(config['bases']) * len(config['variants']) * len(config['q_values'])
                * (1 + len(config['mean_p_values'])) * len(config['executed_pair_ids']))
    if not records or len(records) != expected:
        raise ValueError('Result records do not cover the requested BREC conditions')
    return {'identity': manifest['identity'], 'records': records,
            'summary': json.loads((root / 'summary.json').read_text())}


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
