#!/usr/bin/env python3
"""Export archived aggregate data to CSV, without running an experiment."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reference_results')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose a new or empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = ('representative_capacity', 'pairwise_resolution',
               'representative_capacity_lower_bound')
    object_rows = []
    for file in sorted((args.source / 'objects').glob('*.json')):
        data = json.loads(file.read_text())
        for row in data['summary']:
            result = dict(dataset=file.stem)
            for key in ('items', 'branch_dim', 'encoding', 'channels', 'graphs',
                        'reliable_graphs', 'unreliable_graphs', 'unresolved_graphs',
                        'mis_inexact_graphs'):
                result[key] = row.get(key)
            for metric in metrics:
                for suffix in ('mean', 'std', 'valid_graphs'):
                    key = f'{metric}_{suffix}'
                    result[key] = row.get(key)
            object_rows.append(result)
    write(args.out / 'objects_all_widths.csv', object_rows)
    write(args.out / 'objects_d14.csv', [row for row in object_rows if row['branch_dim'] == 14])
    ppgn = json.loads((args.source / 'ppgn_summary.json').read_text())
    write(args.out / 'ppgn.csv', [dict(order=row['order'], group=row['reference_group'],
          seed=row['seed'], status='provisional', **row['counts']) for row in ppgn
          if row['counts']['requested']])
    brec = json.loads((args.source / 'brec_monomial.json').read_text())
    for section in ('tables', 'full400_totals', 'readout_comparisons'):
        write(args.out / f'brec_{section}.csv', [
            dict(condition=key, status='provisional', **value)
            for key, value in brec.get(section, {}).items()])
    print(f'Exported aggregate records to {args.out}; no graph computations performed.')


if __name__ == '__main__':
    main()
