"""CLI-independent serialization and concise console reporting."""

import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np

from .features import DOMAINS, channel_plan
from .statistics import matched_control_degeneracy


def write_csv(path: Path, rows, compressed=False) -> None:
    opener = gzip.open if compressed else open
    kwargs = {'mode': 'wt', 'newline': ''} if compressed else {'mode': 'w', 'newline': ''}
    with opener(path, **kwargs) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def format_value(value) -> str:
    return 'N/A' if value is None else f'{value:.6f}'


def print_summary(summary) -> None:
    print(f"{'Objects':<8} {'Encoding':<13} {'d':>3} {'Ch':>4} "
          f"{'Pair resolution':>15} {'Rep. capacity':>15} "
          f"{'Reliable':>10}")
    for row in summary:
        print(f"{row['items']:<8} {row['encoding']:<13} {row['branch_dim']:>3} "
              f"{row['channels']:>4} {format_value(row['pairwise_resolution_mean']):>15} "
              f"{format_value(row['representative_capacity_mean']):>15} "
              f"{row['reliable_graphs']:>4}/{row['graphs']:<5}")


def print_diagnostics(rows, summary) -> None:
    orbit_violations = sum(row['orbit_violation_pairs'] for row in rows)
    wl_violations = sum(row['wl_violation_pairs'] for row in rows)
    permutation_changes = sum(row['permutation_pair_mask_changes'] for row in rows)
    monotonicity = sum(row['monotonicity_violations'] or 0 for row in rows)
    print(f'[reliability] same-orbit separated pairs: {orbit_violations}')
    print(f'[reliability] same-WL separated pairs: {wl_violations}')
    print(f'[reliability] relabelled pair-mask changes: {permutation_changes}')
    print(f'[reliability] augmentation monotonicity violations: {monotonicity}')
    print('[capacity] budget-limited MIS rows (lower bounds):',
          sum(row['mis_exact'] is False for row in rows))
    print('[reliability] rows excluded from primary summaries:',
          sum(not row.get('reliable', False) for row in rows))
    print('[reliability] rows with relabelling checks:',
          sum(row.get('permutation_checks', 0) > 0 for row in rows), '/', len(rows))
    identical, total = matched_control_degeneracy(rows)
    print(f'[diagnostics] powers and powers_2d cover equal counts in {identical}/{total} '
          'graph-conditions')


def print_paired(paired) -> None:
    if not paired:
        return
    print(f"\n{'Objects':<8} {'d':>3} {'Comparison':<28} {'Metric':<20} "
          f"{'Mean':>10} {'95% CI':>23} {'W/T/L':>12}")
    for row in paired:
        low, high = row['delta_ci_low'], row['delta_ci_high']
        interval = ('N/A' if low is None else f'[{low:+.4f}, {high:+.4f}]')
        outcome = f"{row['wins']}/{row['ties']}/{row['losses']}"
        print(f"{row['items']:<8} {row['branch_dim']:>3} {row['comparison']:<28} "
              f"{row['metric']:<20} {format_value(row['mean_delta']):>10} "
              f"{interval:>23} {outcome:>12}")


def build_payload(config, summary, paired, difference_summary, timings, started):
    plans = {
        kind: {
            str(width): {
                'power_degrees': channel_plan(kind, width)[0],
                'power_control_degrees': channel_plan(kind, 2 * width)[0],
                'endpoint_degrees': channel_plan(kind, width)[1],
                'endpoint_2d_degrees': channel_plan(kind, 2 * width)[1],
                'endpoint_Q': 1 + max(sum(pair) for pair in channel_plan(kind, width)[1]),
            }
            for width in config['dims']
        }
        for kind in DOMAINS
    }
    return {
        'protocol': 'pair_resolution_representative_capacity_v13_float32',
        'feature_dtype': 'float32',
        'statistics_dtype': 'float64',
        'config': config,
        'edge_domain': 'unordered_upper_triangle_u_lt_v',
        'pair_domain': 'unordered_distinct_u_lt_v',
        'orientation_protocol': 'strict_unordered_two_vector_bijection',
        'orientation_similarity': '(ff and rr) or (fr and rf)',
        'width_unit': 'selected matrix channels per orientation, not independent scalar dimension',
        'theoretical_target': 'cross-automorphism-orbit object pairs',
        'coverage_definition': '|separated intersect cross_class| / |cross_class|',
        'primary_metrics': {
            'pairwise_resolution': 'coverage_orbit on reliable rows',
            'representative_capacity': 'MIS/orbits on reliable rows with exact MIS',
            'orbit_partition_reproduced': 'strict partition attainment on reliable rows',
        },
        'reliability_gate': ('no same-orbit separation; no same-WL separation; '
                             'no relabelled mask change; no applicable augmentation '
                             'monotonicity violation'),
        'baseline_gain_gate': 'both the augmented route and powers baseline must be reliable',
        'wl_reference': {
            'algorithm': '2-FWL (folklore 2-WL)',
            'signature': '(c[i,j], sorted_w (c[w,j], c[i,w]))',
            'role': 'secondary reference and within-class numerical reliability check',
        },
        'channel_plans': plans,
        'graph_count': len(timings),
        'rows': summary,
        'route_differences': difference_summary,
        'paired_comparisons': paired,
        'elapsed_seconds': time.perf_counter() - started,
        'orbit_seconds_total': float(sum(row['orbit_seconds'] for row in timings)),
        'timings': sorted(timings, key=lambda row: row['graph_index']),
    }
