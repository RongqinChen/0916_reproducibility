"""Graph-level aggregation and paired inference."""

from typing import List, Sequence, Tuple

import numpy as np

PAIRINGS = (('endpoints', 'powers'), ('mixed', 'powers'), ('mixed', 'powers_2d'),
            ('endpoints_2d', 'powers_2d'))
PAIRED_METRICS = ('pairwise_resolution', 'representative_capacity', 'coverage_wl')
SUMMARY_FIELDS = ('mis_ratio', 'orbit_ceiling_ratio', 'mis_orbit_ratio',
                  'separation_rate', 'resolution_rate', 'coverage_orbit',
                  'coverage_wl', 'completeness_orbit', 'completeness_wl',
                  'orbit_violation_rate', 'wl_violation_rate', 'wl_classes',
                  'pairwise_resolution', 'representative_fraction',
                  'representative_capacity', 'wl_orbit_pair_ceiling')


def _describe(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not values.size:
        return 0, None, None
    return values.size, float(values.mean()), float(values.std())


def summarize(records: List[dict]) -> List[dict]:
    powers = {(row['graph_index'], row['items'], row['branch_dim']): row
              for row in records if row['encoding'] == 'powers'}
    groups = {}
    for row in records:
        key = row['items'], row['encoding'], row['branch_dim'], row['channels']
        groups.setdefault(key, []).append(row)
    result = []
    for (kind, route, width, channels), rows in sorted(groups.items()):
        out = {'items': kind, 'encoding': route, 'branch_dim': width,
               'channels': channels, 'graphs': len(rows)}
        reliable = [row for row in rows if row.get('reliable', False)]
        # The collision denominator belongs to powers, not to the new route.
        baseline_rows = [row for row in reliable
                         if row.get('resolved_pairs') is not None
                         and powers.get((row['graph_index'], row['items'], width), {})
                         .get('reliable', False)]
        out['reliable_graphs'] = len(reliable)
        out['reliable_graph_ids'] = sorted(row['graph_index'] for row in reliable)
        out['mean_population'] = 'route-specific reliable graphs; compare routes using paired estimates'
        out['unreliable_graphs'] = len(rows) - len(reliable)
        out['unresolved_graphs'] = sum(row.get('missed_orbit_pairs', 0) > 0
                                       for row in reliable)
        out['unresolved_graph_fraction'] = (out['unresolved_graphs'] / len(reliable)
                                            if reliable else None)
        out['unreliable_graph_ids'] = sorted(
            row['graph_index'] for row in rows if not row.get('reliable', False))
        failure_counts = {}
        for row in rows:
            for failure in row.get('reliability_failures', []):
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
        out['reliability_failure_counts'] = failure_counts
        for field in SUMMARY_FIELDS:
            population = baseline_rows if field == 'resolution_rate' else reliable
            count, mean, std = _describe(row.get(field) for row in population)
            out[field + '_valid_graphs'] = int(count)
            out[field + '_mean'] = mean
            out[field + '_std'] = std
        intermediate = [row for row in reliable
                        if row['orbit_stratum'] == 'intermediate']
        for field in ('coverage_orbit', 'coverage_wl',
                      'completeness_orbit', 'completeness_wl'):
            count, mean, std = _describe(row.get(field) for row in intermediate)
            out[field + '_intermediate_valid_graphs'] = int(count)
            out[field + '_intermediate_mean'] = mean
            out[field + '_intermediate_std'] = std
        out['orbit_stratum_counts'] = {
            stratum: sum(row['orbit_stratum'] == stratum for row in rows)
            for stratum in ('empty', 'degenerate', 'trivial', 'intermediate')}
        for field in ('total_pairs', 'separated_pairs', 'cross_orbit_pairs',
                      'cross_wl_pairs', 'covered_orbit_pairs', 'covered_wl_pairs',
                      'missed_orbit_pairs', 'missed_wl_pairs',
                      'orbit_violation_pairs', 'wl_violation_pairs',
                      'same_orbit_pairs', 'same_wl_pairs', 'seed_collisions'):
            out[field] = sum(row.get(field) or 0 for row in reliable)
        out['resolution_baseline_graphs'] = len(baseline_rows)
        out['resolution_baseline_graph_ids'] = sorted(row['graph_index'] for row in baseline_rows)
        out['resolution_seed_collisions'] = (
            sum(row['seed_collisions'] for row in baseline_rows) if baseline_rows else None)
        out['resolved_pairs'] = (sum(row['resolved_pairs'] for row in baseline_rows)
                                 if baseline_rows else None)
        out['separation_rate_micro'] = _safe_ratio(out['separated_pairs'],
                                                   out['total_pairs'])
        out['coverage_orbit_micro'] = _safe_ratio(out['covered_orbit_pairs'],
                                                  out['cross_orbit_pairs'])
        out['coverage_wl_micro'] = _safe_ratio(out['covered_wl_pairs'],
                                               out['cross_wl_pairs'])
        out['orbit_violation_rate_micro'] = _safe_ratio(
            out['orbit_violation_pairs'], out['same_orbit_pairs'])
        out['wl_violation_rate_micro'] = _safe_ratio(
            out['wl_violation_pairs'], out['same_wl_pairs'])
        out['resolution_rate_micro'] = (_safe_ratio(out['resolved_pairs'],
                                                    out['resolution_seed_collisions'])
                                        if baseline_rows else None)
        exact_representatives = [row for row in reliable
                                 if row.get('representative_count_exact') is True
                                 and row['objects'] > 0]
        bounded_representatives = [row for row in reliable
                                   if row.get('representative_capacity_lower_bound') is not None]
        _, lower_mean, lower_std = _describe(
            row.get('representative_capacity_lower_bound')
            for row in bounded_representatives)
        out['representative_capacity_lower_bound_valid_graphs'] = len(
            bounded_representatives)
        out['representative_capacity_lower_bound_mean'] = lower_mean
        out['representative_capacity_lower_bound_std'] = lower_std
        attained = sum(row.get('representative_ceiling_attained') is True
                       for row in exact_representatives)
        out['representative_ceiling_attained_graphs'] = attained
        out['representative_ceiling_valid_graphs'] = len(exact_representatives)
        out['representative_ceiling_attainment_rate'] = (
            attained / len(exact_representatives) if exact_representatives else None)
        # Compatibility aliases, now based only on reliable exact rows.
        out['saturated_graphs'] = attained
        out['saturation_rate'] = out['representative_ceiling_attainment_rate']
        out['mis_inexact_graphs'] = sum(row['mis_exact'] is False for row in rows)
        out['mis_exceeds_orbits'] = sum(row['mis_exceeds_orbits'] is True
                                        for row in rows)
        out['orbit_violation_graphs'] = sum(row['orbit_violation_pairs'] > 0 for row in rows)
        out['wl_violation_graphs'] = sum(row['wl_violation_pairs'] > 0 for row in rows)
        out['diagnostic_orbit_violation_pairs_all'] = sum(
            row['orbit_violation_pairs'] for row in rows)
        out['diagnostic_wl_violation_pairs_all'] = sum(
            row['wl_violation_pairs'] for row in rows)
        for reference in ('orbit', 'wl'):
            field = reference + '_partition_reproduced'
            eligible = [row for row in reliable if row[field] is not None]
            out[field + '_valid_graphs'] = len(eligible)
            out[field + '_graphs'] = sum(row[field] for row in eligible)
            out[field + '_rate'] = _safe_ratio(out[field + '_graphs'], len(eligible))
        out['permutation_pair_mask_changes'] = sum(
            row.get('permutation_pair_mask_changes', 0) for row in rows)
        out['permutation_checked_graphs'] = sum(
            row.get('permutation_checks', 0) > 0 for row in rows)
        result.append(out)
    return result


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def delta_statistics(deltas, permutations: int, rng,
                     chunk: int = 256) -> Tuple[float, float, float, float]:
    """Mean, sign-flip p-value, and graph-bootstrap 95% interval.

    A single observation cannot support population inference.  Its interval is
    therefore undefined instead of the misleading degenerate [x, x].
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    mean = float(deltas.mean())
    if deltas.size < 2:
        return mean, None, None, None
    exceed, boot_means = 0, np.empty(permutations)
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(deltas)))
        exceed += int(np.count_nonzero(
            np.abs((signs * deltas).mean(axis=1)) >= abs(mean)))
        samples = rng.integers(0, len(deltas), size=(size, len(deltas)))
        boot_means[start:start + size] = deltas[samples].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, (1 + exceed) / (1 + permutations), float(low), float(high)


def _metric_deltas(first, second, metric: str, graphs: Sequence[int]) -> np.ndarray:
    return np.asarray([first[g].get(metric) - second[g].get(metric) for g in graphs
                       if first[g].get(metric) is not None
                       and second[g].get(metric) is not None])


def paired_comparisons(records: List[dict], permutations: int, seed: int,
                       sesoi: float = 0.01) -> List[dict]:
    groups = {}
    for row in records:
        groups.setdefault((row['items'], row['branch_dim'], row['encoding']), {})[
            row['graph_index']] = row
    rng, result = np.random.default_rng(seed), []
    for kind, width in sorted({(kind, width) for kind, width, _ in groups}):
        for first_name, second_name in PAIRINGS:
            first = groups.get((kind, width, first_name), {})
            second = groups.get((kind, width, second_name), {})
            shared_all = sorted(set(first) & set(second))
            shared = [g for g in shared_all
                      if first[g].get('reliable', False)
                      and second[g].get('reliable', False)]
            if not shared:
                continue
            base = {
                'items': kind, 'branch_dim': width,
                'comparison': f'{first_name}_vs_{second_name}',
                'channels': first[shared[0]]['channels'], 'paired_graphs': len(shared),
                'first_channels': first[shared[0]]['channels'],
                'second_channels': second[shared[0]]['channels'],
                'width_matched': first[shared[0]]['channels'] == second[shared[0]]['channels'],
                'candidate_paired_graphs': len(shared_all),
                'excluded_unreliable_graphs': len(shared_all)-len(shared),
            }
            informative = [g for g in shared if any(
                row.get('coverage_orbit') is not None and row['coverage_orbit'] < 1
                for row in (first[g], second[g]))]
            for metric in PAIRED_METRICS:
                metric_graphs = shared
                if metric == 'representative_capacity':
                    metric_graphs = [g for g in shared
                                     if first[g].get('representative_count_exact') is True
                                     and second[g].get('representative_count_exact') is True]
                deltas = _metric_deltas(first, second, metric, metric_graphs)
                if not deltas.size:
                    continue
                metric_first = np.asarray([first[g][metric] for g in metric_graphs
                                           if first[g].get(metric) is not None
                                           and second[g].get(metric) is not None])
                metric_second = np.asarray([second[g][metric] for g in metric_graphs
                                            if first[g].get(metric) is not None
                                            and second[g].get(metric) is not None])
                mean, pvalue, low, high = delta_statistics(deltas, permutations, rng)
                bounded = low is not None and high is not None
                out = {**base, 'metric': metric, 'mean_delta': mean,
                       'metric_graph_ids': [g for g in metric_graphs
                                            if first[g].get(metric) is not None
                                            and second[g].get(metric) is not None],
                       'wins': int(np.sum(metric_first > metric_second)),
                       'ties': int(np.sum(metric_first == metric_second)),
                       'losses': int(np.sum(metric_first < metric_second)),
                       'signflip_p_value': pvalue, 'delta_ci_low': low,
                       'delta_ci_high': high, 'sesoi': sesoi,
                       # Compatibility field retained but accurately named beside it.
                       'detectable_gain': max(abs(low), abs(high)) if bounded else None,
                       'max_abs_ci_bound': max(abs(low), abs(high)) if bounded else None,
                       'equivalent_within_sesoi': (low > -sesoi and high < sesoi)
                                                  if bounded else None,
                       'metric_graphs': int(deltas.size),
                       'informative_graphs': len(informative)}
                metric_graph_set = set(metric_graphs)
                informative_metric_graphs = [g for g in informative if g in metric_graph_set
                                             and first[g].get(metric) is not None
                                             and second[g].get(metric) is not None]
                ideltas = _metric_deltas(first, second, metric, informative_metric_graphs)
                # This subset is selected using observed outcomes, so it is
                # descriptive only; no post-selection interval is reported.
                out.update(informative_metric_graphs=len(informative_metric_graphs),
                           informative_mean_delta=(float(ideltas.mean())
                                                   if ideltas.size else None),
                           informative_delta_ci_low=None,
                           informative_delta_ci_high=None,
                           informative_detectable_gain=None,
                           informative_equivalent_within_sesoi=None)
                result.append(out)
    return result


def summarize_differences(rows: List[dict]) -> List[dict]:
    groups = {}
    for row in rows:
        key = row['items'], row['branch_dim'], row['comparison']
        groups.setdefault(key, []).append(row)
    result = []
    for (kind, width, comparison), items in sorted(groups.items()):
        reliable = [row for row in items if row.get('reliable', False)]
        result.append({
            'items': kind, 'branch_dim': width, 'comparison': comparison,
            'graphs': len(items),
            'reliable_graphs': len(reliable),
            'unreliable_graph_ids': [row['graph_index'] for row in items
                                     if not row.get('reliable', False)],
            'reliable_graph_ids': [row['graph_index'] for row in reliable],
            **{field: sum(row[field] for row in reliable)
               for field in ('first_only', 'second_only', 'shared', 'missed_by_both')},
        })
    return result


def matched_control_degeneracy(records: List[dict]) -> Tuple[int, int]:
    masks = {(row['graph_index'], row['items'], row['branch_dim'], row['encoding']):
             row['covered_orbit_pairs'] for row in records}
    identical = total = 0
    for (graph, kind, width, route), value in masks.items():
        if route == 'powers':
            other = masks.get((graph, kind, width, 'powers_2d'))
            if other is not None:
                total += 1
                identical += int(value == other)
    return identical, total
