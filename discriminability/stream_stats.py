"""Memory-bounded final aggregation for streamed per-graph results.

The non-stream protocol keeps every row in memory before summarising.  When
``--stream-results`` is used, rows are appended to disk graph-by-graph and the
final aggregation below consumes them one graph at a time, so peak memory is
independent of the number of graphs.
"""
import json

import numpy as np

from .statistics import PAIRINGS, PAIRED_METRICS, _safe_ratio, delta_statistics


SUMMARY_FIELDS = ('mis_ratio', 'orbit_ceiling_ratio', 'mis_orbit_ratio',
                  'separation_rate', 'resolution_rate', 'coverage_orbit',
                  'coverage_wl', 'completeness_orbit', 'completeness_wl',
                  'orbit_violation_rate', 'wl_violation_rate', 'wl_classes',
                  'pairwise_resolution', 'representative_fraction',
                  'representative_capacity', 'wl_orbit_pair_ceiling', 'missed_orbit_pairs')
INTERMEDIATE_FIELDS = ('coverage_orbit', 'coverage_wl',
                       'completeness_orbit', 'completeness_wl')
MICRO_FIELDS = ('total_pairs', 'separated_pairs', 'cross_orbit_pairs',
                'cross_wl_pairs', 'covered_orbit_pairs', 'covered_wl_pairs',
                'missed_orbit_pairs', 'missed_wl_pairs',
                'orbit_violation_pairs', 'wl_violation_pairs',
                'same_orbit_pairs', 'same_wl_pairs', 'seed_collisions')

ROW_FIELDS = frozenset({
    'graph_index', 'items', 'encoding', 'branch_dim', 'degree_budget', 'channels',
    'max_compare_distance', 'comparison_gate', 'eligible_comparison_pairs',
    'orbit_stratum', 'reliable', 'reliability_failures',
    'mis_exact', 'mis_exceeds_orbits', 'orbit_violation_pairs',
    'wl_violation_pairs', 'permutation_pair_mask_changes',
    'permutation_checks', 'monotonicity_violations',
    *MICRO_FIELDS,
    'mis_ratio', 'orbit_ceiling_ratio', 'mis_orbit_ratio', 'separation_rate',
    'resolution_rate', 'coverage_orbit', 'coverage_wl', 'completeness_orbit',
    'completeness_wl', 'orbit_violation_rate', 'wl_violation_rate',
    'wl_classes', 'pairwise_resolution', 'representative_fraction',
    'representative_capacity', 'wl_orbit_pair_ceiling', 'resolved_pairs',
    'objects', 'representative_count_exact', 'representative_ceiling_attained',
    'representative_capacity_lower_bound', 'orbit_partition_reproduced',
    'wl_partition_reproduced',
})
DIFFERENCE_FIELDS = frozenset({
    'items', 'branch_dim', 'comparison', 'graph_index', 'reliable',
    'first_only', 'second_only', 'shared', 'missed_by_both',
})


def _update_stats(stats, value):
    if value is None:
        return
    count, total, squares = stats
    stats[0] = count + 1
    stats[1] = total + float(value)
    stats[2] = squares + float(value) * float(value)


def _finalise_stats(stats):
    count, total, squares = stats
    if count == 0:
        return 0, None, None
    mean = total / count
    variance = max(0.0, squares / count - mean * mean)
    return count, mean, float(np.sqrt(variance))


def _group_summary(group):
    out = {
        'items': group['kind'], 'encoding': group['route'],
        'branch_dim': group['width'], 'channels': group['channels'],
        'graphs': group['graphs'],
        'reliable_graphs': group['reliable_graphs'],
        'reliable_graph_ids': sorted(group['reliable_ids']),
        'mean_population': 'route-specific reliable graphs; compare routes using paired estimates',
        'unreliable_graphs': group['unreliable_graphs'],
        'unresolved_graphs': group['unresolved_graphs'],
        'unresolved_graph_fraction': (group['unresolved_graphs'] / group['reliable_graphs']
                                      if group['reliable_graphs'] else None),
        'unreliable_graph_ids': sorted(group['unreliable_ids']),
        'reliability_failure_counts': group['failure_counts'],
    }
    for field in SUMMARY_FIELDS:
        count, mean, std = _finalise_stats(group['field_stats'][field])
        out[field + '_valid_graphs'] = count
        out[field + '_mean'] = mean
        out[field + '_std'] = std
    for field in INTERMEDIATE_FIELDS:
        count, mean, std = _finalise_stats(group['intermediate_stats'][field])
        out[field + '_intermediate_valid_graphs'] = count
        out[field + '_intermediate_mean'] = mean
        out[field + '_intermediate_std'] = std
    out['orbit_stratum_counts'] = group['stratum_counts']
    for field in MICRO_FIELDS:
        out[field] = group['micro'][field]
    out['resolution_baseline_graphs'] = len(group['resolution_baseline_ids'])
    out['resolution_baseline_graph_ids'] = sorted(group['resolution_baseline_ids'])
    out['resolution_seed_collisions'] = (
        group['resolution_seed_collisions']
        if group['resolution_baseline_ids'] else None)
    out['resolved_pairs'] = (group['resolved_pairs']
                             if group['resolution_baseline_ids'] else None)
    out['separation_rate_micro'] = _safe_ratio(
        group['micro']['separated_pairs'], group['micro']['total_pairs'])
    out['coverage_orbit_micro'] = _safe_ratio(
        group['micro']['covered_orbit_pairs'], group['micro']['cross_orbit_pairs'])
    out['coverage_wl_micro'] = _safe_ratio(
        group['micro']['covered_wl_pairs'], group['micro']['cross_wl_pairs'])
    out['orbit_violation_rate_micro'] = _safe_ratio(
        group['micro']['orbit_violation_pairs'], group['micro']['same_orbit_pairs'])
    out['wl_violation_rate_micro'] = _safe_ratio(
        group['micro']['wl_violation_pairs'], group['micro']['same_wl_pairs'])
    out['resolution_rate_micro'] = (
        _safe_ratio(group['resolved_pairs'], group['resolution_seed_collisions'])
        if group['resolution_baseline_ids'] else None)
    exact = group['exact_representative_count']
    out['representative_capacity_lower_bound_valid_graphs'] = \
        group['bounded_representative_count']
    out['representative_capacity_lower_bound_mean'] = \
        group['bounded_representative_total'] / group['bounded_representative_count'] \
        if group['bounded_representative_count'] else None
    out['representative_capacity_lower_bound_std'] = (
        float(np.sqrt(max(0.0, group['bounded_representative_squares'] /
                          group['bounded_representative_count'] -
                          (group['bounded_representative_total'] /
                           group['bounded_representative_count']) ** 2)))
        if group['bounded_representative_count'] else None)
    out['representative_ceiling_attained_graphs'] = group['attained_count']
    out['representative_ceiling_valid_graphs'] = exact
    out['representative_ceiling_attainment_rate'] = (
        group['attained_count'] / exact if exact else None)
    out['saturated_graphs'] = group['attained_count']
    out['saturation_rate'] = out['representative_ceiling_attainment_rate']
    out['mis_inexact_graphs'] = group['mis_inexact']
    out['mis_exceeds_orbits'] = group['mis_exceeds_orbits']
    out['orbit_violation_graphs'] = group['orbit_violation_graphs']
    out['wl_violation_graphs'] = group['wl_violation_graphs']
    out['diagnostic_orbit_violation_pairs_all'] = group['orbit_violation_all']
    out['diagnostic_wl_violation_pairs_all'] = group['wl_violation_all']
    for reference in ('orbit', 'wl'):
        field = reference + '_partition_reproduced'
        valid = group[field + '_count']
        out[field + '_valid_graphs'] = valid
        out[field + '_graphs'] = group[field + '_sum']
        out[field + '_rate'] = _safe_ratio(group[field + '_sum'], valid)
    out['permutation_pair_mask_changes'] = group['permutation_changes']
    out['permutation_checked_graphs'] = group['permutation_checked']
    return out


def _new_group(kind, route, width, channels):
    return {
        'kind': kind, 'route': route, 'width': width, 'channels': channels,
        'graphs': 0, 'reliable_graphs': 0, 'reliable_ids': [],
        'unresolved_graphs': 0,
        'unreliable_graphs': 0, 'unreliable_ids': [], 'failure_counts': {},
        'field_stats': {field: [0, 0.0, 0.0] for field in SUMMARY_FIELDS},
        'intermediate_stats': {field: [0, 0.0, 0.0]
                               for field in INTERMEDIATE_FIELDS},
        'stratum_counts': {stratum: 0 for stratum in
                           ('empty', 'degenerate', 'trivial', 'intermediate')},
        'micro': {field: 0 for field in MICRO_FIELDS},
        'resolution_baseline_ids': [], 'resolution_seed_collisions': 0,
        'resolved_pairs': 0, 'bounded_representative_count': 0,
        'bounded_representative_total': 0.0,
        'bounded_representative_squares': 0.0,
        'exact_representative_count': 0, 'attained_count': 0,
        'mis_inexact': 0, 'mis_exceeds_orbits': 0,
        'orbit_violation_graphs': 0, 'wl_violation_graphs': 0,
        'orbit_violation_all': 0, 'wl_violation_all': 0,
        'orbit_partition_reproduced_count': 0,
        'orbit_partition_reproduced_sum': 0,
        'wl_partition_reproduced_count': 0,
        'wl_partition_reproduced_sum': 0,
        'permutation_changes': 0, 'permutation_checked': 0,
    }


def summarize_stream(record_iter):
    """Equivalent of ``statistics.summarize`` over a stream of graph records."""
    groups = {}
    powers_reliable = set()
    for rows, differences, timing in record_iter:
        graph_index = rows[0]['graph_index'] if rows else None
        graph_powers = {}
        for row in rows:
            key = (row['items'], row['encoding'], row['branch_dim'],
                   row['channels'])
            group = groups.setdefault(key, _new_group(*key))
            group['graphs'] += 1
            group['stratum_counts'][row['orbit_stratum']] += 1
            if row['encoding'] == 'powers':
                graph_powers[(row['items'], row['branch_dim'])] = \
                    bool(row.get('reliable', False))
            if row.get('reliable', False):
                group['reliable_graphs'] += 1
                group['reliable_ids'].append(graph_index)
                group['unresolved_graphs'] += int(row.get('missed_orbit_pairs', 0) > 0)
            else:
                group['unreliable_graphs'] += 1
                group['unreliable_ids'].append(graph_index)
            for failure in row.get('reliability_failures', []):
                group['failure_counts'][failure] = \
                    group['failure_counts'].get(failure, 0) + 1
            if row.get('mis_exact') is False:
                group['mis_inexact'] += 1
            if row.get('mis_exceeds_orbits') is True:
                group['mis_exceeds_orbits'] += 1
            if row['orbit_violation_pairs'] > 0:
                group['orbit_violation_graphs'] += 1
            if row['wl_violation_pairs'] > 0:
                group['wl_violation_graphs'] += 1
            group['orbit_violation_all'] += row['orbit_violation_pairs']
            group['wl_violation_all'] += row['wl_violation_pairs']
            group['permutation_changes'] += row.get('permutation_pair_mask_changes', 0)
            group['permutation_checked'] += int(row.get('permutation_checks', 0) > 0)
            for field in MICRO_FIELDS:
                group['micro'][field] += row.get(field) or 0
            if row.get('reliable', False):
                for field in SUMMARY_FIELDS:
                    _update_stats(group['field_stats'][field], row.get(field))
                if row['orbit_stratum'] == 'intermediate':
                    for field in INTERMEDIATE_FIELDS:
                        _update_stats(group['intermediate_stats'][field],
                                      row.get(field))
            baseline_key = (graph_index, row['items'], row['branch_dim'])
            is_baseline_row = (
                row.get('resolved_pairs') is not None
                and graph_powers.get((row['items'], row['branch_dim']), False))
            if is_baseline_row and row.get('reliable', False):
                group['resolution_baseline_ids'].append(graph_index)
                group['resolution_seed_collisions'] += row['seed_collisions'] or 0
                group['resolved_pairs'] += row['resolved_pairs'] or 0
            if row.get('representative_count_exact') is True and row['objects'] > 0:
                group['exact_representative_count'] += 1
                if row.get('representative_ceiling_attained') is True:
                    group['attained_count'] += 1
            if row.get('representative_capacity_lower_bound') is not None \
                    and row.get('reliable', False):
                value = row['representative_capacity_lower_bound']
                group['bounded_representative_count'] += 1
                group['bounded_representative_total'] += value
                group['bounded_representative_squares'] += value * value
            for reference in ('orbit', 'wl'):
                field = reference + '_partition_reproduced'
                if row.get(field) is not None and row.get('reliable', False):
                    group[field + '_count'] += 1
                    group[field + '_sum'] += int(row[field])
        for key, reliable in graph_powers.items():
            if reliable and graph_index is not None:
                powers_reliable.add((graph_index, *key))
    return [_group_summary(group) for group in sorted(groups.values(),
                                                      key=lambda g: (
                                                          g['kind'], g['route'],
                                                          g['width']))]


def summarize_differences_stream(record_iter):
    groups = {}
    for rows, differences, timing in record_iter:
        for row in differences:
            key = (row['items'], row['branch_dim'], row['comparison'])
            group = groups.setdefault(key, {
                'items': key[0], 'branch_dim': key[1], 'comparison': key[2],
                'graphs': 0, 'reliable_graphs': 0, 'reliable_ids': [],
                'unreliable_ids': [],
                'first_only': 0, 'second_only': 0, 'shared': 0,
                'missed_by_both': 0})
            group['graphs'] += 1
            if row.get('reliable', False):
                group['reliable_graphs'] += 1
                group['reliable_ids'].append(row['graph_index'])
                for field in ('first_only', 'second_only', 'shared',
                              'missed_by_both'):
                    group[field] += row[field]
            else:
                group['unreliable_ids'].append(row['graph_index'])
    return [{'items': g['items'], 'branch_dim': g['branch_dim'],
             'comparison': g['comparison'], 'graphs': g['graphs'],
             'reliable_graphs': g['reliable_graphs'],
             'unreliable_graph_ids': sorted(g['unreliable_ids']),
             'reliable_graph_ids': sorted(g['reliable_ids']),
             **{field: g[field] for field in
                ('first_only', 'second_only', 'shared', 'missed_by_both')}}
            for g in sorted(groups.values(),
                            key=lambda g: (g['items'], g['branch_dim'],
                                           g['comparison']))]


def _compact_record(row):
    return {
        'channels': row['channels'],
        'reliable': row.get('reliable', False),
        'coverage_orbit': row.get('coverage_orbit'),
        'coverage_wl': row.get('coverage_wl'),
        'pairwise_resolution': row.get('pairwise_resolution'),
        'representative_count_exact': row.get('representative_count_exact'),
        'representative_capacity': row.get('representative_capacity'),
    }


def paired_comparisons_stream(record_iter, permutations, seed, sesoi=0.01):
    groups = {}
    for rows, differences, timing in record_iter:
        for row in rows:
            groups.setdefault((row['items'], row['branch_dim'], row['encoding']),
                              {})[row['graph_index']] = _compact_record(row)
    rng, result = np.random.default_rng(seed), []
    for kind, width in sorted({(kind, width) for kind, width, _ in groups}):
        for first_name, second_name in PAIRINGS:
            first = groups.get((kind, width, first_name), {})
            second = groups.get((kind, width, second_name), {})
            shared_all = sorted(set(first) & set(second))
            shared = [g for g in shared_all
                      if first[g]['reliable'] and second[g]['reliable']]
            if not shared:
                continue
            base = {'items': kind, 'branch_dim': width,
                    'comparison': f'{first_name}_vs_{second_name}',
                    'channels': first[shared[0]]['channels'],
                    'paired_graphs': len(shared),
                    'first_channels': first[shared[0]]['channels'],
                    'second_channels': second[shared[0]]['channels'],
                    'width_matched': first[shared[0]]['channels'] == second[shared[0]]['channels'],
                    'candidate_paired_graphs': len(shared_all),
                    'excluded_unreliable_graphs': len(shared_all) - len(shared)}
            informative = [g for g in shared if any(
                row['coverage_orbit'] is not None and row['coverage_orbit'] < 1
                for row in (first[g], second[g]))]
            for metric in PAIRED_METRICS:
                metric_graphs = shared
                if metric == 'representative_capacity':
                    metric_graphs = [g for g in shared
                                     if first[g]['representative_count_exact'] is True
                                     and second[g]['representative_count_exact'] is True]
                deltas = np.asarray([first[g][metric] - second[g][metric]
                                     for g in metric_graphs
                                     if first[g][metric] is not None
                                     and second[g][metric] is not None])
                if not deltas.size:
                    continue
                metric_first = np.asarray([first[g][metric] for g in metric_graphs
                                           if first[g][metric] is not None
                                           and second[g][metric] is not None])
                metric_second = np.asarray([second[g][metric] for g in metric_graphs
                                            if first[g][metric] is not None
                                            and second[g][metric] is not None])
                mean, pvalue, low, high = delta_statistics(deltas, permutations, rng)
                bounded = low is not None and high is not None
                out = {**base, 'metric': metric, 'mean_delta': mean,
                       'metric_graph_ids': [g for g in metric_graphs
                                            if first[g][metric] is not None
                                            and second[g][metric] is not None],
                       'wins': int(np.sum(metric_first > metric_second)),
                       'ties': int(np.sum(metric_first == metric_second)),
                       'losses': int(np.sum(metric_first < metric_second)),
                       'signflip_p_value': pvalue, 'delta_ci_low': low,
                       'delta_ci_high': high, 'sesoi': sesoi,
                       'detectable_gain': max(abs(low), abs(high)) if bounded else None,
                       'max_abs_ci_bound': max(abs(low), abs(high)) if bounded else None,
                       'equivalent_within_sesoi': (low > -sesoi and high < sesoi)
                       if bounded else None,
                       'metric_graphs': int(deltas.size),
                       'informative_graphs': len(informative)}
                metric_graph_set = set(metric_graphs)
                informative_metric_graphs = [g for g in informative
                                             if g in metric_graph_set
                                             and first[g][metric] is not None
                                             and second[g][metric] is not None]
                ideltas = np.asarray([first[g][metric] - second[g][metric]
                                      for g in informative_metric_graphs])
                out.update(
                    informative_metric_graphs=len(informative_metric_graphs),
                    informative_mean_delta=(float(ideltas.mean())
                                            if ideltas.size else None),
                    informative_delta_ci_low=None,
                    informative_delta_ci_high=None,
                    informative_detectable_gain=None,
                    informative_equivalent_within_sesoi=None)
                result.append(out)
    return result


def diagnostics_stream(record_iter):
    totals = {
        'orbit_violations': 0, 'wl_violations': 0,
        'permutation_changes': 0, 'monotonicity': 0,
        'budget_limited': 0, 'unreliable': 0, 'checked': 0, 'rows': 0,
    }
    powers = {}
    identical = total = 0
    for rows, differences, timing in record_iter:
        for row in rows:
            totals['rows'] += 1
            totals['orbit_violations'] += row['orbit_violation_pairs']
            totals['wl_violations'] += row['wl_violation_pairs']
            totals['permutation_changes'] += row.get('permutation_pair_mask_changes', 0)
            totals['monotonicity'] += row.get('monotonicity_violations') or 0
            totals['budget_limited'] += int(row['mis_exact'] is False)
            totals['unreliable'] += int(not row.get('reliable', False))
            totals['checked'] += int(row.get('permutation_checks', 0) > 0)
            key = (row['graph_index'], row['items'], row['branch_dim'],
                   row['encoding'])
            if row['encoding'] == 'powers':
                powers[(row['graph_index'], row['items'],
                        row['branch_dim'])] = row['covered_orbit_pairs']
            elif row['encoding'] == 'powers_2d':
                other = powers.get((row['graph_index'], row['items'],
                                    row['branch_dim']))
                if other is not None:
                    total += 1
                    identical += int(row['covered_orbit_pairs'] == other)
    return totals, identical, total


class StreamAggregator:
    """Accumulate every final output while reading the stream file once."""

    def __init__(self):
        self.summary_groups = {}
        self.diff_groups = {}
        self.paired_groups = {}
        self.timings = []
        self.totals = {
            'orbit_violations': 0, 'wl_violations': 0,
            'permutation_changes': 0, 'monotonicity': 0,
            'budget_limited': 0, 'unreliable': 0, 'checked': 0, 'rows': 0,
        }
        self.identical = 0
        self.total_control = 0
        self.powers_cover = {}

    def add(self, rows, differences, timing):
        self.timings.append(timing)
        graph_index = rows[0]['graph_index'] if rows else None
        graph_powers = {}
        for row in rows:
            self.totals['rows'] += 1
            self.totals['orbit_violations'] += row['orbit_violation_pairs']
            self.totals['wl_violations'] += row['wl_violation_pairs']
            self.totals['permutation_changes'] += row.get('permutation_pair_mask_changes', 0)
            self.totals['monotonicity'] += row.get('monotonicity_violations') or 0
            self.totals['budget_limited'] += int(row['mis_exact'] is False)
            self.totals['unreliable'] += int(not row.get('reliable', False))
            self.totals['checked'] += int(row.get('permutation_checks', 0) > 0)

            key = (row['items'], row['encoding'], row['branch_dim'],
                   row['channels'])
            group = self.summary_groups.setdefault(key, _new_group(*key))
            group['graphs'] += 1
            group['stratum_counts'][row['orbit_stratum']] += 1
            if row['encoding'] == 'powers':
                graph_powers[(row['items'], row['branch_dim'])] = \
                    bool(row.get('reliable', False))
                self.powers_cover[(graph_index, row['items'],
                                   row['branch_dim'])] = row['covered_orbit_pairs']
            elif row['encoding'] == 'powers_2d':
                other = self.powers_cover.get((graph_index, row['items'],
                                               row['branch_dim']))
                if other is not None:
                    self.total_control += 1
                    self.identical += int(row['covered_orbit_pairs'] == other)
            if row.get('reliable', False):
                group['reliable_graphs'] += 1
                group['reliable_ids'].append(graph_index)
                group['unresolved_graphs'] += int(row.get('missed_orbit_pairs', 0) > 0)
            else:
                group['unreliable_graphs'] += 1
                group['unreliable_ids'].append(graph_index)
            for failure in row.get('reliability_failures', []):
                group['failure_counts'][failure] = \
                    group['failure_counts'].get(failure, 0) + 1
            if row.get('mis_exact') is False:
                group['mis_inexact'] += 1
            if row.get('mis_exceeds_orbits') is True:
                group['mis_exceeds_orbits'] += 1
            if row['orbit_violation_pairs'] > 0:
                group['orbit_violation_graphs'] += 1
            if row['wl_violation_pairs'] > 0:
                group['wl_violation_graphs'] += 1
            group['orbit_violation_all'] += row['orbit_violation_pairs']
            group['wl_violation_all'] += row['wl_violation_pairs']
            group['permutation_changes'] += row.get('permutation_pair_mask_changes', 0)
            group['permutation_checked'] += int(row.get('permutation_checks', 0) > 0)
            for field in MICRO_FIELDS:
                group['micro'][field] += row.get(field) or 0
            if row.get('reliable', False):
                for field in SUMMARY_FIELDS:
                    _update_stats(group['field_stats'][field], row.get(field))
                if row['orbit_stratum'] == 'intermediate':
                    for field in INTERMEDIATE_FIELDS:
                        _update_stats(group['intermediate_stats'][field],
                                      row.get(field))
            is_baseline_row = (
                row.get('resolved_pairs') is not None
                and graph_powers.get((row['items'], row['branch_dim']), False))
            if is_baseline_row and row.get('reliable', False):
                group['resolution_baseline_ids'].append(graph_index)
                group['resolution_seed_collisions'] += row['seed_collisions'] or 0
                group['resolved_pairs'] += row['resolved_pairs'] or 0
            if row.get('representative_count_exact') is True and row['objects'] > 0:
                group['exact_representative_count'] += 1
                if row.get('representative_ceiling_attained') is True:
                    group['attained_count'] += 1
            if row.get('representative_capacity_lower_bound') is not None \
                    and row.get('reliable', False):
                value = row['representative_capacity_lower_bound']
                group['bounded_representative_count'] += 1
                group['bounded_representative_total'] += value
                group['bounded_representative_squares'] += value * value
            for reference in ('orbit', 'wl'):
                field = reference + '_partition_reproduced'
                if row.get(field) is not None and row.get('reliable', False):
                    group[field + '_count'] += 1
                    group[field + '_sum'] += int(row[field])

            self.paired_groups.setdefault(
                (row['items'], row['branch_dim'], row['encoding']),
                {})[graph_index] = _compact_record(row)

        for row in differences:
            key = (row['items'], row['branch_dim'], row['comparison'])
            group = self.diff_groups.setdefault(key, {
                'items': key[0], 'branch_dim': key[1], 'comparison': key[2],
                'graphs': 0, 'reliable_graphs': 0, 'reliable_ids': [],
                'unreliable_ids': [],
                'first_only': 0, 'second_only': 0, 'shared': 0,
                'missed_by_both': 0})
            group['graphs'] += 1
            if row.get('reliable', False):
                group['reliable_graphs'] += 1
                group['reliable_ids'].append(row['graph_index'])
                for field in ('first_only', 'second_only', 'shared',
                              'missed_by_both'):
                    group[field] += row[field]
            else:
                group['unreliable_ids'].append(row['graph_index'])

    def finish(self, permutations, seed, sesoi=0.01):
        summary = [_group_summary(group)
                   for group in sorted(self.summary_groups.values(),
                                       key=lambda g: (g['kind'], g['route'],
                                                      g['width']))]
        difference_summary = [
            {'items': g['items'], 'branch_dim': g['branch_dim'],
             'comparison': g['comparison'], 'graphs': g['graphs'],
             'reliable_graphs': g['reliable_graphs'],
             'unreliable_graph_ids': sorted(g['unreliable_ids']),
             'reliable_graph_ids': sorted(g['reliable_ids']),
             **{field: g[field] for field in
                ('first_only', 'second_only', 'shared', 'missed_by_both')}}
            for g in sorted(self.diff_groups.values(),
                            key=lambda g: (g['items'], g['branch_dim'],
                                           g['comparison']))]
        paired = _paired_from_groups(self.paired_groups, permutations, seed,
                                     sesoi)
        return (summary, difference_summary, paired,
                self.totals, self.identical, self.total_control,
                self.timings)


def _paired_from_groups(groups, permutations, seed, sesoi):
    rng, result = np.random.default_rng(seed), []
    for kind, width in sorted({(kind, width) for kind, width, _ in groups}):
        for first_name, second_name in PAIRINGS:
            first = groups.get((kind, width, first_name), {})
            second = groups.get((kind, width, second_name), {})
            shared_all = sorted(set(first) & set(second))
            shared = [g for g in shared_all
                      if first[g]['reliable'] and second[g]['reliable']]
            if not shared:
                continue
            base = {'items': kind, 'branch_dim': width,
                    'comparison': f'{first_name}_vs_{second_name}',
                    'channels': first[shared[0]]['channels'],
                    'paired_graphs': len(shared),
                    'first_channels': first[shared[0]]['channels'],
                    'second_channels': second[shared[0]]['channels'],
                    'width_matched': first[shared[0]]['channels'] == second[shared[0]]['channels'],
                    'candidate_paired_graphs': len(shared_all),
                    'excluded_unreliable_graphs': len(shared_all) - len(shared)}
            informative = [g for g in shared if any(
                row['coverage_orbit'] is not None and row['coverage_orbit'] < 1
                for row in (first[g], second[g]))]
            for metric in PAIRED_METRICS:
                metric_graphs = shared
                if metric == 'representative_capacity':
                    metric_graphs = [g for g in shared
                                     if first[g]['representative_count_exact'] is True
                                     and second[g]['representative_count_exact'] is True]
                deltas = np.asarray([first[g][metric] - second[g][metric]
                                     for g in metric_graphs
                                     if first[g][metric] is not None
                                     and second[g][metric] is not None])
                if not deltas.size:
                    continue
                metric_first = np.asarray([first[g][metric] for g in metric_graphs
                                           if first[g][metric] is not None
                                           and second[g][metric] is not None])
                metric_second = np.asarray([second[g][metric] for g in metric_graphs
                                            if first[g][metric] is not None
                                            and second[g][metric] is not None])
                mean, pvalue, low, high = delta_statistics(deltas, permutations, rng)
                bounded = low is not None and high is not None
                out = {**base, 'metric': metric, 'mean_delta': mean,
                       'metric_graph_ids': [g for g in metric_graphs
                                            if first[g][metric] is not None
                                            and second[g][metric] is not None],
                       'wins': int(np.sum(metric_first > metric_second)),
                       'ties': int(np.sum(metric_first == metric_second)),
                       'losses': int(np.sum(metric_first < metric_second)),
                       'signflip_p_value': pvalue, 'delta_ci_low': low,
                       'delta_ci_high': high, 'sesoi': sesoi,
                       'detectable_gain': max(abs(low), abs(high)) if bounded else None,
                       'max_abs_ci_bound': max(abs(low), abs(high)) if bounded else None,
                       'equivalent_within_sesoi': (low > -sesoi and high < sesoi)
                       if bounded else None,
                       'metric_graphs': int(deltas.size),
                       'informative_graphs': len(informative)}
                metric_graph_set = set(metric_graphs)
                informative_metric_graphs = [g for g in informative
                                             if g in metric_graph_set
                                             and first[g][metric] is not None
                                             and second[g][metric] is not None]
                ideltas = np.asarray([first[g][metric] - second[g][metric]
                                      for g in informative_metric_graphs])
                out.update(
                    informative_metric_graphs=len(informative_metric_graphs),
                    informative_mean_delta=(float(ideltas.mean())
                                            if ideltas.size else None),
                    informative_delta_ci_low=None,
                    informative_delta_ci_high=None,
                    informative_detectable_gain=None,
                    informative_equivalent_within_sesoi=None)
                result.append(out)
    return result


def finalize_stream(path, permutations, seed, sesoi=0.01):
    """Read the stream file exactly once and produce every final output."""
    aggregator = StreamAggregator()
    with open(path) as stream_in:
        for line in stream_in:
            record = json.loads(line)
            aggregator.add(record['rows'], record['differences'],
                           record['timing'])
    return aggregator.finish(permutations, seed, sesoi)
