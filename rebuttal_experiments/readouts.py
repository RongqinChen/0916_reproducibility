"""Exact/interval readout decisions with compact evidence for result reports."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

from .intervals import Interval, number_json
from .scaled_exact import (ScaledIntegerArray, common_denominator, scaled_counter,
                           scaled_mean_decision, scaled_row_json)

DIFFERENT = 'certified_different'
EQUAL = 'certified_equal'
UNRESOLVED = 'unresolved'


def mean_p_values_for_q(config, q):
    by_q = config.get('mean_p_values_by_q')
    return config['mean_p_values'] if by_q is None else by_q[str(q)]


def exact_multiset(left, right):
    denominator = common_denominator(left, right)
    if denominator is None:
        first, second = Counter(map(tuple, left)), Counter(map(tuple, right))
    else:
        first = scaled_counter(left, denominator)
        second = scaled_counter(right, denominator)
    same = first == second
    evidence = {'kind': 'exact_multiset', 'objects': [len(left), len(right)],
                'distinct_rows': [len(first), len(second)]}
    if not same:
        row = next(row for group in (first, second) for row in group if first[row] != second[row])
        encoded = (scaled_row_json(row, denominator)
                   if denominator is not None else [number_json(x) for x in row])
        evidence.update(row=encoded,
                        multiplicities=[first[row], second[row]])
    return {'status': EQUAL if same else DIFFERENT, 'evidence': evidence}


def mean_decision(left, right):
    if len(left) != len(right):
        raise ValueError('Mean vectors must use the same channel indices')
    if isinstance(left, ScaledIntegerArray) and isinstance(right, ScaledIntegerArray):
        return scaled_mean_decision(((left, right),), range(len(left)))
    for channel, (a, b) in enumerate(zip(left, right)):
        delta = a - b
        different = not delta.contains_zero() if isinstance(delta, Interval) else delta != 0
        if different:
            return {'status': DIFFERENT,
                    'evidence': {'kind': 'mean_difference', 'channel': channel,
                                 'left': number_json(a), 'right': number_json(b),
                                 'difference': number_json(delta)}}
    if all(not isinstance(x, Interval) or x.is_exact() for x in list(left) + list(right)):
        return {'status': EQUAL, 'evidence': {'kind': 'exact_mean',
                'values': [number_json(x) for x in left]}}
    return {'status': UNRESOLVED, 'reason': 'all_difference_intervals_contain_zero'}


def _project_interval(row, weights):
    result = None
    for value, weight in zip(row, weights):
        if weight:
            term = value * weight
            result = term if result is None else result + term
    if result is None:
        raise ValueError('Projection weights cannot all be zero')
    return result


def _count_bounds(values, threshold):
    """Weighted lower/upper counts using integer endpoint comparisons."""
    numerator = threshold.numerator
    denominator = threshold.denominator
    lower_count = upper_count = 0
    for value, count in values:
        scale = 1 << value.bits
        limit = numerator * scale
        if value.hi * denominator < limit:
            upper_count += count
        if value.lo * denominator < limit:
            lower_count += count
    return upper_count, lower_count


def projection_decision(left, right, budget=None, max_cuts=32, *,
                        left_items=None, right_items=None,
                        evidence_left_items=None, evidence_right_items=None):
    """Fixed projections can certify a full multiset difference."""
    first_items = (list(Counter(map(tuple, left)).items())
                   if left_items is None else left_items)
    second_items = (list(Counter(map(tuple, right)).items())
                    if right_items is None else right_items)
    width = len(first_items[0][0])
    probes = [[int(i == j) for i in range(width)] for j in range(width)]
    probes += [[(-1 if (i + probe) % 2 else 1) * (1 + (i * (probe + 1)) % 7)
                for i in range(width)] for probe in range(4)]
    for weights in probes:
        if budget:
            budget.check()
        first = [(_project_interval(row, weights), count)
                 for row, count in first_items]
        second = [(_project_interval(row, weights), count)
                  for row, count in second_items]
        centers = sorted(set(x.midpoint() for x, _ in first + second))
        count = min(max_cuts, max(0, len(centers) - 1))
        cuts = [Fraction(0)]
        for i in range(count):
            j = i * (len(centers) - 1) // count
            cuts.append((centers[j] + centers[j + 1]) / 2)
        for cut in cuts:
            a, b = _count_bounds(first, cut), _count_bounds(second, cut)
            if a[0] > b[1] or b[0] > a[1]:
                if evidence_left_items is not None:
                    evidence_first = [(_project_interval(row, weights), count)
                                      for row, count in evidence_left_items]
                    evidence_second = [(_project_interval(row, weights), count)
                                       for row, count in evidence_right_items]
                    a = _count_bounds(evidence_first, cut)
                    b = _count_bounds(evidence_second, cut)
                return {'status': DIFFERENT,
                        'evidence': {'kind': 'projection_count', 'weights': weights,
                                     'threshold': number_json(cut),
                                     'left_count_bounds': list(a), 'right_count_bounds': list(b)}}
    return None


def interval_multiset(left, right, budget=None, max_candidate_edges=2_000_000):
    if len(left) != len(right):
        return {'status': DIFFERENT,
                'evidence': {'kind': 'cardinality', 'objects': [len(left), len(right)]}}
    if not len(left):
        return exact_multiset([], [])
    if all(x.is_exact() for row in list(left) + list(right) for x in row):
        return exact_multiset([[x.lower() for x in row] for row in left],
                              [[x.lower() for x in row] for row in right])
    left_counts = Counter(map(tuple, left))
    right_counts = Counter(map(tuple, right))
    if left_counts == right_counts:
        if budget:
            budget.check()
        if len(left_counts) > max_candidate_edges:
            return {'status': UNRESOLVED, 'reason': 'matching_edge_limit'}
        return {'status': UNRESOLVED, 'reason': 'compatible_perfect_matching_exists'}
    first, second = list(left_counts.items()), list(right_counts.items())
    # Equal interval rows can always be paired with each other. Remove those
    # already-certified matches before building the much larger flow graph.
    prematched = 0
    left_residual = []
    right_residual = []
    for row, count in first:
        overlap = min(count, right_counts[row])
        prematched += overlap
        if count > overlap:
            left_residual.append((row, count - overlap))
    for row, count in second:
        overlap = min(count, left_counts[row])
        if count > overlap:
            right_residual.append((row, count - overlap))
    quick = projection_decision(
        left, right, budget,
        left_items=left_residual, right_items=right_residual,
        evidence_left_items=first, evidence_right_items=second)
    if quick:
        return quick
    first, second = left_residual, right_residual
    import networkx as nx
    graph = nx.DiGraph()
    source, sink, total, edges = ('source',), ('sink',), len(left), 0
    for i, (_, count) in enumerate(first):
        graph.add_edge(source, ('left', i), capacity=count)
    for j, (_, count) in enumerate(second):
        graph.add_edge(('right', j), sink, capacity=count)
    for i, (row, _) in enumerate(first):
        if budget:
            budget.check(512 * max(1, edges))
        for j, (other, _) in enumerate(second):
            if all(a.overlaps(b) for a, b in zip(row, other)):
                graph.add_edge(('left', i), ('right', j), capacity=total + 1)
                edges += 1
                if edges > max_candidate_edges:
                    return {'status': UNRESOLVED, 'reason': 'matching_edge_limit'}
    residual = nx.algorithms.flow.preflow_push(graph, source, sink)
    matched = prematched + residual.graph['flow_value']
    if matched == total:
        return {'status': UNRESOLVED, 'reason': 'compatible_perfect_matching_exists'}
    return {'status': DIFFERENT,
            'evidence': {'kind': 'matching_deficit',
                         'matched_objects': matched, 'required_objects': total}}
