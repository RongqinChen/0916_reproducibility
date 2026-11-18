"""Exact/interval readout decisions with compact evidence for result reports."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

from .intervals import Interval, number_json

DIFFERENT = 'certified_different'
EQUAL = 'certified_equal'
UNRESOLVED = 'unresolved'


def mean_p_values_for_q(config, q):
    by_q = config.get('mean_p_values_by_q')
    return config['mean_p_values'] if by_q is None else by_q[str(q)]


def exact_multiset(left, right):
    first, second = Counter(map(tuple, left)), Counter(map(tuple, right))
    same = first == second
    evidence = {'kind': 'exact_multiset', 'objects': [len(left), len(right)],
                'distinct_rows': [len(first), len(second)]}
    if not same:
        row = next(row for group in (first, second) for row in group if first[row] != second[row])
        evidence.update(row=[number_json(x) for x in row],
                        multiplicities=[first[row], second[row]])
    return {'status': EQUAL if same else DIFFERENT, 'evidence': evidence}


def mean_decision(left, right):
    if len(left) != len(right):
        raise ValueError('Mean vectors must use the same channel indices')
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


def _count_bounds(values, threshold):
    return (sum(x.upper() < threshold for x in values),
            sum(x.lower() < threshold for x in values))


def projection_decision(left, right, budget=None, max_cuts=32):
    """Fixed projections can certify a full multiset difference."""
    width = len(left[0])
    probes = [[int(i == j) for i in range(width)] for j in range(width)]
    probes += [[(-1 if (i + probe) % 2 else 1) * (1 + (i * (probe + 1)) % 7)
                for i in range(width)] for probe in range(4)]
    for weights in probes:
        if budget:
            budget.check()
        first = [sum((x * w for x, w in zip(row, weights) if w), 0) for row in left]
        second = [sum((x * w for x, w in zip(row, weights) if w), 0) for row in right]
        centers = sorted(set(x.midpoint() for x in first + second))
        count = min(max_cuts, max(0, len(centers) - 1))
        cuts = [Fraction(0)]
        for i in range(count):
            j = i * (len(centers) - 1) // count
            cuts.append((centers[j] + centers[j + 1]) / 2)
        for cut in cuts:
            a, b = _count_bounds(first, cut), _count_bounds(second, cut)
            if a[0] > b[1] or b[0] > a[1]:
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
    quick = projection_decision(left, right, budget)
    if quick:
        return quick
    import networkx as nx
    first, second = list(Counter(map(tuple, left)).items()), list(Counter(map(tuple, right)).items())
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
    matched = residual.graph['flow_value']
    if matched == total:
        return {'status': UNRESOLVED, 'reason': 'compatible_perfect_matching_exists'}
    return {'status': DIFFERENT,
            'evidence': {'kind': 'matching_deficit',
                         'matched_objects': matched, 'required_objects': total}}
