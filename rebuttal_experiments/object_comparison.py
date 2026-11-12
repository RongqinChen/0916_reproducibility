"""Construct object features and compare them under the stated tolerance rule."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .encodings import matmul, normalized_matrix, scalar_factory
from .intervals import Interval


def object_domains(adjacency):
    n = len(adjacency)
    u, v = np.triu_indices(n, 1)
    edge = adjacency[u, v] > 0
    return {'nodes': (np.arange(n), np.arange(n)), 'edges': (u[edge], v[edge]), 'pairs': (u, v)}


def channel_plan(kind, width):
    first = 2 if kind == 'nodes' else 1
    powers = list(range(first, first + 2 * width))
    endpoints, total = [], 1
    while len(endpoints) < 2 * width:
        endpoints.extend((a, total - a) for a in range(total + 1) if a <= total - a)
        total += 1
    return powers, endpoints[:2 * width]


def build_object_features(adjacency, width, mode, bits=128, budget=None):
    """Build powers/endpoints with a common orientation for each object."""
    make = scalar_factory(mode, bits)
    n = len(adjacency)
    if mode in ('float32', 'float64'):
        # Preserve the original object's degree**(-0.5) computation order.
        array = adjacency.astype(mode)
        degree = array.sum(1)
        inverse = np.zeros(n, dtype=mode)
        inverse[degree > 0] = degree[degree > 0] ** -0.5
        matrix = array * inverse[:, None] * inverse[None, :]
    else:
        matrix = normalized_matrix(adjacency, mode, bits)
    plans = {kind: channel_plan(kind, width) for kind in object_domains(adjacency)}
    eye = np.asarray([[make(int(i == j)) for j in range(n)] for i in range(n)], dtype=matrix.dtype)
    bank = [eye]
    for _ in range(max(max(p) for p, _ in plans.values())):
        bank.append(matmul(bank[-1], matrix, budget))
    ones = np.asarray([[make(1)] for _ in range(n)], dtype=matrix.dtype)
    h = [ones[:, 0]]
    for _ in range(max(max(a, b) for _, pairs in plans.values() for a, b in pairs)):
        h.append(matmul(matrix, h[-1][:, None], budget)[:, 0])
    result = {}
    for kind, (u, v) in object_domains(adjacency).items():
        p, e = plans[kind]
        forward = [np.stack([bank[a][u, v] for a in p], 1),
                   np.stack([h[a][u] * h[b][v] for a, b in e], 1)]
        if kind == 'nodes':
            result[kind] = tuple(x[:, None, :] for x in forward)
        else:
            reverse = [np.stack([bank[a][v, u] for a in p], 1),
                       np.stack([h[a][v] * h[b][u] for a, b in e], 1)]
            result[kind] = tuple(np.stack((a, b), 1) for a, b in zip(forward, reverse))
    return result


def route_features(blocks, route, width):
    powers, endpoints = blocks
    if route == 'powers':
        return powers[..., :width]
    if route == 'powers_2d':
        return powers[..., :2 * width]
    if route == 'mixed':
        return np.concatenate((powers[..., :width], endpoints[..., :width]), -1)
    raise ValueError(route)


def align_features(original_adj, changed, new_to_old):
    old_to_new = np.argsort(new_to_old)
    original_domains = object_domains(original_adj)
    changed_domains = object_domains(original_adj[np.ix_(new_to_old, new_to_old)])
    result = {}
    for kind, (u, v) in original_domains.items():
        new_u, new_v = changed_domains[kind]
        lookup = {(int(a), int(b)): index for index, (a, b) in enumerate(zip(new_u, new_v))}
        indexes, swaps = [], []
        for a, b in zip(u, v):
            a, b = int(old_to_new[a]), int(old_to_new[b])
            indexes.append(lookup[(min(a, b), max(a, b))])
            swaps.append(a > b)
        blocks = []
        for block in changed[kind]:
            aligned = block[np.asarray(indexes, dtype='int64')].copy()
            if aligned.shape[1] == 2:
                swap = np.flatnonzero(swaps)
                aligned[swap] = aligned[swap, ::-1, :]
            blocks.append(aligned)
        result[kind] = tuple(blocks)
    return result


def separation_mask(values, atol, rtol, chunk=128, return_margin=False):
    count = len(values)
    separated = np.zeros((count, count), dtype=bool)
    margins = np.zeros((count, count), dtype='float64') if return_margin else None
    atol = np.asarray(atol, dtype=values.dtype)
    rtol = values.dtype.type(rtol)
    for start in range(0, count, chunk):
        left = values[start:start + chunk]
        direct = np.abs(left[:, None] - values[None])
        direct -= atol + rtol * np.maximum(np.abs(left[:, None]), np.abs(values[None]))
        margin = direct.max(axis=(-1, -2))
        if values.shape[1] == 2:
            reverse = np.abs(left[:, None] - values[None, :, ::-1])
            reverse -= atol + rtol * np.maximum(np.abs(left[:, None]), np.abs(values[None, :, ::-1]))
            margin = np.minimum(margin, reverse.max(axis=(-1, -2)))
        separated[start:start + len(left)] = margin > 0
        if margins is not None:
            margins[start:start + len(left)] = margin
    np.fill_diagonal(separated, False)
    return (separated, margins) if return_margin else separated


def _absolute_bounds(value):
    if isinstance(value, Interval):
        a, b = value.lower(), value.upper()
    else:
        a = b = Fraction(value)
    return (Fraction(0) if a <= 0 <= b else min(abs(a), abs(b)), max(abs(a), abs(b)))


def reference_comparison(left, right, atol, rtol):
    """Conservative comparison of the real-valued tolerance rule."""
    swaps = [False, True] if len(left) == 2 else [False]
    outcomes = []
    for swapped in swaps:
        states = []
        other = right[::-1] if swapped else right
        for a, b in zip(left.flat, other.flat):
            dlo, dhi = _absolute_bounds(a - b)
            alo, ahi = _absolute_bounds(a)
            blo, bhi = _absolute_bounds(b)
            tlo, thi = atol + rtol * max(alo, blo), atol + rtol * max(ahi, bhi)
            states.append('close' if dhi <= tlo else 'far' if dlo > thi else 'unknown')
        outcomes.append('far' if 'far' in states else 'close' if all(x == 'close' for x in states) else 'unknown')
    if 'close' in outcomes:
        return 'certified_similar'
    return 'certified_separated' if all(x == 'far' for x in outcomes) else 'unresolved'


def object_metrics(mask, orbit_labels, wl_labels, config):
    from discriminability.evaluation import maximum_independent_set, partition_metrics
    from discriminability.references import class_count
    result = partition_metrics(mask, orbit_labels, wl_labels)
    similarity = ~mask
    np.fill_diagonal(similarity, False)
    representatives, exact, search = maximum_independent_set(similarity, config['mis_search_nodes'], True)
    if representatives and np.any(similarity[np.ix_(representatives, representatives)]):
        raise AssertionError('Invalid independent-set witness')
    orbits = class_count(orbit_labels)
    result.update(representatives=representatives, mis_exact=exact, mis_search_nodes=search,
                  orbits=orbits, rc=(len(representatives) / orbits if exact and orbits else None))
    return result
