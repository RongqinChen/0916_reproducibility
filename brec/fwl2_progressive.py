"""Exact baseline readout with progressively appended numerical 2-FWL channels."""

from __future__ import annotations

from collections import Counter
from itertools import chain

import numpy as np

from .fwl2_numeric import MODULI, encode_numeric_pairs, extend_numeric_pairs
from .suite_data import tuple_orbits
from rebuttal_experiments.common import simple_adjacency
from rebuttal_experiments.encodings import descriptors
from rebuttal_experiments.polynomial_banks import scaled_polynomial_bank
from rebuttal_experiments.triple_exact import TripleKernel


def _baseline_rows(graph, q, basis, variant):
    kernel = TripleKernel(graph, q, basis)
    vertices, weights = tuple_orbits(kernel.adjacency, 2)
    # Mixed endpoint products are determined by their two factors. Each factor
    # is already an original coordinate, since h_0(v) == 1.
    keys = [key for key in descriptors(q, 2, variant)
            if key[0] != 'endpoint' or 0 in key[1:3]]
    rows = [tuple(kernel.value(pair, key) for key in keys) for pair in vertices]
    return vertices, weights.tolist(), rows


def _row_counts(rows, weights):
    counts = Counter()
    for row, weight in zip(rows, weights, strict=True):
        counts[row] += weight
    return counts


def compare_progressively(first, second, *, q, basis, variant, max_rounds):
    """Return the earliest difference and the numerical prefixes actually computed."""
    if type(max_rounds) is not int or max_rounds < 0:
        raise ValueError('Maximum numerical depth must be a nonnegative integer')
    numeric = [None, None]
    if len(first) != len(second):
        return {'different': True, 'round': 0, 'computed_rounds': 0,
                'status': 'certified_different',
                'vertex_counts': [len(first), len(second)]}, numeric

    graphs = (first, second)
    orbit_data = [_baseline_rows(graph, q, basis, variant) for graph in graphs]
    rows = [data[2] for data in orbit_data]
    base_width = len(rows[0][0])

    for depth in range(max_rounds + 1):
        if depth:
            for side, (vertices, _, _) in enumerate(orbit_data):
                state = numeric[side]
                if state is None:
                    state = encode_numeric_pairs(graphs[side], rounds=0)
                numeric[side] = extend_numeric_pairs(state, depth)
                new_channels = numeric[side][vertices[:, 0], vertices[:, 1], -len(MODULI):]
                rows[side] = [row + tuple(channels) for row, channels in
                              zip(rows[side], new_channels.tolist(), strict=True)]

        left, right = (_row_counts(row, data[1])
                       for row, data in zip(rows, orbit_data, strict=True))
        if left != right:
            witness = next(row for row in chain(left, right) if left[row] != right[row])
            return {'different': True, 'round': depth, 'computed_rounds': depth,
                    'status': 'certified_different', 'witness': {
                        'baseline': [value.json() for value in witness[:base_width]],
                        'numerical': list(witness[base_width:]),
                        'multiplicities': [left[witness], right[witness]],
                    }}, numeric
    return {'different': False, 'round': None, 'computed_rounds': max_rounds,
            'status': 'unresolved'}, numeric


def check_control(first, second, new_to_old, *, q, basis, depth):
    """Check the full progressive prefix under the official isomorphism."""
    graphs = [simple_adjacency(graph) for graph in (first, second)]
    alignment = np.ix_(new_to_old, new_to_old)
    if not np.array_equal(graphs[0][alignment], graphs[1]):
        raise AssertionError('The official BREC control adjacencies do not align')
    # Checking the exact bank and adjacency also checks normalized polynomial
    # values and every endpoint product. Both are determined by these arrays.
    banks = [scaled_polynomial_bank(graph, np.maximum(graph.sum(1), 1).tolist(), q, basis)
             for graph in graphs]
    for left, right in zip(*banks, strict=True):
        if not np.array_equal(left[alignment], right):
            raise AssertionError('The baseline encoding changed on a BREC reliability pair')
    if depth:
        left, right = (encode_numeric_pairs(graph, rounds=depth) for graph in graphs)
        if not np.array_equal(left[alignment], right):
            raise AssertionError('The numerical encoding changed on a BREC reliability pair')
