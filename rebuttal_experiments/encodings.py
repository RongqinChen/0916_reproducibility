"""Original BREC channel formulas with numeric, rational and interval backends."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

from .common import simple_adjacency
from .intervals import Interval


def scalar_factory(mode, bits=128):
    if mode == 'exact':
        return Fraction
    if mode == 'interval':
        return lambda x: Interval.exact(x, bits)
    if mode not in ('float32', 'float64'):
        raise ValueError('Unknown arithmetic backend')
    return np.dtype(mode).type


def normalized_matrix(adjacency, mode, bits=128):
    adjacency = simple_adjacency(adjacency)
    degrees = adjacency.sum(1)
    make = scalar_factory(mode, bits)
    n = len(adjacency)
    if mode in ('float32', 'float64'):
        values = adjacency.astype(mode)
        degree = values.sum(1)
        inv = np.zeros(n, dtype=mode)
        np.divide(make(1), np.sqrt(degree), out=inv, where=degree > 0)
        return inv[:, None] * values * inv[None, :]
    if mode == 'exact' and not np.all(degrees == degrees[0]):
        raise ValueError('Rational backend is restricted to regular graphs')
    result = np.empty((n, n), dtype=object)
    for u in range(n):
        for v in range(n):
            if not adjacency[u, v]:
                result[u, v] = make(0)
            elif mode == 'exact':
                result[u, v] = Fraction(1, int(degrees[u]))
            else:
                result[u, v] = make(1) / Interval.sqrt_integer(int(degrees[u] * degrees[v]), bits)
    return result


def matmul(left, right, budget=None):
    if budget:
        budget.check()
    if left.dtype != object:
        return left @ right
    # Chunk object matrix products so the resource budget is checked regularly.
    result = np.empty((left.shape[0], right.shape[1]), dtype=object)
    for start in range(0, left.shape[0], 4):
        if budget:
            budget.check()
        result[start:start + 4] = left[start:start + 4] @ right
    return result


def polynomial_bank(adjacency, q, basis='monomial', mode='float64', bits=128, budget=None):
    if q < 1 or basis not in ('monomial', 'chebyshev', 'shifted'):
        raise ValueError('Invalid polynomial budget/basis')
    if basis == 'shifted' and q != 1:
        raise ValueError('The additional (I,I+M) basis is defined here only at Q=1')
    matrix = normalized_matrix(adjacency, mode, bits)
    make = scalar_factory(mode, bits)
    eye = np.asarray([[make(int(i == j)) for j in range(len(matrix))]
                      for i in range(len(matrix))], dtype=matrix.dtype)
    bank = [eye, eye + matrix if basis == 'shifted' else matrix]
    for _ in range(2, q + 1):
        product = matmul(matrix, bank[-1], budget)
        # Keep the ndarray on the left: an Interval scalar otherwise tries to
        # coerce the entire matrix to Fraction instead of multiplying its entries.
        bank.append(product if basis == 'monomial' else product * make(2) - bank[-2])
    return np.stack(bank)


def descriptors(q, order, variant):
    if order not in (2, 3) or variant not in ('polynomial', 'endpoint', 'joint'):
        raise ValueError('Invalid encoding')
    if order == 2 and variant == 'joint':
        raise ValueError('Joint channels require ordered triples')
    positions = [(0, 1)] if order == 2 else [(i, j) for i in range(3) for j in range(3) if i != j]
    keys = [('constant',)]
    for matrix in ['I', 'A'] + [f'B{degree}' for degree in range(1, q + 1)]:
        keys.extend(('pair', matrix, i, j) for i, j in positions)
    if variant != 'polynomial':
        for i, j in positions:
            keys.extend(('endpoint', a, b, i, j) for a in range(q)
                        for b in range(q - a) if (a, b) != (0, 0))
    if variant == 'joint':
        keys.extend(('joint', a, b, c) for a in range(q + 1)
                    for b in range(q + 1) for c in range(q + 1))
    return keys


def feature_chunks(adjacency, bank, q, order, variant, chunk_size=256,
                   joint_normalise=False, budget=None):
    adjacency = simple_adjacency(adjacency)
    n = len(adjacency)
    keys = descriptors(q, order, variant)
    # Interval width is representation-dependent; this is a conservative guard,
    # followed by actual RSS checks. Floating-point allocation sizes are exact.
    item_estimate = 256 if bank.dtype == object else bank.dtype.itemsize
    if budget:
        budget.check(chunk_size * len(keys) * item_estimate + bank.size * item_estimate)
    one = bank[0, 0, 0]
    matrix = {'I': bank[0], 'A': adjacency, **{f'B{a}': bank[a] for a in range(1, q + 1)}}
    h = bank.sum(2)
    for start in range(0, n**order, chunk_size):
        if budget:
            budget.check()
        ids = np.arange(start, min(n**order, start + chunk_size))
        coords = np.unravel_index(ids, (n,) * order)
        result = np.empty((len(ids), len(keys)), dtype=bank.dtype)
        for column, key in enumerate(keys):
            if key[0] == 'constant':
                result[:, column] = one
            elif key[0] == 'pair':
                _, name, i, j = key
                values = matrix[name][coords[i], coords[j]]
                result[:, column] = (values * one if bank.dtype == object else values)
            elif key[0] == 'endpoint':
                _, a, b, i, j = key
                result[:, column] = h[a, coords[i]] * h[b, coords[j]]
            else:
                _, a, b, c = key
                products = bank[a, coords[0], :] * bank[b, coords[1], :]
                products = products * bank[c, coords[2], :]
                result[:, column] = products.sum(axis=1)
                if joint_normalise:
                    result[:, column] = result[:, column] / n
        yield start, result


def feature_matrix(adjacency, q, order, variant, basis='monomial', mode='float64',
                   bits=128, budget=None):
    bank = polynomial_bank(adjacency, q, basis, mode, bits, budget)
    n, width = len(adjacency), len(descriptors(q, order, variant))
    if budget:
        budget.check(n**order * width * (256 if bank.dtype == object else bank.dtype.itemsize))
    result = np.empty((n**order, width), dtype=bank.dtype)
    for start, block in feature_chunks(adjacency, bank, q, order, variant, budget=budget):
        result[start:start + len(block)] = block
    return result


def moments(rows, p, budget=None):
    if p < 1 or not len(rows):
        raise ValueError('Positive moment budget and a nonempty object domain are required')
    result = []
    for power in range(1, p + 1):
        if budget:
            budget.check()
        result.extend((rows**power).sum(0) / len(rows))
    return result


def constructed_graphs():
    # Rook(4,4) and Cayley Shrikhande on Z_4 x Z_4.
    vertices = list(itertools.product(range(4), repeat=2))
    rook = np.zeros((16, 16), dtype='int64')
    shrikhande = np.zeros_like(rook)
    generators = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    for i, (a, b) in enumerate(vertices):
        for j, (c, d) in enumerate(vertices):
            rook[i, j] = int(i != j and (a == c or b == d))
            shrikhande[i, j] = int(((c - a) % 4, (d - b) % 4) in generators)
    return rook, shrikhande
