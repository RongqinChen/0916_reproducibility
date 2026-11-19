"""Original BREC channel formulas with numeric, rational and interval backends."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

from .common import simple_adjacency
from .intervals import Interval
from .polynomial_banks import scaled_polynomial_bank
from .scaled_exact import ScaledIntegerArray


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
    if left.size == 0 or right.size == 0:
        return left @ right
    # Normalized adjacency is sparse. Object matmul otherwise constructs an
    # Interval/Fraction product even for every exact zero. Omit only exact-zero
    # factors; products and endpoint sums for the remaining terms are unchanged.
    result = np.empty((left.shape[0], right.shape[1]), dtype=object)
    for start in range(0, left.shape[0], 4):
        if budget:
            budget.check()
        for i in range(start, min(start + 4, left.shape[0])):
            row = left[i]
            nonzero = [k for k, value in enumerate(row)
                       if not (value.lo == value.hi == 0 if isinstance(value, Interval)
                               else value == 0)]
            if not nonzero:
                result[i] = row[0] * right[0, 0]
            elif len(nonzero) < len(row) // 2:
                result[i] = row[nonzero] @ right[nonzero]
            else:
                result[i] = row @ right
    return result


def polynomial_bank(adjacency, q, basis='monomial', mode='float64', bits=128, budget=None,
                    *, accelerator=None, bank_cache=None):
    if q < 1 or basis not in ('monomial', 'chebyshev', 'shifted'):
        raise ValueError('Invalid polynomial budget/basis')
    if basis == 'shifted' and q != 1:
        raise ValueError('The additional (I,I+M) basis is defined here only at Q=1')
    if mode == 'exact' and (accelerator is not None or bank_cache is not None):
        return _regular_polynomial_bank(adjacency, q, basis, budget, accelerator, bank_cache)
    matrix = normalized_matrix(adjacency, mode, bits)
    make = scalar_factory(mode, bits)
    eye = np.asarray([[make(int(i == j)) for j in range(len(matrix))]
                      for i in range(len(matrix))], dtype=matrix.dtype)
    bank = [eye, eye + matrix if basis == 'shifted' else matrix]
    gpu_product = (accelerator.interval_product(matrix, bits)
                   if accelerator is not None and mode == 'interval' and q > 1 else None)
    for _ in range(2, q + 1):
        if gpu_product is None:
            product = matmul(matrix, bank[-1], budget)
        else:
            if budget:
                budget.check()
            product = gpu_product(bank[-1])
        # Keep the ndarray on the left: an Interval scalar otherwise tries to
        # coerce the entire matrix to Fraction instead of multiplying its entries.
        bank.append(product if basis == 'monomial' else product * make(2) - bank[-2])
    return np.stack(bank)


def _regular_polynomial_bank(adjacency, q, basis, budget, accelerator, bank_cache):
    adjacency = simple_adjacency(adjacency)
    degrees = adjacency.sum(1)
    if not np.all(degrees == degrees[0]):
        raise ValueError('Rational backend is restricted to regular graphs')
    d = max(1, int(degrees[0]))
    # D^-1 A equals the symmetric normalization on regular graphs.
    bank = scaled_polynomial_bank(adjacency, [d] * len(adjacency), q, basis, budget,
                                  accelerator=accelerator, cache_directory=bank_cache)
    rational = np.vectorize(Fraction, otypes=[object])
    return np.stack([rational(matrix, d**degree) for degree, matrix in enumerate(bank)])


def scaled_exact_feature_matrix(adjacency, q, order, variant, basis='monomial',
                                mode='exact', bits=128, budget=None, *,
                                accelerator=None, bank_cache=None):
    """Return exact feature numerators on one common integer denominator.

    Regular-graph polynomial banks already use integer numerators with a known
    per-degree power of the degree as denominator. Keeping that representation
    avoids constructing and normalizing a Fraction object for every matrix
    entry while preserving exactly the same rational feature values.
    """
    if mode != 'exact':
        raise ValueError('Scaled exact features require exact mode')
    del bits
    adjacency = simple_adjacency(adjacency)
    degrees = adjacency.sum(1)
    if not np.all(degrees == degrees[0]):
        raise ValueError('Rational backend is restricted to regular graphs')
    degree = max(1, int(degrees[0]))
    bank = scaled_polynomial_bank(
        adjacency, [degree] * len(adjacency), q, basis, budget,
        accelerator=accelerator, cache_directory=bank_cache)
    keys = descriptors(q, order, variant)
    denominator_power = 3 * q if variant == 'joint' else q
    denominator = degree**denominator_power
    n = len(adjacency)
    result = np.empty((n**order, len(keys)), dtype=object)
    coordinates = np.unravel_index(np.arange(n**order), (n,) * order)
    column = 0
    if budget:
        budget.check(result.nbytes + len(keys) * n**order * 256)
    result[:, column] = denominator
    column += 1

    positions = [(0, 1)] if order == 2 else [
        (i, j) for i in range(3) for j in range(3) if i != j]
    matrices = [
        ('I', 0, bank[0]),
        ('A', 0, adjacency),
        *((f'B{power}', power, bank[power]) for power in range(1, q + 1)),
    ]
    for _, power, matrix in matrices:
        scale = degree**(denominator_power - power)
        for i, j in positions:
            values = np.asarray(matrix[coordinates[i], coordinates[j]], dtype=object)
            result[:, column] = values if scale == 1 else values * scale
            column += 1

    if variant != 'polynomial':
        sums = [matrix.sum(1) for matrix in bank[:q + 1]]
        for i, j in positions:
            for a in range(q):
                for b in range(q - a):
                    if (a, b) == (0, 0):
                        continue
                    scale = degree**(denominator_power - a - b)
                    values = sums[a][coordinates[i]] * sums[b][coordinates[j]]
                    result[:, column] = values if scale == 1 else values * scale
                    column += 1

    if variant == 'joint':
        for a in range(q + 1):
            for b in range(q + 1):
                for c in range(q + 1):
                    scale = degree**(denominator_power - a - b - c)
                    products = (bank[a][coordinates[0], :]
                                * bank[b][coordinates[1], :])
                    products = products * bank[c][coordinates[2], :]
                    values = products.sum(axis=1)
                    result[:, column] = values if scale == 1 else values * scale
                    column += 1
    if column != len(keys):
        raise AssertionError(f'Built {column} feature columns, expected {len(keys)}')
    return ScaledIntegerArray(result, denominator)


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
                   bits=128, budget=None, *, accelerator=None, bank_cache=None):
    bank = polynomial_bank(adjacency, q, basis, mode, bits, budget,
                           accelerator=accelerator, bank_cache=bank_cache)
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
    if isinstance(rows, ScaledIntegerArray):
        return [
            Fraction(int(value), block.denominator)
            for block in itertools.islice(moment_blocks(rows, budget), p)
            for value in block
        ]
    return list(itertools.chain.from_iterable(itertools.islice(moment_blocks(rows, budget), p)))


def moment_blocks(rows, budget=None):
    if isinstance(rows, ScaledIntegerArray):
        denominator = rows.denominator
        count = len(rows)
        for power in itertools.count(1):
            if budget:
                budget.check()
            numerator = (rows**power).sum(0)
            yield ScaledIntegerArray(numerator, denominator**power * count)
        return
    for power in itertools.count(1):
        if budget:
            budget.check()
        yield (rows**power).sum(0) / len(rows)


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
