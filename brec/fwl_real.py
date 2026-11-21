"""Bounded real shared-pivot features and exact rational separation witnesses."""

from itertools import combinations

import numpy as np
from numba import njit


LEGACY_NORMALIZER = 256
DEGREES = (19, 23, 29)
ROUNDS = 6
TOLERANCE = 1e-8
PRIMES = (16777213, 16777199)
POSITION_WEIGHTS = (1, 3, 9)


def seed(adjacency, order, rows=None, *, prime=None):
    """Start with I/2 + (S ⊙ S)/4 on pairs and its weighted triple lift."""
    if not len(adjacency):
        raise ValueError('Expected a nonempty graph')
    degree = np.maximum(adjacency.sum(axis=1), 1)
    if prime is None:
        pairs = np.eye(len(adjacency)) / 2 + adjacency / (4 * np.outer(degree, degree))
    else:
        inverse = np.array([pow(int(d), -1, prime) for d in degree], dtype=np.uint64)
        pairs = (adjacency.astype(np.uint64) * (np.outer(inverse, inverse) % prime)
                 * np.uint64(pow(4, -1, prime))) % prime
        np.fill_diagonal(pairs, pow(2, -1, prime))
    if order == 2:
        return pairs
    values = sum(weight * pairs[rows[:, i], rows[:, j]]
                 for (i, j), weight in zip(combinations(range(3), 2), POSITION_WEIGHTS, strict=True))
    if prime is None:
        return values / sum(POSITION_WEIGHTS)
    return ((values % prime) * pow(sum(POSITION_WEIGHTS), -1, prime)) % prime


def shifted_chebyshev(state, order, prime=None):
    """Return (1 + T_d(2x-1))/2, applied separately to each entry."""
    degrees = DEGREES[:order]
    if prime is None:
        x = 2 * state - 1
        previous, current = np.ones_like(x), x
        result = []
        for d in range(2, degrees[-1] + 1):
            previous, current = current, 2 * x * current - previous
            if d in degrees:
                result.append((1 + current) / 2)
        return result
    p = np.uint64(prime)
    x = (2 * state + (p - 1)) % p
    previous, current = np.ones_like(x), x
    result = []
    for d in range(2, degrees[-1] + 1):
        previous, current = current, (2 * x * current + p - previous) % p
        if d in degrees:
            result.append(((1 + current) * pow(2, -1, prime)) % p)
    return result


@njit(cache=True)
def _triple_contract(rows, lookup, pivots, a, b, c):
    result = np.empty_like(a)
    for i, (u, v, w) in enumerate(rows):
        total = 0.0
        for x in pivots:
            total += a[lookup[x, v, w]] * b[lookup[u, x, w]] * c[lookup[u, v, x]]
        result[i] = total
    return result


@njit(cache=True)
def _triple_contract_mod(rows, lookup, a, b, c, prime):
    result = np.empty_like(a)
    for i, (u, v, w) in enumerate(rows):
        total = np.uint64(0)
        for x in range(len(lookup)):
            partial = a[lookup[x, v, w]] * b[lookup[u, x, w]] % prime
            total += partial * c[lookup[u, v, x]]
        result[i] = total % prime
    return result


def advance(state, orbit=None, *, pivots=None, prime=None, normalizer=None):
    """Average over all vertices and retain one quarter of the previous state."""
    n = len(state) if orbit is None else len(orbit.lookup)
    normalizer = n if normalizer is None else normalizer
    if n < 1 or normalizer < n:
        raise ValueError('The normalizer must cover all vertices in a nonempty graph')
    if prime is not None and n * (prime - 1)**2 > np.iinfo(np.uint64).max:
        raise OverflowError('The modular pivot sum exceeds the uint64 bound')
    a, b, *third = shifted_chebyshev(state, 2 if orbit is None else 3, prime)
    if orbit is None:
        # Slot one replaces the first coordinate and slot two the second.
        contract = b @ a
    elif prime is None:
        if pivots is None:
            pivots = np.arange(len(orbit.lookup), dtype=np.int32)
        contract = _triple_contract(orbit.rows, orbit.lookup, pivots, a, b, third[0])
    else:
        contract = _triple_contract_mod(orbit.rows, orbit.lookup, a, b, third[0], np.uint64(prime))
    if prime is None:
        return state / 4 + (3 / (4 * normalizer)) * contract
    return (state * pow(4, -1, prime)
            + (contract % prime) * (3 * pow(4 * normalizer, -1, prime) % prime)) % prime


def weighted_gap(left, right, weights=None):
    """L∞ distance between sorted scalar projections, using orbit multiplicities."""
    if weights is None:
        return float(np.max(np.abs(np.sort(left.ravel()) - np.sort(right.ravel()))))
    li, ri = np.argsort(left), np.argsort(right)
    lc, rc = np.cumsum(weights[0][li]), np.cumsum(weights[1][ri])
    support = np.union1d(lc, rc) - 1
    return float(np.max(np.abs(left[li[lc.searchsorted(support, side='right')]]
                               - right[ri[rc.searchsorted(support, side='right')]])))


@njit(cache=True)
def _residue_witness(left, right, left_weights, right_weights, li, ri):
    i, j = 0, 0
    while i < len(li) or j < len(ri):
        nl, nr = 0, 0
        take_left, take_right = i < len(li), j < len(ri)
        if take_left and take_right:
            for column in range(left.shape[0]):
                a, b = left[column, li[i]], right[column, ri[j]]
                if a != b:
                    take_left, take_right = a < b, b < a
                    break
        key = left[:, li[i]] if take_left else right[:, ri[j]]
        while i < len(li) and np.array_equal(key, left[:, li[i]]):
            nl += left_weights[li[i]]
            i += 1
        while j < len(ri) and np.array_equal(key, right[:, ri[j]]):
            nr += right_weights[ri[j]]
            j += 1
        if nl != nr:
            return key, nl, nr
    return np.empty(0, np.uint64), 0, 0


def residue_witness(prefixes, weights):
    """Return a joint residue row whose exact multiplicity differs."""
    left, right = (np.stack([column.ravel() for column in side]) for side in prefixes)
    if weights is None:
        weights = tuple(np.ones(side.shape[1], dtype=np.int64) for side in (left, right))
    key, nl, nr = _residue_witness(left, right, *weights,
                                  np.lexsort(left[::-1]), np.lexsort(right[::-1]))
    if nl != nr:
        return {'residues': key.tolist(), 'multiplicities': [int(nl), int(nr)]}
