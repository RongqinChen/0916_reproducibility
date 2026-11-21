"""Check that the certificate evaluates the same polynomial as the real features."""

from fractions import Fraction
from itertools import product
from math import prod

import numpy as np
import pytest

from brec import fwl_real as real
from brec.fwl3_numeric import triple_orbits


def _chebyshev(value, degree):
    x = 2 * value - 1
    a, b = 1, x
    for _ in range(1, degree):
        a, b = b, 2 * x * b - a
    return (1 + b) / 2


def _exact_step(values, n, normalizer):
    transformed = [{row: _chebyshev(value, degree) for row, value in values.items()}
                   for degree in real.DEGREES[:len(next(iter(values)))]]
    out = {}
    for row, value in values.items():
        total = sum(prod(f[row[:j] + (pivot,) + row[j + 1:]]
                         for j, f in enumerate(transformed)) for pivot in range(n))
        out[row] = value / 4 + 3 * total / (4 * normalizer)
    return out


@pytest.mark.parametrize('normalizer', [real.LEGACY_NORMALIZER, None])
def test_rational_recurrence_matches_both_computational_paths(normalizer):
    adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    n = len(adjacency)
    degree = adjacency.sum(1).tolist()
    pairs = {(u, v): Fraction(int(u == v), 2) + Fraction(int(adjacency[u, v]), 4 * degree[u] * degree[v])
             for u, v in product(range(n), repeat=2)}
    triple = {(u, v, w): (pairs[u, v] + 3 * pairs[u, w] + 9 * pairs[v, w]) / 13
              for u, v, w in product(range(n), repeat=3)}
    orbit = triple_orbits(adjacency)
    for k, values, plan in ((2, pairs, None), (3, triple, orbit)):
        rows = None if plan is None else plan.rows
        keys = list(values) if rows is None else list(map(tuple, rows))
        expected = values
        actual = real.seed(adjacency, k, rows)
        modular = [real.seed(adjacency, k, rows, prime=p) for p in real.PRIMES]
        for depth in range(3):
            expected_values = [expected[key] for key in keys]
            np.testing.assert_allclose(actual.ravel(), list(map(float, expected_values)),
                                       rtol=1e-12, atol=1e-15)
            for prime, evaluated in zip(real.PRIMES, modular, strict=True):
                assert evaluated.ravel().tolist() == [
                    (x.numerator * pow(x.denominator, -1, prime)) % prime for x in expected_values]
            if depth == 2:
                break
            expected = _exact_step(expected, n, n if normalizer is None else normalizer)
            actual = real.advance(actual, plan, normalizer=normalizer)
            modular = [real.advance(state, plan, prime=p, normalizer=normalizer)
                       for state, p in zip(modular, real.PRIMES, strict=True)]


def test_joint_certificate_can_detect_equal_marginals():
    a = np.array([1, 2], dtype=np.uint64)
    b = np.array([3, 4], dtype=np.uint64)
    weights = (np.array([2, 2]), np.array([2, 2]))
    assert real.residue_witness(([a], [a]), weights) is None
    result = real.residue_witness(([a, b], [a, b[::-1]]), weights)
    assert result == {'residues': [1, 3], 'multiplicities': [2, 0]}
