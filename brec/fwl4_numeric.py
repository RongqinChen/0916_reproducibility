"""Exact orbit layout and ordered four-coordinate numerical contractions."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pynauty
from numba import njit

from .fwl3_numeric import MODULI, NUMBA_VERSION, _shifted_power
from .suite_data import nauty_graph


PAIR_POSITIONS = tuple(combinations(range(4), 2))


@dataclass(frozen=True)
class QuadrupleOrbits:
    rows: np.ndarray
    weights: np.ndarray
    lookup: np.ndarray


@njit(cache=True)
def _indices(generators, n, queue_size):
    table = np.full(n**4, -1, np.int32)
    queue = np.empty(queue_size, np.int32)
    roots, weights = [], []
    for start in range(len(table)):
        if table[start] >= 0:
            continue
        orbit = len(roots)
        table[start] = orbit
        queue[0] = start
        front, back = 0, 1
        while front < back:
            index = queue[front]
            front += 1
            u, rest = divmod(index, n**3)
            v, rest = divmod(rest, n*n)
            w, z = divmod(rest, n)
            for p in generators:
                target = ((p[u]*n+p[v])*n+p[w])*n+p[z]
                if table[target] < 0:
                    if back == queue_size:
                        raise OverflowError('Orbit exceeds the automorphism group bound')
                    table[target] = orbit
                    queue[back] = target
                    back += 1
        roots.append(start)
        weights.append(back)
    return table, np.array(roots, dtype=np.int32), np.array(weights, dtype=np.int64)


def quadruple_orbits(graph):
    n = len(graph)
    if not n or n**4 > np.iinfo(np.int32).max:
        raise ValueError('Quadruple lookup requires a nonempty graph with int32 tuple indices')
    generators, order, exponent, *_ = pynauty.autgrp(nauty_graph(graph))
    generators = np.asarray(generators, dtype=np.int32).reshape(-1, n)
    capacity = min(n**4, 1+round(order*10**exponent))
    table, roots, weights = _indices(generators, n, capacity)
    rows = np.column_stack(np.unravel_index(roots, (n,)*4)).astype(np.int32)
    if int(weights.sum()) != n**4:
        raise AssertionError('The orbit weights must account for every ordered quadruple')
    return QuadrupleOrbits(rows, weights, table.reshape((n,)*4))


def atomic_types(graph, rows):
    code = np.zeros(len(rows), dtype=np.uint64)
    for index, (i, j) in enumerate(PAIR_POSITIONS):
        a, b = rows[:, i], rows[:, j]
        code += (3**index)*((2*(a == b) + graph[a, b]).astype(np.uint64))
    return code


@njit(cache=True)
def pivot_profiles(rows, lookup, colours, radix, one_word):
    profiles = np.empty((len(rows), len(lookup), 1 if one_word else 2), dtype=np.uint64)
    for i, (u, v, w, z) in enumerate(rows):
        for x in range(len(lookup)):
            a = colours[lookup[x, v, w, z]]
            b = colours[lookup[u, x, w, z]]
            c = colours[lookup[u, v, x, z]]
            d = colours[lookup[u, v, w, x]]
            high = a*radix+b
            low = c*radix+d
            if one_word:
                profiles[i, x, 0] = high*radix*radix+low
            else:
                profiles[i, x, 0] = high
                profiles[i, x, 1] = low
    return profiles


@njit(cache=True)
def _contract(rows, lookup, state, a, b, c, d, modulus):
    result = np.empty_like(state)
    for i, (u, v, w, z) in enumerate(rows):
        total = np.uint64(0)
        for x in range(len(lookup)):
            p = a[lookup[x, v, w, z]]*b[lookup[u, x, w, z]] % modulus
            q = c[lookup[u, v, x, z]]*d[lookup[u, v, w, x]] % modulus
            total += p*q
        result[i] = (state[i]+total) % modulus
    return result


def extend_quadruples(orbits, prefix, depth):
    """Append one full shared-pivot round using the previous ordered state."""
    if (depth < 1 or prefix.dtype != np.uint64
            or prefix.shape != (len(orbits.rows), 1+2*(depth-1))):
        raise ValueError('Expected the complete ordered uint64 prefix from the previous round')
    result = []
    for j, modulus in enumerate(MODULI):
        state = prefix[:, 0 if depth == 1 else -2+j].copy()
        if len(orbits.lookup)*(modulus-1)**2+modulus-1 > np.iinfo(np.uint64).max:
            raise OverflowError('Shared-pivot accumulation exceeds uint64 capacity')
        shifts = (7853+104729*depth+7919*j, 65537+130363*depth+6511*j,
                  196613+171007*depth+10427*j, 262147+196613*depth+13007*j)
        values = [_shifted_power(state, shift % modulus, degree, modulus)
                  for shift, degree in zip(shifts[:3], (5, 7, 11), strict=True)]
        x = (state+shifts[3] % modulus) % modulus
        square = x*x % modulus
        fourth = square*square % modulus
        eighth = fourth*fourth % modulus
        values.append((eighth*fourth % modulus)*x % modulus)
        result.append(_contract(orbits.rows, orbits.lookup, state, *values, np.uint64(modulus)))
    return np.column_stack((prefix, *result))
