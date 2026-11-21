"""Ordered triple orbits and dictionary-free joint replacement channels."""

from dataclasses import dataclass

import numpy as np
import pynauty
from numba import __version__ as NUMBA_VERSION, njit

from .fwl2_numeric import MODULI
from .suite_data import nauty_graph


@dataclass(frozen=True)
class TripleOrbits:
    rows: np.ndarray
    weights: np.ndarray
    lookup: np.ndarray


@njit(cache=True)
def _orbit_indices(generators, n):
    lookup = np.full(n**3, -1, np.int32)
    queue = np.empty(n**3, np.int32)
    count = 0
    for start in range(n**3):
        if lookup[start] >= 0:
            continue
        front, back = 0, 1
        queue[0] = start
        lookup[start] = count
        while front < back:
            index = queue[front]
            front += 1
            u, tail = divmod(index, n*n)
            v, w = divmod(tail, n)
            for generator in generators:
                image = (generator[u]*n + generator[v])*n + generator[w]
                if lookup[image] < 0:
                    lookup[image] = count
                    queue[back] = image
                    back += 1
        count += 1
    return lookup, count


def triple_orbits(graph):
    n = len(graph)
    if not n or n**3 > np.iinfo(np.int32).max:
        raise ValueError('Triple orbit lookup requires a nonempty graph and int32 tuple indices')
    generators = np.asarray(pynauty.autgrp(nauty_graph(graph))[0], dtype=np.int32).reshape(-1, n)
    flat, count = _orbit_indices(generators, n)
    weights = np.bincount(flat, minlength=count)
    first = np.full(count, len(flat), dtype=np.int32)
    np.minimum.at(first, flat, np.arange(len(flat), dtype=np.int32))
    rows = np.column_stack(np.unravel_index(first, (n, n, n))).astype(np.int32)
    return TripleOrbits(rows, weights, flat.reshape(n, n, n))


def atomic_types(graph, rows):
    u, v, w = rows.T
    return ((2*(u == v) + graph[u, v])
            + 3*(2*(u == w) + graph[u, w])
            + 9*(2*(v == w) + graph[v, w])).astype(np.uint64)


@njit(cache=True)
def _contract(rows, lookup, prior, first, second, third, modulus):
    result = np.empty_like(prior)
    for index, (u, v, w) in enumerate(rows):
        total = np.uint64(0)
        for x in range(len(lookup)):
            # Reduce after two factors so every product fits in uint64.
            product = first[lookup[x, v, w]] * second[lookup[u, x, w]] % modulus
            total += product * third[lookup[u, v, x]]
        result[index] = (prior[index] + total) % modulus
    return result


def _shifted_power(state, shift, exponent, modulus):
    value = (state + shift) % modulus
    squared = value*value % modulus
    fourth = squared*squared % modulus
    if exponent == 5:
        return fourth*value % modulus
    if exponent == 7:
        return (fourth*squared % modulus)*value % modulus
    eighth = fourth*fourth % modulus
    return (eighth*squared % modulus)*value % modulus


def transformed_states(state, depth, index, modulus):
    shifts = (7853 + 104729*depth + 7919*index,
              65537 + 130363*depth + 6511*index,
              196613 + 171007*depth + 10427*index)
    return [_shifted_power(state, shift % modulus, exponent, modulus)
            for shift, exponent in zip(shifts, (5, 7, 11), strict=True)]


def extend_triples(orbits, prefix, depth):
    """Append one round to the same graph's previous numerical prefix."""
    if depth < 1 or prefix.shape != (len(orbits.rows), 1 + 2*(depth-1)):
        raise ValueError('Expected a complete prefix from the previous triple round')
    result = []
    for j, modulus in enumerate(MODULI):
        state = prefix[:, 0 if depth == 1 else -2+j].copy()
        if len(orbits.lookup)*(modulus-1)**2 + modulus-1 > np.iinfo(np.uint64).max:
            raise OverflowError('Triple contraction exceeds uint64 accumulation capacity')
        three = transformed_states(state, depth, j, modulus)
        result.append(_contract(orbits.rows, orbits.lookup, state, *three, np.uint64(modulus)))
    return np.column_stack((prefix, *result))


@njit(cache=True)
def joint_profiles(rows, lookup, colours, radix):
    """Encode exact ordered colour triples before sorting the pivot multisets."""
    profiles = np.empty((len(rows), len(lookup)), dtype=np.uint64)
    for index, (u, v, w) in enumerate(rows):
        for x in range(len(lookup)):
            a = colours[lookup[x, v, w]]
            b = colours[lookup[u, x, w]]
            c = colours[lookup[u, v, x]]
            profiles[index, x] = (a*radix + b)*radix + c
    return profiles
