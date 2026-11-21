"""Shared float32 Walk, Chebyshev, and sorted-channel computations."""

import numpy as np

from rebuttal_experiments.encodings import normalized_matrix

from .protocol import WALK_DEGREE


DOT_BITS, DOT_DIGITS = 8, 5
DOT_BLOCK = 256


def binary_digits(matrix, bits, count):
    """Split values on a fixed binary grid without changing the input dtype."""
    remainder = np.rint(np.ldexp(matrix, count*bits))
    digits = []
    for shift in range((count-1)*bits, 0, -bits):
        digit = np.floor(np.ldexp(remainder, -shift))
        digits.append(digit)
        remainder -= np.ldexp(digit, shift)
    return (*digits, remainder)


def sum_digit_products(left, right, bits):
    """The individual dot sums are exact integers in the working dtype."""
    result = np.zeros_like(left[0])
    left_digits = [(i, x) for i, x in enumerate(left) if x.any()]
    right_digits = [(j, y) for j, y in enumerate(right) if y.any()]
    for i, x in left_digits:
        for j, y in right_digits:
            if x.shape[1] <= DOT_BLOCK:
                product = x @ y
            else:
                # Each float32 block is exact. Extend the same binary grid to
                # larger graphs by accumulating the exact block counts once.
                counts = np.zeros_like(result, dtype=np.int64)
                for start in range(0, x.shape[1], DOT_BLOCK):
                    block = slice(start, start+DOT_BLOCK)
                    counts += (x[:, block] @ y[block]).astype(np.int64)
                product = counts.astype(np.float32)
            result += np.ldexp(product, -(i+j+2)*bits)
    return result


def reproducible_product(left, right):
    """Use exact partial dot sums for all BREC matrices and a fixed recombination order."""
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise TypeError('Expected float32 inputs')
    left_digits = binary_digits(left, DOT_BITS, DOT_DIGITS)
    right_digits = binary_digits(right, DOT_BITS, DOT_DIGITS)
    return sum_digit_products(left_digits, right_digits, DOT_BITS)


def walk_bank(graph):
    matrix = normalized_matrix(graph, 'float32')
    powers = [np.eye(len(matrix), dtype=matrix.dtype), matrix]
    fixed = binary_digits(matrix, DOT_BITS, DOT_DIGITS)
    for _ in range(2, WALK_DEGREE+1):
        previous = binary_digits(powers[-1], DOT_BITS, DOT_DIGITS)
        powers.append(sum_digit_products(fixed, previous, DOT_BITS))
    return np.stack(powers)


def chebyshev_pair(x, degree):
    """Evaluate T_degree and T_(degree+1) by the Chebyshev doubling identities."""
    if degree == 1:
        return x, 2*x*x - 1
    first, second = chebyshev_pair(x, degree//2)
    middle = 2*first*second - x
    if degree % 2:
        return middle, 2*second*second - 1
    return 2*first*first - 1, middle


def adjacent_walk_average(bank):
    """Keep both even- and odd-length routes in bipartite graphs."""
    return (bank[-2]+bank[-1])*np.float32(0.5)


def weighted_gap(left, right, weights=None):
    """Compute the sorted scalar-channel gap in the input dtype."""
    if weights is None:
        return float(np.max(np.abs(np.sort(left.ravel()) - np.sort(right.ravel()))))
    left_order, right_order = np.argsort(left), np.argsort(right)
    left_counts = np.cumsum(weights[0][left_order])
    right_counts = np.cumsum(weights[1][right_order])
    support = np.union1d(left_counts, right_counts)-1
    left_values = left[left_order[left_counts.searchsorted(support, side='right')]]
    right_values = right[right_order[right_counts.searchsorted(support, side='right')]]
    return float(np.max(np.abs(left_values-right_values)))


def gap_on_sorted(sorted_states):
    """Compare large sorted arrays without allocating a full difference array."""
    result = 0.0
    left, right = sorted_states
    chunk_size = 2**21
    for start in range(0, len(left), chunk_size):
        chunk = slice(start, start+chunk_size)
        first, second = left[chunk], right[chunk]
        current = float(np.max(np.abs(first-second)))
        if not np.isfinite(current):
            raise ValueError('Nonfinite quadruple channel')
        result = max(result, current)
    return result


def sorted_gap(states):
    ordered = [np.sort(state) for state in states]
    return gap_on_sorted(ordered)
