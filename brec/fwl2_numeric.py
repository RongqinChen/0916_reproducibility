"""Dictionary-free numerical pair features using modular contractions."""

from __future__ import annotations

import numpy as np

from rebuttal_experiments.common import simple_adjacency


MODULI = (16777213, 16777199)


def _matmul_mod(left: np.ndarray, right: np.ndarray, modulus: int) -> np.ndarray:
    terms_per_block = int(np.iinfo(np.uint64).max) // (modulus - 1)**2
    n = left.shape[1]
    if n <= terms_per_block:
        return (left @ right) % modulus
    result = np.zeros((len(left), right.shape[1]), dtype=np.uint64)
    for start in range(0, n, terms_per_block):
        stop = start + terms_per_block
        partial = (left[:, start:stop] @ right[start:stop]) % modulus
        result = (result + partial) % modulus
    return result


def encode_numeric_pairs(adjacency: np.ndarray, rounds: int = 5) -> np.ndarray:
    """Return an (n, n, 1 + 2*rounds) array of independently computed features."""
    adjacency = simple_adjacency(adjacency)
    initial = adjacency.astype(np.uint64) + 2 * np.eye(len(adjacency), dtype=np.uint64)
    return extend_numeric_pairs(initial[..., None], rounds)


def extend_numeric_pairs(features: np.ndarray, rounds: int) -> np.ndarray:
    """Append rounds to a saved feature tensor using its last numerical state."""
    if (features.ndim != 3 or features.shape[0] != features.shape[1]
            or not len(features) or features.dtype != np.dtype(np.uint64)):
        raise ValueError('Expected a nonempty uint64 pair feature tensor')
    width = features.shape[-1]
    completed, remainder = divmod(width - 1, len(MODULI))
    if width < 1 or remainder:
        raise ValueError('Expected an initial channel followed by complete numerical rounds')
    if type(rounds) is not int or rounds < completed:
        raise ValueError('The requested depth must be at least the depth already computed')
    if rounds == completed:
        return features

    if completed:
        states = [features[..., width - len(MODULI) + index] for index in range(len(MODULI))]
    else:
        states = [features[..., 0]] * len(MODULI)
    channels = []
    for step in range(completed + 1, rounds + 1):
        for index, modulus in enumerate(MODULI):
            state = states[index]
            left_shift = (7853 + 104729 * step + 7919 * index) % modulus
            right_shift = (65537 + 130363 * step + 6511 * index) % modulus
            x, y = (state + left_shift) % modulus, (state + right_shift) % modulus
            x2, y2 = x*x % modulus, y*y % modulus
            x4, y4 = x2*x2 % modulus, y2*y2 % modulus
            left = x4*x % modulus
            right = ((y4*y2 % modulus)*y) % modulus
            states[index] = (state + _matmul_mod(left, right, modulus)) % modulus
        channels.extend(states)
    return np.concatenate((features, np.stack(channels, axis=-1)), axis=-1)


def compare_numeric_encodings(first: np.ndarray, second: np.ndarray) -> dict:
    """Compare exact row counts and locate the first distinguishing prefix."""
    if first.shape[-1] != second.shape[-1]:
        raise ValueError('Compare encodings with the same number of rounds')
    if len(first) != len(second):
        return {'different': True, 'round': 0, 'vertex_counts': [len(first), len(second)]}

    rows = []
    for encoding in (first, second):
        flat = encoding.reshape(-1, encoding.shape[-1])
        rows.append(flat[np.lexsort(flat.T[::-1])])
    for step, width in enumerate(range(1, first.shape[-1] + 1, len(MODULI))):
        left, right = (row[:, :width] for row in rows)
        mismatches = np.flatnonzero(np.any(left != right, axis=1))
        if mismatches.size:
            index = mismatches[0]
            # The smaller first mismatch has different counts in the two sorted prefixes.
            feature = min(left[index].tolist(), right[index].tolist())
            counts = [int(np.count_nonzero(np.all(row == np.asarray(feature, np.uint64), axis=1)))
                      for row in (left, right)]
            return {'different': True, 'round': step,
                    'witness': {'features': feature, 'multiplicities': counts}}
    return {'different': False, 'round': None}
