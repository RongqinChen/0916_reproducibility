"""Nine protein encoding recipes from one shared degree-two channel bank."""
from __future__ import annotations

from itertools import product

import numpy as np

PROTOCOL = 'protein-encoding-grid-v1'
WIDTH = 64
POSITIONS = [(i, j) for i in range(3) for j in range(3) if i != j]
ENCODINGS = {
    'none': {'family': 'zero', 'q': 0, 'basis': 'none', 'channels': 0},
    'polynomial_q1': {'family': 'polynomial', 'q': 1, 'basis': 'monomial', 'channels': 19},
    'nonjoint_q1': {'family': 'degree-2 pair lifts plus h1', 'q': 2, 'basis': 'monomial', 'channels': 27},
    'joint_q1': {'family': 'polynomial plus joint', 'q': 1, 'basis': 'monomial', 'channels': 27},
    'joint_q1_shifted': {'family': 'polynomial plus joint', 'q': 1, 'basis': '(I,I+M)', 'channels': 27},
    'polynomial_q2': {'family': 'polynomial', 'q': 2, 'basis': 'monomial', 'channels': 25},
    'endpoint_q2': {'family': 'polynomial plus endpoint', 'q': 2, 'basis': 'monomial', 'channels': 37},
    'joint_q2': {'family': 'polynomial plus endpoint plus joint', 'q': 2, 'basis': 'monomial', 'channels': 64},
    'joint_q2_chebyshev': {'family': 'polynomial plus endpoint plus joint', 'q': 2, 'basis': '(I,M,2M^2-I)', 'channels': 64},
}
VARIANTS = tuple(ENCODINGS)
COMPARISONS = (
    ('polynomial_q1', 'none'),
    ('nonjoint_q1', 'polynomial_q1'),
    ('joint_q1', 'polynomial_q1'),
    ('joint_q1', 'nonjoint_q1'),
    ('joint_q1_shifted', 'joint_q1'),
    ('polynomial_q2', 'polynomial_q1'),
    ('endpoint_q2', 'polynomial_q2'),
    ('joint_q2', 'endpoint_q2'),
    ('joint_q2', 'joint_q1'),
    ('joint_q2_chebyshev', 'joint_q2'),
)


def validate_config(config):
    if config.get('encoding_protocol') != PROTOCOL or config.get('structural_width') != WIDTH:
        raise ValueError('Use the 64-slot protein encoding-grid configuration and a new output directory')
    variants = config['variants']
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError('Invalid encoding list')
    if (len(config['primary_comparison']) != 2 or len(set(config['primary_comparison'])) != 2
            or not set(config['primary_comparison']) <= set(variants)):
        raise ValueError('The primary comparison must be included in the encoding list')


def channel_projection(variant):
    """Map canonical [25 polynomial, 12 endpoint, 27 joint] rows to a recipe.

    Every recipe is retained in full and then zero-padded to 64 slots. Basis
    changes are complete linear transforms, not selected channel prefixes.
    """
    identity = np.eye(WIDTH)
    q1_joint = [37 + 9*a + 3*b + c for a, b, c in product(range(2), repeat=3)]
    if variant == 'none':
        return identity[:0]
    if variant == 'polynomial_q1':
        return identity[:19]
    if variant == 'nonjoint_q1':
        return identity[list(range(25)) + [26, 25]]  # h1(u), h1(v)
    if variant in ('joint_q1', 'joint_q1_shifted'):
        base, joint = identity[:19].copy(), identity[q1_joint]
        if variant == 'joint_q1_shifted':
            base[13:19] += identity[1:7]
            change = np.asarray([[1, 0], [1, 1]], dtype='float64')
            joint = np.kron(np.kron(change, change), change) @ joint
        return np.concatenate((base, joint), axis=0)
    if variant == 'polynomial_q2':
        return identity[:25]
    if variant == 'endpoint_q2':
        return identity[:37]
    if variant == 'joint_q2':
        return identity
    if variant == 'joint_q2_chebyshev':
        result = identity.copy()
        result[19:25] = 2 * identity[19:25] - identity[1:7]
        change = np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 2]], dtype='float64')
        result[37:] = np.kron(np.kron(change, change), change) @ identity[37:]
        return result
    raise ValueError(variant)


def canonical_rows(base, joint, h1, coords):
    """Reconstruct endpoint columns without storing their repeated values."""
    rows = np.zeros((len(coords[0]), WIDTH), dtype='float64')
    rows[:, :25] = base
    if joint is not None:
        rows[:, 37:] = joint
    for position, (i, j) in enumerate(POSITIONS):
        rows[:, 25 + 2*position] = h1[coords[j]]
        rows[:, 26 + 2*position] = h1[coords[i]]
    return rows


def project_rows(rows, projection):
    """Apply the sparse channel transform in float64 and pad to equal width."""
    result = np.zeros((len(rows), WIDTH), dtype='float64')
    if not len(projection):
        return result
    if np.count_nonzero(projection) == len(projection) and np.all(projection.max(1) == 1):
        result[:, :len(projection)] = rows[:, projection.argmax(1)]
        return result
    for target, weights in enumerate(projection):
        for source in np.flatnonzero(weights):
            result[:, target] += weights[source] * rows[:, source]
    return result
