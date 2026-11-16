"""Legacy Q=8 input plus matched Q=1 ordered-triple augmentations."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from brec.budget_sweep import polynomial_bank
from .common import simple_adjacency

VARIANTS = ('original', 'baseline', 'joint')


def plane(matrix, first, second):
    shape = [1, 1, 1]
    shape[first] = shape[second] = len(matrix)
    return (matrix if first < second else matrix.T).reshape(shape)


def q1_joint(matrix):
    """All eight K_abc channels, lexicographic a,b,c in {0,1}."""
    n = len(matrix)
    eye = np.eye(n, dtype=bool)
    result = np.empty((8, n, n, n), dtype=np.float32)
    result[0] = eye[:, :, None] & eye[:, None, :]
    result[1] = eye[:, :, None] * matrix.T[:, None, :]
    result[2] = eye[:, None, :] * matrix.T[:, :, None]
    result[3] = matrix.T[:, :, None] * matrix.T[:, None, :]
    result[4] = eye[None, :, :] * matrix[:, :, None]
    result[5] = matrix[:, :, None] * matrix.T[None, :, :]
    result[6] = matrix[:, None, :] * matrix[None, :, :]
    # Only common neighbors contribute. Ascending pivots give a fixed sum order.
    result[7] = 0
    for pivot in range(n):
        neighbors = np.flatnonzero(matrix[:, pivot])
        weights = matrix[neighbors, pivot]
        contribution = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        result[7][np.ix_(neighbors, neighbors, neighbors)] += contribution
    return result


def input_array(adjacency, variant, seed_q=8, bank=None):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown triple PPGN variant: {variant}')
    adjacency = simple_adjacency(adjacency)
    n = len(adjacency)
    bank = polynomial_bank(adjacency, seed_q, 'chebyshev') if bank is None else bank
    if bank.shape != (seed_q + 1, n, n) or bank.dtype != np.float32:
        raise ValueError('Expected the legacy float32 Chebyshev bank')
    legacy_width = 1 + 6 * (seed_q + 2)
    result = np.empty((legacy_width + 27, n, n, n), dtype=np.float32)
    result[0] = 1
    positions = [(i, j) for i in range(3) for j in range(3) if i != j]
    offset = 1
    for matrix in [bank[0], adjacency.astype('float32'), *bank[1:]]:
        for first, second in positions:
            result[offset] = plane(matrix, first, second)
            offset += 1
    result[legacy_width:] = 0
    if variant != 'original':
        # Q=1 baseline is the 19-channel prefix of the legacy dictionary.
        result[legacy_width:legacy_width + 19] = result[:19]
    if variant == 'joint':
        result[legacy_width + 19:] = q1_joint(bank[1])
    if not np.isfinite(bank).all():
        raise FloatingPointError('Nonfinite legacy polynomial bank')
    return result


class TripleInputs:
    """Bound CPU caching; each official graph is encoded in its own numbering."""

    def __init__(self, graphs, seed_q=8, cache_gib=1):
        self.graphs, self.seed_q = graphs, seed_q
        self.limit = int(cache_gib * 2**30)
        self.cache, self.cached_bytes = OrderedDict(), 0
        self.banks = {}

    def array(self, index, variant):
        key = index, variant
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if index not in self.banks:
            self.banks[index] = polynomial_bank(self.graphs[index], self.seed_q, 'chebyshev')
        value = input_array(self.graphs[index], variant, self.seed_q, self.banks[index])
        if value.nbytes <= self.limit:
            while self.cache and self.cached_bytes + value.nbytes > self.limit:
                _, old = self.cache.popitem(last=False)
                self.cached_bytes -= old.nbytes
            self.cache[key] = value
            self.cached_bytes += value.nbytes
        return value

    def tensor(self, index, variant, device):
        import torch
        return torch.from_numpy(self.array(index, variant)).unsqueeze(0).to(device)
