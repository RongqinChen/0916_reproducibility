"""Order-independent exact walk polynomial numerators."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .common import atomic_npz


def _extend(bank, adjacency, degrees, q, basis, budget, accelerator):
    """Extend L**j B_j(D^-1 A), with L the lcm of nonzero scaling degrees."""
    scale = math.lcm(*degrees)
    factors = np.asarray([scale // degree for degree in degrees], dtype=object)[:, None]
    if not bank:
        eye = np.eye(len(adjacency), dtype=int).astype(object)
        transition = adjacency * factors
        bank.extend((eye, transition if basis != 'shifted' else scale * eye + transition))
    if len(bank) > q:
        return bank
    if accelerator is not None:
        multiply = accelerator.integer_product(adjacency)
    else:
        neighbors = [np.flatnonzero(row) for row in adjacency]

        def multiply(right):
            return np.asarray([right[indices].sum(0) for indices in neighbors], dtype=object)

    for _ in range(len(bank), q + 1):
        if budget:
            budget.check()
        product = factors * multiply(bank[-1])
        bank.append(product if basis == 'monomial' else 2 * product - scale**2 * bank[-2])
    return bank


def _pack(bank):
    width = (max(abs(int(x)).bit_length() for matrix in bank for x in matrix.flat) + 8) // 8
    contents = b''.join(int(x).to_bytes(width, 'little', signed=True)
                        for matrix in bank for x in matrix.flat)
    return np.frombuffer(contents, dtype=np.uint8).reshape(len(bank), *bank[0].shape, width)


def _digest(packed):
    return hashlib.sha256(memoryview(packed).cast('B')).hexdigest()


def _unpack(packed):
    rows = (int.from_bytes(x, 'little', signed=True) for x in packed.reshape(-1, packed.shape[-1]))
    values = np.fromiter(rows, dtype=object, count=math.prod(packed.shape[:-1]))
    return list(values.reshape(packed.shape[:-1]))


def scaled_polynomial_bank(adjacency, degrees, q, basis, budget=None, *,
                           accelerator=None, cache_directory=None):
    """Share the exact bank by labeled adjacency and basis, extending Q on demand."""
    if cache_directory is None:
        return _extend([], adjacency, degrees, q, basis, budget, accelerator)

    import fcntl
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph = np.ascontiguousarray(adjacency, dtype='<i8')
    identity = hashlib.sha256(f'scaled-walk-v1:{basis}:'.encode() + graph.tobytes()).hexdigest()
    target = directory / f'{identity}.npz'
    # A separate lock remains stable when atomic_npz replaces the data file.
    with (directory / f'{identity}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if budget:
            budget.check()
        bank = []
        if target.exists():
            with np.load(target, allow_pickle=False) as stored:
                packed = stored['numerators']
                if (packed.dtype != np.uint8 or packed.ndim != 4 or packed.shape[0] < 2
                        or packed.shape[1:3] != graph.shape or packed.shape[-1] < 1
                        or str(stored['identity']) != identity or str(stored['checksum']) != _digest(packed)):
                    raise ValueError(f'Invalid exact polynomial bank: {target}')
                bank = _unpack(packed[:q + 1])
            if len(bank) > q:
                return bank

        _extend(bank, adjacency, degrees, q, basis, budget, accelerator)
        packed = _pack(bank)
        atomic_npz(target, numerators=packed, identity=np.asarray(identity),
                   checksum=np.asarray(_digest(packed)))
        return bank
