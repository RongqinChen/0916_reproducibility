"""Preflight bounds for ordered quadruple refinement and encoding."""

from math import ceil

import numpy as np
import pynauty

from .suite_data import nauty_graph, tuple_orbits


GIB = 2**30


def _quadruple_bound(graph):
    """Use Burnside for small groups, or extend the exact triple orbit count."""
    generators, size, exponent, *_ = pynauty.autgrp(nauty_graph(graph))
    order = round(size*10**exponent)
    n = len(graph)
    if order > 16384:
        triples, _ = tuple_orbits(graph, 3)
        return len(triples)*n, len(generators)
    identity = np.arange(n, dtype=np.uint8)
    perms = [identity.tobytes()]
    seen = set(perms)
    suffix = bytes(range(n, 256))
    translations = [bytes(generator)+suffix for generator in generators]
    fixed_fourth_powers = 0
    for permutation in perms:
        fixed = int(np.count_nonzero(np.frombuffer(permutation, dtype=np.uint8) == identity))
        fixed_fourth_powers += fixed**4
        for table in translations:
            image = permutation.translate(table)
            if image not in seen:
                seen.add(image)
                perms.append(image)
    orbits, remainder = divmod(fixed_fourth_powers, order)
    if len(seen) != order or remainder:
        raise AssertionError('Orbit reservation disagrees with the automorphism group')
    return orbits, len(generators)


def plan_quadruple_work(graphs, max_rounds):
    n = max(map(len, graphs))
    bounds, generators = zip(*(_quadruple_bound(graph) for graph in graphs), strict=True)
    lookup_bytes = sum(4*len(graph)**4 for graph in graphs)
    # Exact-profile dictionary entries include two words per pivot plus
    # per-entry Python and count storage. Only the left signatures are stored.
    profiles = bounds[0]*(16*n+512)
    # Full ordered prefixes, two exact baselines, and the multiset sort scratch.
    row_workspace = sum(bounds)*(400+64*max_rounds)
    reservation = ceil((1.25*(lookup_bytes+max(profiles, row_workspace)))/GIB + 1.5)*GIB
    work = lookup_bytes*max(generators, default=1)//4 + sum(bounds)*n*8
    return {'orbit_rows_upper': list(bounds), 'lookup_bytes': lookup_bytes,
            'estimated_bytes': reservation, 'work': work}
