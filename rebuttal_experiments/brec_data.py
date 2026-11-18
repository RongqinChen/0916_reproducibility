"""Load BREC graph records and align the official reliability pairs."""
from __future__ import annotations

import numpy as np

from .common import sha256, simple_adjacency

PAIR_COUNT = 400
RELABELS_PER_SIDE = 32
BLOCK_SIZE = 2 * RELABELS_PER_SIDE
GRAPH_COUNT = 2 * PAIR_COUNT * BLOCK_SIZE
OFFICIAL_RELIABILITY = 'brec_official'


def validate_reliability_configuration(config):
    protocol = config.get('reliability_protocol', 'sampled')
    if protocol not in ('sampled', OFFICIAL_RELIABILITY):
        raise ValueError('reliability_protocol must be sampled or brec_official')
    for key, minimum in (('relabels', 1), ('reliability_relabels', 0)):
        if protocol == OFFICIAL_RELIABILITY:
            value = config.get(key, 0)
            if type(value) is not int or value != 0:
                raise ValueError(f'{key} must be zero for brec_official reliability')
        elif type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')


def load_brec_data(path, expected_sha256):
    if expected_sha256 and sha256(path) != expected_sha256:
        raise ValueError('BREC input hash differs from the frozen protocol')
    data = np.load(path, allow_pickle=False, mmap_mode='r')
    if data.shape != (GRAPH_COUNT,) or data.dtype.kind not in ('S', 'U'):
        raise ValueError('Expected the 51200-entry official graph6 byte/string array')
    return data


def graph_at(data, index):
    import networkx as nx
    value = data[index]
    value = value.encode() if isinstance(value, str) else bytes(value)
    return simple_adjacency(nx.to_numpy_array(nx.from_graph6_bytes(value.strip()), dtype='int64'))


def main_graph_pair(data, pid):
    return [graph_at(data, BLOCK_SIZE * pid + side) for side in (0, 1)]


def official_reliability_pairs(data, pid, config):
    """Read adjacent comparisons from the official 64-graph reliability block."""
    offset = (pid + PAIR_COUNT) * BLOCK_SIZE
    count = RELABELS_PER_SIDE if config.get('reliability_protocol') == OFFICIAL_RELIABILITY else 1
    for pair_index in range(count):
        first = offset + 2 * pair_index
        indices = (first, first + 1)
        yield indices, [graph_at(data, index) for index in indices]


def reliability_mapping(first, second):
    """Find and verify a control-only isomorphism without VF2 backtracking.

    Canonical vertex orders come from nauty. The resulting permutation is
    independently checked against the integer adjacencies before it is used.
    Main graph pairs are never passed to this function.
    """
    import pynauty
    first, second = simple_adjacency(first), simple_adjacency(second)
    n = len(first)
    if len(second) != n:
        raise ValueError('The official reliability representatives have different node counts')
    orders = []
    for adjacency in (first, second):
        graph = pynauty.Graph(n, directed=False, adjacency_dict={
            i: np.flatnonzero(adjacency[i]).tolist() for i in range(n)})
        order = pynauty.canon_label(graph)
        if sorted(order) != list(range(n)):
            raise ValueError('Invalid canonical vertex ordering')
        orders.append(order)
    old_to_new = np.empty(n, dtype='int64')
    old_to_new[orders[0]] = orders[1]
    new_to_old = np.argsort(old_to_new)
    if not np.array_equal(first[np.ix_(new_to_old, new_to_old)], second):
        raise ValueError('The official reliability representatives are not isomorphic')
    return old_to_new
