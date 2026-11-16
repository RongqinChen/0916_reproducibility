"""Official input ordering, exact graph checks, and weighted tuple orbits."""
import hashlib
import itertools
import os
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pynauty

EXPECTED_SHA256 = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class Graph6Store:
    """Read individual fixed-width NPY strings without mapping the entire dataset.

    Each worker reads only the graph records it needs. pread uses an explicit
    byte offset, so reads never share or mutate a file-position cursor.
    """
    def __init__(self, path):
        self._file = Path(path).open("rb")
        version = np.lib.format.read_magic(self._file)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(self._file)
        elif version in ((2, 0), (3, 0)):
            # Accepted scalar S/U dtypes have ASCII descriptors. For these
            # headers, NPY v2/v3 share the same 32-bit header length and syntax.
            shape, _, dtype = np.lib.format.read_array_header_2_0(self._file)
        else:
            raise ValueError("Unsupported NPY format version")
        if shape != (51200,) or dtype.kind not in "SU" or dtype.itemsize < 1:
            raise ValueError("Expected 51,200 fixed-width graph6 strings; pickle/object arrays are rejected")
        self.dtype, self.offset = dtype, self._file.tell()
        self.row_bytes = dtype.itemsize
        if os.fstat(self._file.fileno()).st_size != self.offset+51200*self.row_bytes:
            raise ValueError("Dataset byte length does not match its NPY header")

    def __len__(self):
        return 51200

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < 51200:
            raise IndexError("Graph record index out of range")
        raw = os.pread(self._file.fileno(), self.row_bytes, self.offset+int(index)*self.row_bytes)
        if len(raw) != self.row_bytes:
            raise ValueError("Truncated graph record")
        return np.frombuffer(raw, dtype=self.dtype, count=1)[0]


def load_data(path, expected=EXPECTED_SHA256):
    if sha256(path) != expected:
        raise ValueError("Dataset SHA256 does not match the selected protocol")
    return Graph6Store(path)


def adjacency(value):
    a = np.asarray(value)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
            or not np.isin(a, [0, 1]).all() or np.any(a.diagonal())
            or not np.array_equal(a, a.T)):
        raise ValueError("A nonempty simple undirected adjacency matrix is required")
    return a.astype(np.int64)


def graph_at(data, index):
    raw = data[index]
    raw = raw.encode("ascii") if isinstance(raw, str) else bytes(raw)
    return adjacency(nx.to_numpy_array(nx.from_graph6_bytes(raw.strip()), dtype=np.int64))


def graphs(data, pair_id, reliability=False, count=2):
    if not 0 <= pair_id < 400 or count not in (2, 64):
        raise ValueError("Invalid pair ID or graph block size")
    offset = 64 * (pair_id + (400 if reliability else 0))
    result = [graph_at(data, offset + i) for i in range(count)]
    if len({len(g) for g in result}) != 1:
        raise ValueError("Graphs in a BREC block must have equal vertex counts")
    return result


def reference_group(pair_id):
    """Historical author-supplied reference labels; never used to skip runs."""
    if 340 <= pair_id < 360:
        return "not_3fwl"
    if 110 <= pair_id < 160 or 320 <= pair_id < 340 or 360 <= pair_id < 400:
        return "only_3fwl"
    return "2fwl"


def category(pair_id):
    for end, name in [(60, "Basic"), (160, "Regular"), (260, "Extension"),
                      (360, "CFI"), (380, "4-Vertex Condition"), (400, "Distance Regular")]:
        if pair_id < end:
            return name
    raise ValueError("Pair ID out of range")


def nauty_graph(a, fixed=()):
    fixed = list(dict.fromkeys(fixed))
    rest = set(range(len(a))) - set(fixed)
    coloring = [{i} for i in fixed] + ([rest] if rest else [])
    return pynauty.Graph(len(a), adjacency_dict={i: np.flatnonzero(row).tolist()
                         for i, row in enumerate(a)}, vertex_coloring=coloring)


def isomorphism(first, second):
    """Return old-to-new vertex indices; verify the map using integer adjacency."""
    if len(first) != len(second):
        raise ValueError("Isomorphism controls have different sizes")
    left = pynauty.canon_label(nauty_graph(first))
    right = pynauty.canon_label(nauty_graph(second))
    mapping = np.empty(len(first), dtype=np.int64)
    mapping[left] = right
    if not np.array_equal(first, second[np.ix_(mapping, mapping)]):
        raise ValueError("Official reliability graphs are not isomorphic")
    return mapping


def tuple_orbits(a, order, compress=True):
    """Pointwise stabilizer chain; weights cover all ordered tuples, repeats included."""
    if not compress:
        rows = np.asarray(list(itertools.product(range(len(a)), repeat=order)), dtype=np.int64)
        return rows, np.ones(len(rows), dtype=np.int64)
    rows, weights = [], []

    def descend(prefix, weight):
        if len(prefix) == order:
            rows.append(prefix)
            weights.append(weight)
            return
        ids = pynauty.autgrp(nauty_graph(a, prefix))[3]
        groups = {}
        for vertex, color in enumerate(ids):
            groups.setdefault(color, []).append(vertex)
        for group in groups.values():
            descend(prefix + (group[0],), weight * len(group))

    descend((), 1)
    if sum(weights) != len(a)**order or min(weights) < 1:
        raise AssertionError("Tuple orbit weights do not cover the full domain")
    return np.asarray(rows, dtype=np.int64), np.asarray(weights, dtype=np.int64)


def two_fwl(first, second):
    """Joint color vocabulary for stable pair refinement, with multiplicity counts.

    The update retains the multiset of paired colors (c(u,t), c(t,v)).
    This is the folklore pair refinement, not the weaker separate-coordinate WL.
    """
    n = len(first)
    if len(second) != n:
        return {"equal": False, "rounds": 0, "reason": "different_orders"}
    colors = [np.eye(n, dtype=np.int64) * 2 + a for a in (first, second)]
    rounds = 0
    while True:
        vocabulary, updated = {}, []
        for old in colors:
            current = np.empty((n, n), dtype=np.int64)
            for u in range(n):
                for v in range(n):
                    profile = tuple(sorted(Counter(zip(old[u, :].tolist(),
                                                       old[:, v].tolist())).items()))
                    key = int(old[u, v]), profile
                    current[u, v] = vocabulary.setdefault(key, len(vocabulary))
            updated.append(current)
        rounds += 1
        if Counter(updated[0].flat) != Counter(updated[1].flat):
            return {"equal": False, "rounds": rounds, "reason": "color_histogram"}
        old_count = len(set(colors[0].flat) | set(colors[1].flat))
        colors = updated
        if len(vocabulary) == old_count:
            return {"equal": True, "rounds": rounds, "stable_colors": old_count}


def mechanism_graphs():
    vertices = list(itertools.product(range(4), repeat=2))
    rook = np.zeros((16, 16), dtype=np.int64)
    shrikhande = np.zeros_like(rook)
    steps = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    for i, (a, b) in enumerate(vertices):
        for j, (c, d) in enumerate(vertices):
            rook[i, j] = i != j and (a == c or b == d)
            shrikhande[i, j] = ((c-a) % 4, (d-b) % 4) in steps
    return [rook, shrikhande]
