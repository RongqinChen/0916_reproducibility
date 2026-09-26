"""Automorphism-orbit and 2-FWL reference partitions."""

import time
from typing import Dict, Tuple

import numpy as np

from .features import DOMAINS, adjacency_matrix, object_domains


def class_count(labels) -> int:
    labels = np.asarray(labels)
    return int(np.unique(labels).size) if labels.size else 0


def same_class_pairs(labels) -> int:
    labels = np.asarray(labels)
    if not labels.size:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def cross_pairs(total_pairs: int, labels) -> int:
    return int(total_pairs) - same_class_pairs(labels)


def orbit_stratum(orbits: int, objects: int) -> str:
    if orbits == 1:
        return 'degenerate'
    if orbits == objects:
        return 'trivial'
    return 'intermediate'


def orbit_reference(mis_size: int, orbits: int, objects: int) -> dict:
    """Compatibility fields for the legacy auxiliary MIS statistic."""
    return {'orbits': orbits,
            'orbit_ceiling_ratio': orbits / objects if objects else None,
            'mis_orbit_ratio': mis_size / orbits if orbits else None,
            'mis_exceeds_orbits': mis_size > orbits}


def domain_orbits(generators, u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    count = len(u)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    lookup = np.full(n * n, -1, dtype=np.int64)
    lookup[u.astype(np.int64) * n + v.astype(np.int64)] = np.arange(count)
    parent = list(range(count))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for generator in generators:
        permutation = np.asarray(generator, dtype=np.int64)
        gu, gv = permutation[u], permutation[v]
        image = lookup[np.minimum(gu, gv) * n + np.maximum(gu, gv)]
        if np.any(image < 0):
            raise ValueError('automorphism moved an object outside its domain')
        for source, target in enumerate(image.tolist()):
            source_root, target_root = find(source), find(target)
            if source_root != target_root:
                parent[source_root] = target_root
    roots = np.array([find(i) for i in range(count)], dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


def graph_orbits(edge_index: np.ndarray, n: int) -> Tuple[Dict[str, np.ndarray], float]:
    if n == 0:
        return {kind: np.zeros(0, dtype=np.int64) for kind in DOMAINS}, 0.0
    import pynauty
    started = time.perf_counter()
    adjacency = adjacency_matrix(edge_index, n)
    graph = pynauty.Graph(
        n, directed=False,
        adjacency_dict={i: np.flatnonzero(adjacency[i]).tolist() for i in range(n)})
    generators, _, _, _, vertex_orbits = pynauty.autgrp(graph)
    labels = {kind: domain_orbits(generators, u, v, n)
              for kind, (u, v) in object_domains(adjacency).items()}
    if class_count(labels['nodes']) != vertex_orbits:
        raise ValueError('vertex orbit closure disagrees with nauty')
    return labels, time.perf_counter() - started


def one_wl_colours(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency)
    if not len(adjacency):
        return np.zeros(0, dtype=np.int64)
    neighbours = [np.flatnonzero(adjacency[i]) for i in range(len(adjacency))]
    colours = np.array([len(nodes) for nodes in neighbours], dtype=np.int64)
    while True:
        signatures = [(int(colours[i]), tuple(sorted(colours[neighbours[i]].tolist())))
                      for i in range(len(adjacency))]
        dense = {signature: i for i, signature in enumerate(sorted(set(signatures)))}
        updated = np.array([dense[signature] for signature in signatures])
        if np.unique(updated).size == np.unique(colours).size:
            return updated
        colours = updated


def refines(finer, coarser) -> bool:
    finer = np.asarray(finer).reshape(-1)
    coarser = np.asarray(coarser).reshape(-1)
    if finer.size != coarser.size:
        raise ValueError('label vectors must have equal length')
    return all(np.unique(coarser[finer == label]).size == 1
               for label in np.unique(finer))


def fwl2_colours(adjacency: np.ndarray) -> np.ndarray:
    """Stable 2-FWL colours from the standard coordinate replacements.

    For tuple (i, j), replacing its first and second coordinates by the shared
    pivot w gives (w, j) and (i, w).  The update therefore uses the joint
    multiset ``(c[w,j], c[i,w])`` rather than two independently aggregated
    coordinate multisets.
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    colours = np.full((n, n), 2, dtype=np.int64)
    np.fill_diagonal(colours, 0)
    colours[adjacency > 0] = 1
    block = max(1, 2_000_000 // max(n * n, 1))
    while True:
        before = np.unique(colours).size
        count = int(colours.max()) + 1
        signatures = np.empty((n * n, n + 1), dtype=np.int64)
        for start in range(0, n, block):
            stop = min(n, start + block)
            # Shape: (i block, j, w).  colours.T[w, j] supplies c[w,j],
            # while colours[i,w] supplies c[i,w].
            joint = (colours.T[None, :, :] * count
                     + colours[start:stop, None, :])
            joint.sort(axis=-1)
            signatures[start * n:stop * n, 0] = colours[start:stop].reshape(-1)
            signatures[start * n:stop * n, 1:] = joint.reshape((stop - start) * n, n)
        colours = np.unique(signatures, axis=0, return_inverse=True)[1].reshape(n, n)
        if np.unique(colours).size == before:
            return colours


def project_colours(colours: np.ndarray, adjacency: np.ndarray, kind: str) -> np.ndarray:
    if kind == 'nodes':
        indexes = np.arange(len(adjacency))
        raw = colours[indexes, indexes][:, None]
    else:
        u, v = object_domains(adjacency)[kind]
        raw = np.stack((np.minimum(colours[u, v], colours[v, u]),
                        np.maximum(colours[u, v], colours[v, u])), axis=1)
    if not raw.shape[0]:
        return np.zeros(0, dtype=np.int64)
    return np.unique(raw, axis=0, return_inverse=True)[1].astype(np.int64)


def wl_domain_labels(adjacency: np.ndarray, kind: str) -> np.ndarray:
    return project_colours(fwl2_colours(adjacency), adjacency, kind)
