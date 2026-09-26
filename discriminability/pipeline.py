"""Dataset loading and per-graph protocol orchestration."""

import json
import os
import platform
import resource
import time

import numpy as np

from .evaluation import difference_details, evaluate_domain
from .features import (adjacency_matrix, graph_features, object_domains,
                       standardize_columns)
from .references import fwl2_colours, graph_orbits, project_colours

_CONFIG = {}


def degree_route_specs(kind, q, routes):
    """Channel counts for complete strict-degree dictionaries at Q."""
    power_width = max(1, q - 1) if kind == 'nodes' else q
    endpoint_width = sum(total // 2 + 1 for total in range(1, q))
    counts = {
        'powers': (power_width, 0),
        'endpoints': (0, endpoint_width),
        'mixed': (power_width, endpoint_width),
    }
    return {route: counts[route] for route in routes}


def init_worker(config: dict) -> None:
    global _CONFIG
    _CONFIG = config
    from threadpoolctl import threadpool_limits
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=config.get('threads_per_worker', 1))


def _aligned_relabelled_features(edge_index, n, width, permutation, *,
                                 power_width=None, endpoint_width=None,
                                 complete_budget=False, domains=("nodes","edges","pairs")):
    """Relabel a graph, then map domain objects and orientations back to source order."""
    relabelled_edges = permutation[edge_index]
    generated = graph_features(relabelled_edges, n, width,
                               power_width=power_width,
                               endpoint_width=endpoint_width,
                               complete_budget=complete_budget, domains=domains)
    original_domains = object_domains(adjacency_matrix(edge_index, n))
    relabelled_domains = object_domains(adjacency_matrix(relabelled_edges, n))
    aligned = {}
    for kind, blocks in generated.items():
        new_u, new_v = relabelled_domains[kind]
        lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(new_u, new_v))}
        old_u, old_v = original_domains[kind]
        indexes, swaps = [], []
        for a, b in zip(old_u, old_v):
            pa, pb = int(permutation[a]), int(permutation[b])
            key = (pa, pb) if pa <= pb else (pb, pa)
            indexes.append(lookup[key])
            swaps.append(pa > pb)
        domain_blocks = []
        for block in blocks:
            values = block[np.asarray(indexes, dtype=np.int64)].copy()
            if values.shape[1] == 2:
                swap_indexes = np.flatnonzero(swaps)
                values[swap_indexes] = values[swap_indexes, ::-1, :]
            domain_blocks.append(values)
        aligned[kind] = tuple(domain_blocks)
    return aligned


def bounded_node_distance_mask(adjacency, maximum):
    """Return node-pair eligibility for a shortest-path distance cutoff."""
    n = len(adjacency)
    limit = int(maximum)
    distances = np.full((n, n), limit + 1, dtype=np.int32)
    neighbours = [np.flatnonzero(adjacency[i]) for i in range(n)]
    for source in range(n):
        distances[source, source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            next_distance = distances[source, node] + 1
            if next_distance > limit:
                continue
            for target in neighbours[node]:
                if distances[source, target] > next_distance:
                    distances[source, target] = next_distance
                    queue.append(int(target))
    eligible = distances <= limit
    np.fill_diagonal(eligible, False)
    return eligible


def worker(task):
    index, split, edge_index, n = task
    config = _CONFIG
    started = time.perf_counter()
    orbit_labels, orbit_seconds = graph_orbits(edge_index, n)
    adjacency = adjacency_matrix(edge_index, n)
    domains = object_domains(adjacency)
    wl_colours = fwl2_colours(adjacency)
    max_compare_distance = config.get('max_compare_distance')
    node_distance_mask = (bounded_node_distance_mask(adjacency, max_compare_distance)
                          if max_compare_distance is not None else None)

    feature_started = time.perf_counter()
    degree_budgets = config.get('degree_budgets')
    if degree_budgets:
        max_endpoint_width = sum(total // 2 + 1
                                 for total in range(1, max(degree_budgets)))
        power_width = max(degree_budgets)
        endpoint_width = max_endpoint_width
        width = max(power_width, endpoint_width)
        complete_budget = True
    else:
        width = max(config['dims'])
        power_width = endpoint_width = None
        complete_budget = False
    features = graph_features(edge_index, n, width,
                              power_width=power_width,
                              endpoint_width=endpoint_width,
                              complete_budget=complete_budget,
                              domains=config.get("domains", ("nodes","edges","pairs")))
    rng = np.random.default_rng(config['seed'] + index)
    permuted = [_aligned_relabelled_features(edge_index, n, width,
                                             rng.permutation(n),
                                             power_width=power_width,
                                             endpoint_width=endpoint_width,
                                             complete_budget=complete_budget,
                                             domains=config.get("domains", ("nodes","edges","pairs")))
                for _ in range(config['permutation_checks'])]
    if config.get('standardize'):
        def transform(blocks):
            return tuple(standardize_columns(block, config['atol'], config['rtol'])
                         for block in blocks)

        features = {kind: transform(blocks) for kind, blocks in features.items()}
        permuted = [{kind: transform(blocks) for kind, blocks in sample.items()}
                    for sample in permuted]
    feature_seconds = time.perf_counter() - feature_started

    rows, differences, details = [], [], []
    for kind, (powers, endpoints) in features.items():
        wl_labels = project_colours(wl_colours, adjacency, kind)
        conditions = ([(q, q) for q in degree_budgets] if degree_budgets else
                      [(None, d) for d in config['dims']])
        domain_rows, domain_differences, masks = [], [], {}
        for q, branch_dim in conditions:
            route_specs = (None if q is None else
                           {q: degree_route_specs(kind, q, config['routes'])})
            current_rows, current_differences, current_masks = evaluate_domain(
                kind, powers, endpoints, [branch_dim], orbit_labels[kind], wl_labels,
                config['atol'], config['rtol'], config['chunk_size'], config['budget'],
                [sample[kind] for sample in permuted], routes=config['routes'],
                compute_mis=not config.get('no_mis', False), compact_masks=True,
                route_specs=route_specs,
                comparison_mask=(node_distance_mask if kind == 'nodes' else None),
                max_compare_distance=(max_compare_distance if kind == 'nodes' else None))
            if q is not None:
                for row in current_rows + current_differences:
                    row['degree_budget'] = q
            domain_rows.extend(current_rows)
            domain_differences.extend(current_differences)
            masks.update(current_masks)
        for row in domain_rows:
            row.update(graph_index=index, split=split)
        for row in domain_differences:
            row.update(graph_index=index, split=split)
        rows.extend(domain_rows)
        differences.extend(domain_differences)
        domain_details = []
        if config.get('pair_details', 'differences') != 'none':
            count = len(powers)
            for width, packed in masks.items():
                unpacked = {route: np.unpackbits(mask, count=count*count)
                            .reshape(count, count).astype(bool)
                            for route, mask in packed.items()}
                domain_details.extend(difference_details(
                    index, split, kind, domains[kind], orbit_labels[kind], wl_labels,
                    {width: unpacked}, config.get('pair_details', 'differences'),
                    comparison_mask=(node_distance_mask if kind == 'nodes' else None)))
        audits = {(row['branch_dim'], row.get('comparison', row.get('encoding'))): row
                  for row in domain_rows + domain_differences}
        for detail in domain_details:
            audit = audits[(detail['branch_dim'], detail['comparison'])]
            detail.update(reliable=audit['reliable'],
                          reliability_failures=audit['reliability_failures'])
        details.extend(domain_details)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        peak_rss *= 1024
    timing = {'graph_index': index, 'orbit_seconds': orbit_seconds,
              'feature_seconds': feature_seconds,
              'total_seconds': time.perf_counter() - started,
              'worker_peak_rss_bytes': int(peak_rss),
              'worker_pid': os.getpid(),
              'memory_measurement': 'worker-process high-water RSS; cumulative across its assigned graphs'}
    return rows, differences, details, timing


def load_zinc(root: str, full: bool, limit: int):
    from torch_geometric.datasets import ZINC
    index = 0
    for split in ('train', 'val', 'test'):
        for data in ZINC(root=root, subset=not full, split=split):
            if limit and index >= limit:
                return
            yield index, split, data.edge_index.numpy().astype(np.int64), int(data.num_nodes)
            index += 1


def load_npz(path: str, limit: int):
    data = np.load(path, allow_pickle=True)
    edges, counts, offsets = data['edges'], data['counts'], data['offsets']
    splits = data['splits'] if 'splits' in data else np.array(['all'] * len(counts))
    for index, count in enumerate(counts):
        if limit and index >= limit:
            return
        start, stop = int(offsets[index]), int(offsets[index + 1])
        yield index, str(splits[index]), edges[:, start:stop].astype(np.int64), int(count)


def topology_provenance(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return json.loads(str(data['provenance'])) if 'provenance' in data else {}


def load_graphs(dataset: str, root: str, full: bool, limit: int):
    if dataset == 'zinc':
        return load_zinc(root, full, limit)
    if dataset == 'npz':
        return load_npz(root, limit)
    raise ValueError(f'unknown dataset family: {dataset}')
