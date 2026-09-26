"""Graph validation and structural feature construction."""

from typing import Dict, FrozenSet, List, Tuple

import numpy as np

DOMAINS = ('nodes', 'edges', 'pairs')
ROUTES = ('powers', 'endpoints', 'mixed', 'powers_2d', 'endpoints_2d')
WIDE_ROUTES: FrozenSet[str] = frozenset({'mixed', 'powers_2d', 'endpoints_2d'})
AUGMENTED_ROUTES: FrozenSet[str] = frozenset({'mixed', 'powers_2d'})


def channel_plan(kind: str, width: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return ordered power degrees and unordered endpoint degree pairs."""
    first = 2 if kind == 'nodes' else 1
    power_degrees = list(range(first, first + width))
    endpoint_degrees = []
    total = 1
    while len(endpoint_degrees) < width:
        endpoint_degrees.extend((a, total - a) for a in range(total + 1)
                                if a <= total - a)
        total += 1
    return power_degrees, endpoint_degrees[:width]


def adjacency_matrix(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Build a dense adjacency matrix and enforce the protocol input contract."""
    edge_index = np.asarray(edge_index)
    if edge_index.shape[0] != 2:
        raise ValueError('edge_index must have shape (2, edges)')
    if np.any(edge_index < 0) or np.any(edge_index >= n):
        raise ValueError('edge_index contains a vertex outside [0, n)')
    adjacency = np.zeros((n, n), dtype=np.float32)
    np.add.at(adjacency, (edge_index[0], edge_index[1]), 1.0)
    if (np.any(np.diag(adjacency)) or np.any(adjacency > 1)
            or not np.array_equal(adjacency, adjacency.T)):
        raise ValueError('Expected a simple undirected graph with both edge orientations')
    return adjacency


def upper_triangle(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, columns = np.triu_indices(len(mask), k=1)
    keep = mask[rows, columns]
    return rows[keep], columns[keep]


def object_domains(adjacency: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = len(adjacency)
    return {
        'nodes': (np.arange(n), np.arange(n)),
        'edges': upper_triangle(adjacency > 0),
        'pairs': upper_triangle(~np.eye(n, dtype=bool)),
    }


def as_orientations(features) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        return values[:, None, :]
    if values.ndim == 3 and values.shape[1] in (1, 2):
        return values
    raise ValueError('features must have shape (objects, channels) or '
                     '(objects, 1|2, channels)')


def orientation_pairs(orientations: int):
    """Legacy helper retained for callers; strict matching no longer ORs these."""
    if orientations <= 1:
        return ((0, 0),)
    return tuple((left, right) for left in range(orientations)
                 for right in range(orientations))


def slice_channels(features, count: int):
    return features[..., :count]


def mixed_block(powers, endpoints, width: int):
    return np.concatenate((slice_channels(powers, width),
                           slice_channels(endpoints, width)), axis=-1)


def select_route(route: str, powers, endpoints, width: int):
    if route == 'powers':
        return slice_channels(powers, width)
    if route == 'endpoints':
        return slice_channels(endpoints, width)
    if route == 'mixed':
        return mixed_block(powers, endpoints, width)
    if route == 'powers_2d':
        return slice_channels(powers, 2 * width)
    if route == 'endpoints_2d':
        return slice_channels(endpoints, 2 * width)
    raise ValueError(f'unknown route {route!r}')


def graph_features(edge_index: np.ndarray, n: int, width: int, *,
                   power_width=None, endpoint_width=None,
                   complete_budget=False, domains=DOMAINS):
    """Construct power and endpoint features for all three object domains."""
    adjacency = adjacency_matrix(edge_index, n)
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros(n, dtype=np.float32)
    inverse_sqrt[degree > 0] = degree[degree > 0] ** -0.5
    normalized = adjacency * inverse_sqrt[:, None] * inverse_sqrt[None, :]

    if complete_budget:
        if power_width is None or endpoint_width is None:
            raise ValueError('Complete-budget features require both channel counts')
        plans = {
            'nodes': (channel_plan('nodes', max(1, power_width - 1))[0],
                      channel_plan('nodes', endpoint_width)[1]),
            'edges': (channel_plan('edges', power_width)[0],
                      channel_plan('edges', endpoint_width)[1]),
            'pairs': (channel_plan('pairs', power_width)[0],
                      channel_plan('pairs', endpoint_width)[1]),
        }
    else:
        plans = {kind: channel_plan(kind, 2 * width) for kind in DOMAINS}
    matrix_powers = [np.eye(n, dtype=np.float32)]
    max_power = max(max(plan[0]) for plan in plans.values())
    for _ in range(max_power):
        matrix_powers.append(matrix_powers[-1] @ normalized)

    max_endpoint = max(max(a, b) for _, endpoints in plans.values()
                       for a, b in endpoints)
    propagated_ones = [np.ones(n, dtype=np.float32)]
    for _ in range(max_endpoint):
        propagated_ones.append(normalized @ propagated_ones[-1])
    propagated_ones = np.stack(propagated_ones, axis=1)

    result = {}
    for kind, (u, v) in object_domains(adjacency).items():
        if kind not in domains:
            continue
        powers, endpoints = plans[kind]
        forward_power = np.stack([matrix_powers[k][u, v] for k in powers], axis=1)
        forward_endpoint = np.stack(
            [propagated_ones[u, a] * propagated_ones[v, b]
             for a, b in endpoints], axis=1)
        if kind == 'nodes':
            result[kind] = (forward_power[:, None, :],
                            forward_endpoint[:, None, :])
            continue
        reverse_power = np.stack([matrix_powers[k][v, u] for k in powers], axis=1)
        reverse_endpoint = np.stack(
            [propagated_ones[v, a] * propagated_ones[u, b]
             for a, b in endpoints], axis=1)
        result[kind] = (np.stack((forward_power, reverse_power), axis=1),
                        np.stack((forward_endpoint, reverse_endpoint), axis=1))
    return result


def standardize_columns(features, atol: float, rtol: float):
    """Z-score channels jointly over objects and orientations."""
    original = np.asarray(features, dtype=np.float32)
    values = as_orientations(original)
    if values.shape[0] == 0 or values.shape[2] == 0:
        return original.copy()
    flat = values.reshape(-1, values.shape[2])
    centered = flat - flat.mean(axis=0)
    spread = centered.max(axis=0) - centered.min(axis=0)
    band = atol + rtol * np.maximum(np.abs(flat.max(axis=0)),
                                    np.abs(flat.min(axis=0)))
    informative = spread > band
    scaled = centered.copy()
    scaled[:, informative] /= scaled[:, informative].std(axis=0)
    scaled[:, ~informative] = 0.0
    scaled = scaled.reshape(values.shape)
    return scaled[:, 0, :] if original.ndim == 2 else scaled
