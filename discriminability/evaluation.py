"""Strict unordered-object comparison and pair-set accounting."""

from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from .features import ROUTES, WIDE_ROUTES, as_orientations, select_route
from .references import class_count


def _close(left, right, atol: float, rtol: float) -> np.ndarray:
    difference = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    return difference <= atol + rtol * scale


def similarity_blocks(left, right, atol: float, rtol: float,
                      stops: Sequence[int]) -> Dict[int, np.ndarray]:
    """Compare complete unordered encodings at each requested channel count.

    A two-orientation object is an unordered pair of vectors.  Equality therefore
    requires a bijective match: forward/forward AND reverse/reverse, or the one
    global swap forward/reverse AND reverse/forward.  Sharing just one oriented
    vector is insufficient.
    """
    left, right = as_orientations(left), as_orientations(right)
    if left.shape[1] != right.shape[1]:
        raise ValueError('both blocks must have the same orientation count')
    requested = sorted({int(stop) for stop in stops})
    if not requested or requested[0] < 1:
        raise ValueError('channel stops must be positive')
    top = min(requested[-1], left.shape[2], right.shape[2])
    if top < requested[-1]:
        raise ValueError('requested more channels than the feature block contains')
    shape = (left.shape[0], right.shape[0])
    result = {stop: np.zeros(shape, dtype=bool) for stop in requested}
    orientations = left.shape[1]
    if orientations == 1:
        direct = np.ones(shape, dtype=bool)
        for channel in range(top):
            direct &= _close(left[:, 0, channel, None],
                             right[None, :, 0, channel], atol, rtol)
            if channel + 1 in result:
                result[channel + 1] = direct.copy()
        return result
    if orientations != 2:
        raise ValueError('strict unordered comparison supports one or two orientations')

    ff = np.ones(shape, dtype=bool)
    rr = np.ones(shape, dtype=bool)
    fr = np.ones(shape, dtype=bool)
    rf = np.ones(shape, dtype=bool)
    for channel in range(top):
        ff &= _close(left[:, 0, channel, None], right[None, :, 0, channel], atol, rtol)
        rr &= _close(left[:, 1, channel, None], right[None, :, 1, channel], atol, rtol)
        fr &= _close(left[:, 0, channel, None], right[None, :, 1, channel], atol, rtol)
        rf &= _close(left[:, 1, channel, None], right[None, :, 0, channel], atol, rtol)
        if channel + 1 in result:
            result[channel + 1] = (ff & rr) | (fr & rf)
    return result


pair_similarity = similarity_blocks


def separation_mask(features, atol: float, rtol: float, chunk: int) -> np.ndarray:
    """Return a symmetric mask whose off-diagonal true entries are separated."""
    features = as_orientations(features)
    count, channels = features.shape[0], features.shape[2]
    separated = np.zeros((count, count), dtype=bool)
    for row in range(0, count, chunk):
        for column in range(row, count, chunk):
            similar = similarity_blocks(
                features[row:row + chunk], features[column:column + chunk],
                atol, rtol, {channels})[channels]
            block = ~similar
            separated[row:row + chunk, column:column + chunk] = block
            separated[column:column + chunk, row:row + chunk] = block.T
    np.fill_diagonal(separated, False)
    return separated


def class_pair_masks(labels) -> Tuple[np.ndarray, np.ndarray]:
    """Upper-triangle masks for pairs inside and across reference classes."""
    labels = np.asarray(labels).reshape(-1)
    upper = np.triu(np.ones((labels.size, labels.size), dtype=bool), k=1)
    same = upper & (labels[:, None] == labels[None, :])
    return same, upper & ~same


def shared_route_masks(powers, endpoints, dims, routes, atol, rtol, chunk,
                       route_specs=None):
    """Traverse each feature family once; store masks compactly between uses."""
    if route_specs is not None:
        packed = {}
        for width in dims:
            for route in routes:
                power_width, endpoint_width = route_specs[width][route]
                if route == 'powers':
                    features = powers[..., :power_width]
                elif route == 'endpoints':
                    features = endpoints[..., :endpoint_width]
                elif route == 'mixed':
                    features = np.concatenate((powers[..., :power_width],
                                               endpoints[..., :endpoint_width]), axis=-1)
                else:
                    raise ValueError(f'Route {route!r} is unsupported in degree-budget mode')
                mask = separation_mask(features, atol, rtol, chunk)
                packed[width, route] = np.packbits(mask, axis=None)
        return packed
    families = {}
    aliases = {}
    for width in dims:
        for route in routes:
            family = 'powers' if route.startswith('powers') else 'endpoints'
            stop = width * (2 if route.endswith('_2d') else 1)
            if route == 'mixed':
                family, stop = 'mixed', 2 * width
            aliases[width, route] = family, stop
            families.setdefault(family, set()).add(stop)
    count = len(powers)
    packed = {}
    for family, stops in families.items():
        # Interleaving preserves a common orientation assignment for mixed.
        features = (np.stack((powers, endpoints), axis=-1).reshape(
            *powers.shape[:-1], 2 * powers.shape[-1]) if family == 'mixed'
            else powers if family == 'powers' else endpoints)
        masks = {stop: np.zeros((count, count), dtype=bool) for stop in stops}
        for i in range(0, count, chunk):
            for j in range(i, count, chunk):
                blocks = similarity_blocks(features[i:i+chunk], features[j:j+chunk],
                                           atol, rtol, stops)
                for stop, similar in blocks.items():
                    masks[stop][i:i+chunk, j:j+chunk] = ~similar
                    masks[stop][j:j+chunk, i:i+chunk] = (~similar).T
        for stop, mask in masks.items():
            np.fill_diagonal(mask, False)
            packed[family, stop] = np.packbits(mask, axis=None)
    return {key: packed[value] for key, value in aliases.items()}


def _count(mask) -> int:
    return int(np.count_nonzero(mask))


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def partition_metrics(separated: np.ndarray, orbit_labels, wl_labels,
                      comparison_mask=None) -> dict:
    """Measure pair-set coverage and forbidden within-class separations."""
    upper = np.triu(np.ones_like(separated, dtype=bool), k=1)
    if comparison_mask is not None:
        upper &= comparison_mask
    separated = separated & upper
    same_orbit, cross_orbit = class_pair_masks(orbit_labels)
    same_wl, cross_wl = class_pair_masks(wl_labels)
    same_orbit &= upper
    cross_orbit &= upper
    same_wl &= upper
    cross_wl &= upper
    covered_orbit = _count(separated & cross_orbit)
    covered_wl = _count(separated & cross_wl)
    missed_orbit = _count(~separated & cross_orbit)
    missed_wl = _count(~separated & cross_wl)
    orbit_violations = _count(separated & same_orbit)
    wl_violations = _count(separated & same_wl)
    cross_orbit_count, cross_wl_count = _count(cross_orbit), _count(cross_wl)
    same_orbit_count, same_wl_count = _count(same_orbit), _count(same_wl)
    return {
        'total_pairs': _count(upper),
        'separated_pairs': _count(separated),
        'cross_orbit_pairs': cross_orbit_count,
        'cross_wl_pairs': cross_wl_count,
        'covered_orbit_pairs': covered_orbit,
        'covered_wl_pairs': covered_wl,
        'missed_orbit_pairs': missed_orbit,
        'missed_wl_pairs': missed_wl,
        'same_orbit_pairs': same_orbit_count,
        'same_wl_pairs': same_wl_count,
        'orbit_violation_pairs': orbit_violations,
        'wl_violation_pairs': wl_violations,
        'coverage_orbit': _ratio(covered_orbit, cross_orbit_count),
        'coverage_wl': _ratio(covered_wl, cross_wl_count),
        'wl_orbit_pair_ceiling': _ratio(cross_wl_count, cross_orbit_count),
        'orbit_violation_rate': _ratio(orbit_violations, same_orbit_count),
        'wl_violation_rate': _ratio(wl_violations, same_wl_count),
        'orbit_partition_reproduced': (missed_orbit == 0 and orbit_violations == 0) if len(orbit_labels) else None,
        'wl_partition_reproduced': (missed_wl == 0 and wl_violations == 0) if len(wl_labels) else None,
        'exact_reliability_pass': orbit_violations == 0 and wl_violations == 0,
        # Compatibility names now use the mathematically valid intersection.
        'completeness_orbit': _ratio(covered_orbit, cross_orbit_count),
        'completeness_wl': _ratio(covered_wl, cross_wl_count),
        'separation_rate': _ratio(_count(separated), _count(upper)),
    }


def maximum_independent_set(sim: np.ndarray, budget: int, return_vertices=False):
    """Return size (or a feasible witness), optimality flag, and search cost."""
    count = sim.shape[0]
    if count == 0:
        return ([] if return_vertices else 0), True, 0
    neighbours = [np.flatnonzero(sim[i]) for i in range(count)]
    state = {'expanded': 0, 'exact': True}
    memo = {}

    def component(start: int, pending: set) -> List[int]:
        pending.discard(start)
        stack, order = [start], [start]
        while stack:
            node = stack.pop()
            for other in neighbours[node]:
                if other in pending:
                    pending.remove(other)
                    order.append(other)
                    stack.append(other)
        return order

    def greedy(vertices: FrozenSet[int]):
        chosen, remaining = [], set(vertices)
        while remaining:
            pivot = min(remaining, key=lambda x: len(set(neighbours[x]) & remaining))
            remaining.remove(pivot)
            remaining -= set(neighbours[pivot])
            chosen.append(int(pivot))
        return chosen

    def solve(vertices: FrozenSet[int]):
        if not vertices:
            return []
        if all(len(set(neighbours[v]) & vertices) == len(vertices) - 1 for v in vertices):
            return [int(min(vertices))]
        if vertices in memo:
            return memo[vertices]
        state['expanded'] += 1
        if state['expanded'] > budget:
            state['exact'] = False
            return greedy(vertices)
        pending, parts = set(vertices), []
        while pending:
            parts.append(frozenset(component(pending.pop(), pending)))
        if len(parts) > 1:
            value = [v for part in parts for v in solve(part)]
        else:
            pivot = max(vertices, key=lambda x: len(set(neighbours[x]) & vertices))
            blocked = set(neighbours[pivot])
            include = [int(pivot)] + solve(frozenset(vertices - {pivot} - blocked))
            exclude = solve(frozenset(vertices - {pivot}))
            value = include if len(include) >= len(exclude) else exclude
        memo[vertices] = value
        return value

    total, pending = [], set(range(count))
    for start in range(count):
        if start in pending:
            total += solve(frozenset(component(start, pending)))
    return (sorted(total) if return_vertices else len(total)), state['exact'], state['expanded']


def mis_measure(features, atol: float, rtol: float, chunk: int, budget: int) -> dict:
    separated = separation_mask(features, atol, rtol, chunk)
    similarity = ~separated
    np.fill_diagonal(similarity, False)
    size, exact, expanded = maximum_independent_set(similarity, budget)
    fraction = size / len(features) if len(features) else None
    return {'representative_count': size,
            'representative_count_exact': exact,
            'representative_search_nodes': expanded,
            'representative_fraction': fraction if exact else None,
            'representative_fraction_lower_bound': fraction,
            # Compatibility aliases.
            'mis_size': size, 'mis_exact': exact, 'mis_search_nodes': expanded,
            'mis_ratio': fraction if exact else None}


def compare_objects(powers, endpoints, dims: Sequence[int], atol: float,
                    rtol: float, chunk: int) -> Dict[int, Dict[str, int]]:
    """Compatibility count API backed by the strict separation masks."""
    result = {}
    for width in dims:
        masks = {route: separation_mask(select_route(route, powers, endpoints, width),
                                        atol, rtol, chunk)
                 for route in ROUTES}
        upper = np.triu(np.ones_like(masks['powers']), k=1).astype(bool)
        seed_collisions = upper & ~masks['powers']
        result[width] = {
            'total_pairs': _count(upper),
            'seed_collisions': _count(seed_collisions),
            **{route: _count(mask & upper) for route, mask in masks.items()},
            'powers_2d_resolved': _count(seed_collisions & masks['powers_2d']),
            'mixed_resolved': _count(seed_collisions & masks['mixed']),
            'endpoint_only_pairs': _count(upper & masks['mixed'] & ~masks['powers_2d']),
            'powers_only_pairs': _count(upper & masks['powers_2d'] & ~masks['mixed']),
        }
    return result


def _difference_counts(first, second, theoretical) -> dict:
    return {
        'first_only': _count(theoretical & first & ~second),
        'second_only': _count(theoretical & second & ~first),
        'shared': _count(theoretical & first & second),
        'missed_by_both': _count(theoretical & ~first & ~second),
    }


def evaluate_domain(kind: str, powers, endpoints, dims: Sequence[int],
                    orbit_labels, wl_labels, atol: float, rtol: float,
                    chunk: int, budget: int, permuted=(), routes=ROUTES,
                    compute_mis=True, compact_masks=False, route_specs=None,
                    comparison_mask=None, max_compare_distance=None):
    """Evaluate every route and return rows plus auditable route differences."""
    objects = len(powers)
    same_orbit, theoretical = class_pair_masks(orbit_labels)
    eligible = np.triu(np.ones((objects, objects), dtype=bool), k=1)
    if comparison_mask is not None:
        eligible &= comparison_mask
    theoretical &= eligible
    rows, differences, route_masks = [], [], {}
    packed = shared_route_masks(powers, endpoints, dims, routes, atol, rtol, chunk,
                                route_specs)
    perm_packed = [shared_route_masks(p, e, dims, routes, atol, rtol, chunk)
                   for p, e in permuted] if route_specs is None else [
                       shared_route_masks(p, e, dims, routes, atol, rtol, chunk,
                                          route_specs) for p, e in permuted]
    mis_cache = {}
    for width in dims:
        masks = {route: np.unpackbits(packed[width, route], count=objects*objects)
                 .reshape(objects, objects).astype(bool) for route in routes}
        route_masks[width] = ({route: packed[width, route] for route in routes}
                              if compact_masks else masks)
        seed_collisions = (eligible & ~masks['powers']
                           if 'powers' in masks else None)
        direct_gain = (seed_collisions & masks['mixed']
                       if seed_collisions is not None and 'mixed' in masks else None)
        matched = (_difference_counts(masks['mixed'], masks['powers_2d'], theoretical)
                   if {'mixed', 'powers_2d'} <= masks.keys() else None)
        standalone = (_difference_counts(masks['endpoints'], masks['powers'], theoretical)
                      if {'endpoints', 'powers'} <= masks.keys() else None)
        for comparison, first, second in (
                ('endpoints_vs_powers', 'endpoints', 'powers'),
                ('mixed_vs_powers', 'mixed', 'powers'),
                ('mixed_vs_powers_2d', 'mixed', 'powers_2d'),
                ('endpoints_2d_vs_powers_2d', 'endpoints_2d', 'powers_2d')):
            if first not in masks or second not in masks:
                continue
            differences.append({'items': kind, 'branch_dim': width,
                                'comparison': comparison,
                                **_difference_counts(masks[first], masks[second],
                                                     theoretical)})
        for route, separated in masks.items():
            metrics = partition_metrics(separated, orbit_labels, wl_labels,
                                        comparison_mask)
            if compute_mis:
                similarity = ~separated
                if comparison_mask is not None:
                    similarity &= comparison_mask
                np.fill_diagonal(similarity, False)
                cache_key = id(packed[width, route])
                if cache_key not in mis_cache:
                    mis_cache[cache_key] = maximum_independent_set(
                        similarity, budget, return_vertices=True)
                representatives, mis_exact, search_nodes = mis_cache[cache_key]
                mis_size = len(representatives)
                if len(set(representatives)) != mis_size or np.any(
                        similarity[np.ix_(representatives, representatives)]):
                    raise ValueError('Invalid independent-set witness')
            else:
                mis_size = mis_exact = search_nodes = None
                representatives = None
            permutation_changes = 0
            for sample in perm_packed:
                permutation_changes += int(not np.array_equal(sample[width, route],
                                                               packed[width, route]))
            orbits = class_count(orbit_labels)
            wl_classes = class_count(wl_labels)
            augmented = route in ('mixed', 'powers_2d') and seed_collisions is not None
            resolved = _count(seed_collisions & separated) if augmented else None
            monotonicity = (_count(eligible & masks['powers'] & ~masks['mixed'])
                            if route in ('powers', 'mixed')
                            and {'powers', 'mixed'} <= masks.keys() else None)
            failures = []
            if metrics['orbit_violation_pairs']:
                failures.append('same_orbit_separation')
            if metrics['wl_violation_pairs']:
                failures.append('same_wl_separation')
            if permutation_changes:
                failures.append('relabelled_pair_mask_changed')
            if monotonicity:
                failures.append('augmentation_removed_separation')
            reliable = not failures
            representative_fraction = (mis_size / objects
                                       if mis_size is not None and objects else None)
            representative_capacity_raw = (mis_size / orbits
                                           if mis_size is not None and orbits else None)
            if route_specs is None:
                power_channels = width
                endpoint_channels = width
                total_channels = 2 * width if route in WIDE_ROUTES else width
                seed_channels = width
                added_channels = width if augmented else 0
            else:
                power_channels, endpoint_channels = route_specs[width][route]
                total_channels = power_channels + endpoint_channels
                seed_channels = power_channels if route == 'powers' else endpoint_channels
                added_channels = endpoint_channels if route == 'mixed' else 0
            rows.append({
                'items': kind, 'encoding': route, 'branch_dim': width,
                'max_compare_distance': max_compare_distance,
                'comparison_gate': ('node shortest-path distance <= maximum'
                                    if max_compare_distance is not None and kind == 'nodes'
                                    else 'all object pairs'),
                'eligible_comparison_pairs': _count(eligible),
                'channels': total_channels,
                'seed_channels': seed_channels, 'added_channels': added_channels,
                'orientations': powers.shape[1], 'objects': objects,
                'stored_scalars_per_object': powers.shape[1] * total_channels,
                'width_unit': ('complete degree-budget dictionary channels'
                               if route_specs is not None else
                               'selected matrix channels per orientation'),
                'orbits': orbits, 'wl_classes': wl_classes,
                'orbit_ceiling_ratio': orbits / objects if objects else None,
                'orbit_stratum': ('empty' if not objects else 'degenerate' if orbits == 1 else
                                  'trivial' if orbits == objects else 'intermediate'),
                **metrics,
                'pairwise_resolution': metrics['coverage_orbit'],
                'exact_reliability_pass': reliable,
                'seed_collisions': (_count(seed_collisions)
                                    if seed_collisions is not None else None),
                'resolved_pairs': resolved,
                'resolution_rate': _ratio(resolved, _count(seed_collisions))
                                   if resolved is not None else None,
                'endpoint_gain_pairs': (_count(direct_gain)
                                        if route == 'mixed' and direct_gain is not None
                                        else None),
                'endpoint_only_pairs': matched['first_only'] if matched else None,
                'powers_only_pairs': matched['second_only'] if matched else None,
                'standalone_endpoint_only_pairs': (standalone['first_only']
                                                   if standalone else None),
                'standalone_power_only_pairs': (standalone['second_only']
                                                if standalone else None),
                'mis_size': mis_size, 'mis_exact': mis_exact,
                'mis_search_nodes': search_nodes,
                'mis_ratio': representative_fraction if mis_exact else None,
                'mis_orbit_ratio': representative_capacity_raw if mis_exact else None,
                'mis_exceeds_orbits': (mis_size > orbits
                                       if mis_size is not None else None),
                'representative_count': mis_size,
                'representative_object_indices': representatives,
                'representative_count_exact': mis_exact,
                'representative_search_nodes': search_nodes,
                'representative_fraction': representative_fraction if mis_exact else None,
                'representative_fraction_lower_bound': representative_fraction,
                'representative_capacity': (representative_capacity_raw
                                            if mis_exact else None),
                'representative_capacity_lower_bound': representative_capacity_raw,
                'representative_ceiling_attained': bool(
                    reliable and mis_exact and mis_size is not None
                    and mis_size == orbits) if objects and mis_exact is True else None,
                # Preserved field, with its corrected partition meaning.
                'wl_ceiling_attained': metrics['wl_partition_reproduced'],
                'permutation_pair_mask_changes': permutation_changes,
                'permutation_checks': len(perm_packed),
                'monotonicity_violations': monotonicity,
                'reliable': reliable,
                'reliability_failures': failures,
            })
    indexed = {(row['branch_dim'], row['encoding']): row for row in rows}
    for row in rows:
        baseline = indexed.get((row['branch_dim'], 'powers'))
        row['baseline_comparison_reliable'] = (
            row['reliable'] and baseline['reliable']
            if row['encoding'] in ('mixed', 'powers_2d') and baseline else None)
    for difference in differences:
        first, second = difference['comparison'].split('_vs_')
        members = [indexed[(difference['branch_dim'], name)] for name in (first, second)]
        difference['reliable'] = all(row['reliable'] for row in members)
        difference['reliability_failures'] = [
            f"{row['encoding']}:{failure}" for row in members
            for failure in row['reliability_failures']]
    return rows, differences, route_masks


def difference_details(graph_index: int, split: str, kind: str, objects,
                       orbit_labels, wl_labels, route_masks, mode='differences',
                       comparison_mask=None):
    """Materialize auditable object-pair identities for route-set differences."""
    if mode == 'none':
        return []
    u, v = objects
    _, theoretical = class_pair_masks(orbit_labels)
    if comparison_mask is not None:
        theoretical &= np.triu(comparison_mask, k=1)
    comparisons = (
        ('endpoints_vs_powers', 'endpoints', 'powers'),
        ('mixed_vs_powers', 'mixed', 'powers'),
        ('mixed_vs_powers_2d', 'mixed', 'powers_2d'),
        ('endpoints_2d_vs_powers_2d', 'endpoints_2d', 'powers_2d'),
    )
    details = []
    for width, masks in route_masks.items():
        for comparison, first_name, second_name in comparisons:
            if first_name not in masks or second_name not in masks:
                continue
            first, second = masks[first_name], masks[second_name]
            relations = {
                'first_only': theoretical & first & ~second,
                'second_only': theoretical & second & ~first,
            }
            if mode == 'all':
                relations.update(shared=theoretical & first & second,
                                 missed_by_both=theoretical & ~first & ~second)
            for relation, mask in relations.items():
                for left, right in np.argwhere(np.triu(mask, k=1)):
                    details.append({
                        'graph_index': graph_index, 'split': split, 'items': kind,
                        'branch_dim': width, 'comparison': comparison,
                        'relation': relation,
                        'object_i_u': int(u[left]), 'object_i_v': int(v[left]),
                        'object_j_u': int(u[right]), 'object_j_v': int(v[right]),
                        'orbit_i': int(orbit_labels[left]),
                        'orbit_j': int(orbit_labels[right]),
                        'wl_i': int(wl_labels[left]), 'wl_j': int(wl_labels[right]),
                    })
        same_orbit, _ = class_pair_masks(orbit_labels)
        same_wl, _ = class_pair_masks(wl_labels)
        for route, separated in masks.items():
            for relation, mask in (
                    ('orbit_violation', same_orbit & separated),
                    ('wl_violation', same_wl & separated)):
                if comparison_mask is not None:
                    mask &= comparison_mask
                for left, right in np.argwhere(np.triu(mask, k=1)):
                    details.append({
                        'graph_index': graph_index, 'split': split, 'items': kind,
                        'branch_dim': width, 'comparison': route,
                        'relation': relation,
                        'object_i_u': int(u[left]), 'object_i_v': int(v[left]),
                        'object_j_u': int(u[right]), 'object_j_v': int(v[right]),
                        'orbit_i': int(orbit_labels[left]),
                        'orbit_j': int(orbit_labels[right]),
                        'wl_i': int(wl_labels[left]), 'wl_j': int(wl_labels[right]),
                    })
    return details
