"""Run and summarize object-domain precision, tolerance and unit experiments."""
from __future__ import annotations

import argparse
from itertools import product
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from .common import (Budget, ResourceLimit, atomic_json, atomic_npz, configuration,
                     hard_wall_limit, permutations, run_directory, sha256, topology)
from .object_comparison import (align_features, object_domains, build_object_features, reference_comparison,
                                route_features, separation_mask, object_metrics)


def graph_experiment(graph_id, adjacency, config, out):
    from discriminability.references import graph_orbits, fwl2_colours, project_colours
    edges = np.vstack(np.nonzero(adjacency))
    orbits, _ = graph_orbits(edges, len(adjacency))
    wl = fwl2_colours(adjacency)
    width = config['width']
    perm = permutations(len(adjacency), config['relabels'], config['seed'] + graph_id)
    arrays, masks, records, selected = {}, {}, [], set()
    metric_cache = {}
    for mode in config['dtypes']:
        original = build_object_features(adjacency, width, mode)
        changed = [align_features(adjacency, build_object_features(adjacency[np.ix_(p, p)], width, mode), p) for p in perm]
        for kind, blocks in original.items():
            arrays[f'{mode}_{kind}_powers'], arrays[f'{mode}_{kind}_endpoints'] = blocks
            labels = project_colours(wl, adjacency, kind)
            for route in config['routes']:
                values = route_features(blocks, route, width)
                count = len(values)
                controls = product(config['tolerance_factors'], config['unit_controls'])
                for factor, (scale, compensated) in controls:
                    atol = config['atol'] * factor * (abs(scale) if compensated else 1)
                    rtol = config['rtol'] * factor
                    key = f'{mode}_{kind}_{route}_t{factor}_s{scale}_c{int(compensated)}'
                    scaled = values * values.dtype.type(scale)
                    mask, margin = separation_mask(scaled, atol, rtol, config['chunk_size'], True)
                    masks[key] = mask
                    arrays[key + '_mask'] = np.packbits(mask)
                    changed_count = 0
                    for item in changed:
                        other = route_features(item[kind], route, width)
                        other_mask = separation_mask(other * other.dtype.type(scale), atol, rtol, config['chunk_size'])
                        changed_count += int(not np.array_equal(mask, other_mask))
                    cache_key = (kind, mask.tobytes())
                    if cache_key not in metric_cache:
                        metric_cache[cache_key] = object_metrics(mask, orbits[kind], labels, config)
                    metrics = dict(metric_cache[cache_key])
                    reliable = (metrics['orbit_violation_pairs'] == 0 and metrics['wl_violation_pairs'] == 0
                                and changed_count == 0 and count > 0)
                    records.append({'key': key, 'graph_id': graph_id, 'domain': kind,
                                    'route': route, 'dtype': mode, 'factor': factor,
                                    'scale': scale, 'compensated': compensated,
                                    'atol': atol, 'rtol': rtol, 'objects': count,
                                    'channels': values.shape[-1], 'reliable': reliable,
                                    'relabel_mask_changes': changed_count, **metrics})
                    if factor == 1 and scale == 1 and compensated and count:
                        band = config['boundary_epsilon_multiplier'] * np.finfo(mode).eps * max(1, float(np.abs(values).max()))
                        ii, jj = np.nonzero(np.triu(np.abs(margin) <= band, 1))
                        selected.update((kind, route, int(i), int(j)) for i, j in zip(ii, jj))
    # Also include every observed float32/64 disagreement before the fixed cap.
    for kind in object_domains(adjacency):
        for route in config['routes']:
            first = masks[f'float32_{kind}_{route}_t1_s1_c1']
            second = masks[f'float64_{kind}_{route}_t1_s1_c1']
            ii, jj = np.nonzero(np.triu(first != second, 1))
            selected.update((kind, route, int(i), int(j)) for i, j in zip(ii, jj))
    candidates = sorted(selected)
    cases = candidates[:config['max_boundary_pairs_per_graph']]
    boundary = []
    pending = list(cases)
    budget = Budget(config['boundary_seconds_per_graph'], config['memory_gib'])
    try:
        with hard_wall_limit(config['boundary_seconds_per_graph']):
            for bits in config['precision_bits']:
                if not pending:
                    break
                refs = build_object_features(adjacency, width, 'interval', bits, budget)
                remaining = []
                for kind, route, i, j in pending:
                    values = route_features(refs[kind], route, width)
                    state = reference_comparison(values[i], values[j], Fraction(str(config['atol'])), Fraction(str(config['rtol'])))
                    if state == 'unresolved':
                        remaining.append((kind, route, i, j))
                        continue
                    decisions = {mode: bool(masks[f'{mode}_{kind}_{route}_t1_s1_c1'][i, j]) for mode in config['dtypes']}
                    boundary.append({'domain': kind, 'route': route, 'i': i, 'j': j, 'bits': bits,
                                     'reference': state, 'numeric_separated': decisions,
                                     'arithmetic_disagreement': {mode: (decision != (state == 'certified_separated'))
                                                                 for mode, decision in decisions.items()}})
                pending = remaining
    except (ResourceLimit, MemoryError) as exc:
        reason = str(exc)
    else:
        reason = 'precision_exhausted'
    completed_cases = {(x['domain'], x['route'], x['i'], x['j']) for x in boundary}
    boundary += [{'domain': a, 'route': b, 'i': i, 'j': j, 'reference': 'unresolved', 'reason': reason}
                 for a, b, i, j in pending if (a, b, i, j) not in completed_cases]
    arrays['adjacency'] = adjacency
    arrays['permutations_new_to_old'] = np.asarray(perm)
    arrays['node_orbits'] = orbits['nodes']
    arrays['edge_orbits'] = orbits['edges']
    arrays['pair_orbits'] = orbits['pairs']
    array_file = Path('arrays') / f'graph{graph_id:04d}.npz'
    atomic_npz(out / array_file, **arrays)
    return {'graph_id': graph_id, 'status': 'complete', 'records': records,
            'array_file': str(array_file), 'array_sha256': sha256(out / array_file),
            'boundary': boundary, 'boundary_candidates': len(candidates),
            'boundary_not_checked_due_to_cap': len(candidates) - len(cases),
            'scope': 'Numerical protocol observations; exact MIS does not certify feature values.'}


def summarize(out, expected_count):
    grouped, completed = {}, []
    for graph_id in range(expected_count):
        path = out / 'records' / f'graph{graph_id:04d}.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record['status'] != 'complete':
            continue
        if sha256(out / record['array_file']) != record['array_sha256']:
            raise ValueError('Saved object arrays changed')
        completed.append(graph_id)
        index = {x['key']: x for x in record['records']}
        for key, mixed in index.items():
            if mixed['route'] != 'mixed':
                continue
            powers = index[key.replace('_mixed_', '_powers_2d_')]
            group = grouped.setdefault(key, {'domain': mixed['domain'], 'eligible': [], 'excluded': []})
            if mixed['reliable'] and powers['reliable'] and mixed['rc'] is not None and powers['rc'] is not None:
                group['eligible'].append({'graph_id': graph_id, 'difference': mixed['rc'] - powers['rc']})
            else:
                group['excluded'].append(graph_id)
    for group in grouped.values():
        differences = [item['difference'] for item in group['eligible']]
        group['mean_paired_rc_difference'] = float(np.mean(differences)) if differences else None
        group['eligible_count'] = len(differences)
    common = {}
    for domain in ('nodes', 'edges', 'pairs'):
        subset = [group for group in grouped.values() if group['domain'] == domain]
        ids = set.intersection(*(set(x['graph_id'] for x in group['eligible']) for group in subset)) if subset else set()
        common[domain] = sorted(ids)
        for group in subset:
            differences = [x['difference'] for x in group['eligible'] if x['graph_id'] in ids]
            group['common_across_all_settings_count'] = len(differences)
            group['common_across_all_settings_mean_difference'] = float(np.mean(differences)) if differences else None
    atomic_json(out / 'summary.json', {'total_graph_records': expected_count, 'completed_ids': completed,
                                      'missing_ids': sorted(set(range(expected_count)) - set(completed)),
                                      'paired_comparisons': grouped, 'common_across_all_settings_ids': common,
                                      'scope': 'Each setting uses its explicitly reported common eligible records; no predictive or exact-information claim.'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rebuttal/objects.json')
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = configuration(args.config)
    if config.get('expected_topology_sha256') and sha256(args.topology) != config['expected_topology_sha256']:
        raise ValueError('Topology differs from the frozen MUTAG archive; do not silently substitute inputs')
    with run_directory(args.out, config, [args.config, args.topology], args.resume) as (out, identity):
        total = 0
        for graph_id, adjacency in topology(args.topology):
            total += 1
            target = out / 'records' / f'graph{graph_id:04d}.json'
            if target.exists():
                if json.loads(target.read_text())['identity_sha256'] != identity:
                    raise ValueError('Saved graph identity mismatch')
                continue
            try:
                with hard_wall_limit(config['seconds_per_graph']):
                    result = graph_experiment(graph_id, adjacency, config, out)
            except (ResourceLimit, MemoryError) as exc:
                result = {'graph_id': graph_id, 'status': 'resource_limited', 'reason': str(exc)}
            atomic_json(target, {**result, 'identity_sha256': identity})
            print(f'graph {graph_id}: {result["status"]}', flush=True)
        if total != config['expected_graphs']:
            raise ValueError(f'Expected {config["expected_graphs"]} graph records; received {total}')
        summarize(out, total)


if __name__ == '__main__':
    main()
