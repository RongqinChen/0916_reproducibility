"""Memory-bounded ordered 4-FWL encoding on small BREC candidates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from math import lcm
from multiprocessing import get_context
from pathlib import Path
import resource
from time import perf_counter

from .determinism import ensure_startup, prepare_environment

prepare_environment()

import numpy as np
import pynauty
from numba import njit

from . import fwl4_numeric as numeric
from .fwl2_encoding import _progressive_summary
from .fwl2_progressive import check_control as check_polynomial_banks
from .suite_data import EXPECTED_SHA256, nauty_graph
from .suite_resources import Resources, enforce_address_space_limit
from rebuttal_experiments.brec_data import (
    OFFICIAL_RELIABILITY, load_brec_data, main_graph_pair,
    official_reliability_pairs, reliability_mapping,
)
from rebuttal_experiments.common import atomic_json, atomic_npz, run_directory


_DATA = None
GIB = 2**30
WORKER_GIB = 10
RESERVE_GIB = 8
PROFILE_BLOCK = 4096


def _profile_blocks(orbit, colours, radix):
    one_word = radix**4 <= 2**64
    for start in range(0, len(orbit.rows), PROFILE_BLOCK):
        stop = min(start+PROFILE_BLOCK, len(orbit.rows))
        profiles = numeric.pivot_profiles(orbit.rows[start:stop], orbit.lookup, colours,
                                          np.uint64(radix), one_word)
        if one_word:
            profiles[:, :, 0].sort(axis=1)
        else:
            order = np.lexsort((profiles[:, :, 1], profiles[:, :, 0]), axis=1)
            profiles = np.take_along_axis(profiles, order[:, :, None], axis=1)
        yield start, profiles


def _histogram(colours, weights, width):
    counts = np.zeros(width, dtype=np.int64)
    np.add.at(counts, colours.astype(np.intp, copy=False), weights)
    return counts


def exact_four_fwl(orbits, atomic):
    """Exact shared-pivot refinement with weighted orbit rows and streamed profiles."""
    colours, round_ = atomic, 0
    width = int(max(side.max() for side in atomic))+1
    counts = [_histogram(side, orbit.weights, width)
              for side, orbit in zip(colours, orbits, strict=True)]
    if not np.array_equal(*counts):
        return {'different': True, 'round': 0}
    occupied = int(np.count_nonzero(counts[0]))
    while True:
        if width**2 > 2**64:
            raise OverflowError('Four reference colours need more than two uint64 words')
        vocabulary, remaining = {}, []
        left, right = (np.empty(len(side), dtype=np.uint64) for side in colours)
        for side, orbit in enumerate(orbits):
            current = (left, right)[side]
            for start, profiles in _profile_blocks(orbit, colours[side], width):
                for i, row in enumerate(profiles, start):
                    key = int(colours[side][i]), row.tobytes()
                    value = vocabulary.get(key)
                    weight = int(orbit.weights[i])
                    if side == 0:
                        if value is None:
                            value = len(vocabulary)
                            vocabulary[key] = value
                            remaining.append(0)
                        remaining[value] += weight
                    else:
                        if value is None or remaining[value] < weight:
                            return {'different': True, 'round': round_+1}
                        remaining[value] -= weight
                    current[i] = value
        if any(remaining):
            return {'different': True, 'round': round_+1}
        round_ += 1
        width = len(vocabulary)
        if width == occupied:
            return {'different': False, 'round': None, 'stable_after_round': round_}
        occupied, colours = width, (left, right)


@njit(cache=True)
def _joint_num(graph, rows, offsets, neighbours, factors):
    result = np.zeros(len(rows), dtype=np.uint64)
    for i, (u, v, w, z) in enumerate(rows):
        total = np.uint64(0)
        for p in range(offsets[u], offsets[u+1]):
            x = neighbours[p]
            if graph[v, x] and graph[w, x] and graph[z, x]:
                total += factors[x]*factors[x]
        result[i] = total*total*factors[u]*factors[v]*factors[w]*factors[z]
    return result


def baseline_rows(graph, rows, atomic, scale):
    # The nonnegative normalized pair channels and all-positive joint channel
    # can be compared by exact squares with denominators L**2 and L**8.
    degrees = graph.sum(axis=1)
    factors = (scale//np.maximum(degrees, 1)).astype(np.uint64)
    if len(graph)**2 * int(factors.max())**8 > np.iinfo(np.uint64).max:
        raise OverflowError('Exact Q=1 baseline numerators exceed uint64 capacity')
    pair_features = []
    for i, j in numeric.PAIR_POSITIONS:
        u, v = rows[:, i], rows[:, j]
        pair_features.append(factors[u]*factors[v]*graph[u, v].astype(np.uint64))
    offsets = np.r_[0, np.cumsum(degrees)]
    joint = _joint_num(graph, rows, offsets, np.nonzero(graph)[1], factors)
    return np.column_stack((atomic, *pair_features, joint))


def _row_counts(rows, weights):
    order = np.lexsort(rows.T[::-1])
    sorted_rows = rows[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(sorted_rows[1:] != sorted_rows[:-1], axis=1))]
    return sorted_rows[starts], np.add.reduceat(weights[order], starts, dtype=np.int64)


def _difference(rows, orbits):
    a, b = [_row_counts(side, orbit.weights) for side, orbit in zip(rows, orbits, strict=True)]
    width = min(len(a[0]), len(b[0]))
    mismatches = np.flatnonzero(
        np.any(a[0][:width] != b[0][:width], axis=1) | (a[1][:width] != b[1][:width]))
    if not len(mismatches) and len(a[0]) == len(b[0]):
        return None
    index = int(mismatches[0]) if len(mismatches) else width
    feature = min(side[0][index].tolist() for side in (a, b) if index < len(side[0]))
    multiplicities = [int(counts[np.all(values == feature, axis=1)].sum())
                      for values, counts in (a, b)]
    return {'features': feature, 'multiplicities': multiplicities}


def compare_progressively(graphs, orbits, atomic, max_rounds):
    scales = [lcm(*np.maximum(graph.sum(axis=1), 1).tolist()) for graph in graphs]
    common_scale = lcm(*scales)
    base = [baseline_rows(g, o.rows, c, common_scale)
            for g, o, c in zip(graphs, orbits, atomic, strict=True)]
    numeric_prefixes = [None, None]
    for t in range(max_rounds+1):
        if t:
            for side, orbit in enumerate(orbits):
                previous = atomic[side][:, None] if t == 1 else numeric_prefixes[side]
                numeric_prefixes[side] = numeric.extend_quadruples(orbit, previous, t)
            rows = [np.column_stack((features, p[:, 1:]))
                    for features, p in zip(base, numeric_prefixes, strict=True)]
        else:
            rows = base
        witness = _difference(rows, orbits)
        if witness is not None:
            return {'different': True, 'status': 'certified_different', 'round': t,
                    'computed_rounds': t, 'baseline_denominator_scale': common_scale,
                    'witness': witness}, numeric_prefixes
    return {'different': False, 'status': 'unresolved', 'round': None,
            'computed_rounds': max_rounds, 'baseline_denominator_scale': common_scale}, numeric_prefixes


def _official_control(graphs, depth, cache):
    first, second = graphs
    mapping = reliability_mapping(first, second)
    check_polynomial_banks(first, second, np.argsort(mapping), q=1, basis='monomial', depth=0)
    if not depth:
        return
    order = np.asarray(pynauty.canon_label(nauty_graph(first)))
    canonical = first[np.ix_(order, order)]
    key = canonical.tobytes()
    if key not in cache:
        # Keep only the current topology resident. Numerical channels on the
        # two official inputs are computed independently.
        cache.clear()
        cache[key] = numeric.quadruple_orbits(canonical)
    orbit = cache[key]
    prefixes = []
    for graph, indices in zip(graphs, (order, mapping[order]), strict=True):
        prefix = numeric.atomic_types(graph, indices[orbit.rows])[:, None]
        for t in range(1, depth+1):
            prefix = numeric.extend_quadruples(orbit, prefix, t)
        prefixes.append(prefix)
    if not np.array_equal(*prefixes):
        raise AssertionError('Ordered 4-FWL numerical prefix changed on an official isomorphism')


def _evaluate(pid, out, max_rounds, reference3):
    started = perf_counter()
    graphs = main_graph_pair(_DATA, pid)
    orbits = tuple(numeric.quadruple_orbits(graph) for graph in graphs)
    atomic = tuple(numeric.atomic_types(g, o.rows) for g, o in zip(graphs, orbits, strict=True))
    reference4 = exact_four_fwl(orbits, atomic)
    elapsed_reference = perf_counter()-started
    print(f'4-FWL reference pair {pid}: {reference4}', flush=True)
    record = {'pair_id': pid, 'vertices': len(graphs[0]), 'reference_3fwl': reference3,
              'reference_4fwl': reference4, 'orbit_rows': [len(o.rows) for o in orbits],
              'elapsed_reference_seconds': elapsed_reference}
    if reference4['different']:
        decision, prefixes = compare_progressively(graphs, orbits, atomic, max_rounds)
        print(f"4-FWL encoding pair {pid}: {decision['status']} at layer {decision['round']}",
              flush=True)
        artifact = Path('encodings') / f'pair{pid:03d}.npz'
        arrays = {}
        for side, graph, orbit, prefix in zip(('left', 'right'), graphs, orbits, prefixes, strict=True):
            arrays[f'{side}_adjacency'] = graph.astype(np.uint8)
            arrays[f'{side}_representatives'] = orbit.rows
            arrays[f'{side}_weights'] = orbit.weights
            if prefix is not None:
                arrays[f'{side}_numerical_prefix'] = prefix
        atomic_npz(out/artifact, **arrays)
        del orbit, orbits
        controls, cache = [], {}
        for indices, control_graphs in official_reliability_pairs(
                _DATA, pid, {'reliability_protocol': OFFICIAL_RELIABILITY}):
            _official_control(control_graphs, decision['computed_rounds'], cache)
            controls.append(list(indices))
            if len(controls) % 8 == 0:
                print(f'4-FWL official controls pair {pid}: {len(controls)}/32', flush=True)
        record.update(decision, artifact=str(artifact), official_reliability_pairs_passed=controls)
    else:
        record.update(status='outside_reference_cohort', official_reliability_pairs_passed=[])
    record.update(elapsed_pair_seconds=perf_counter()-started,
                  worker_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    atomic_json(out/'records'/f'pair{pid:03d}.json', record)
    return record


def _warmup():
    graph = np.zeros((1, 1), dtype=np.int64)
    orbit = numeric.quadruple_orbits(graph)
    atomic = numeric.atomic_types(graph, orbit.rows)
    exact_four_fwl((orbit, orbit), (atomic, atomic))
    compare_progressively((graph, graph), (orbit, orbit), (atomic, atomic), 1)


def _limit_worker():
    enforce_address_space_limit(WORKER_GIB)


def main(argv=None):
    ensure_startup()
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-3fwl', type=Path,
                        default=Path('outputs/brec_fwl3_reference_340_359.json'))
    parser.add_argument('--max-nodes', type=int, default=100)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--numerical-rounds', type=int, default=10)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.max_nodes <= 100 or args.workers < 1 or args.numerical_rounds < 0:
        parser.error('Use a positive node cap up to 100, positive workers, and nonnegative depth')
    reference = json.loads(args.reference_3fwl.read_text())
    if reference['dataset_sha256'] != EXPECTED_SHA256:
        parser.error('The 3-FWL reference must match the official input')
    candidates = {r['pair_id']: r['reference_3fwl'] for r in reference['records']
                  if not r['reference_3fwl']['different']
                  and r['reference_3fwl'].get('stable_after_round') is not None}
    if set(candidates) != set(range(340, 360)):
        parser.error('Expected exact stable 3-FWL reference records for IDs 340–359')
    global _DATA
    _DATA = load_brec_data(args.brec_file, EXPECTED_SHA256)
    sizes = {pid: max(map(len, main_graph_pair(_DATA, pid))) for pid in candidates}
    sizes = {pid: n for pid, n in sizes.items() if n <= args.max_nodes}
    if not sizes:
        parser.error('No 3-FWL-indistinguishable candidates within the node cap')
    resources = Resources()
    available = resources.memory_available()/GIB
    workers = min(len(sizes), args.workers, resources.cpus(), int((available-RESERVE_GIB)//WORKER_GIB))
    if workers < 1:
        parser.error(f'Only {available:.2f} GiB available, need {WORKER_GIB+RESERVE_GIB} GiB')
    print(f'4-FWL preflight: {available:.2f} GiB available, {RESERVE_GIB} GiB reserve, '
          f'{WORKER_GIB} GiB address-space cap × {workers} workers', flush=True)
    for pid, nodes in sorted(sizes.items()):
        print(f'  pair {pid}: {nodes} nodes, two int32 orbit tables <= {8*nodes**4/GIB:.2f} GiB',
              flush=True)
    if args.preflight_only:
        return
    config = {'encoding': 'progressive_4fwl_ordered_quadruples', 'baseline': 'monomial_joint_q1',
              'node_cap': args.max_nodes, 'nodes_by_pair': sizes, 'workers': workers,
              'max_numerical_rounds': args.numerical_rounds, 'moduli': numeric.MODULI,
              'profile_block_rows': PROFILE_BLOCK, 'worker_address_space_gib': WORKER_GIB,
              'reserve_gib': RESERVE_GIB, 'preflight_available_gib': available,
              'reliability_protocol': OFFICIAL_RELIABILITY, 'numba_version': numeric.NUMBA_VERSION}
    inputs = [args.brec_file, args.reference_3fwl, Path(__file__), Path(numeric.__file__)]
    inputs.extend(Path(__file__).with_name(name) for name in
                  ('fwl2_encoding.py', 'fwl2_progressive.py', 'fwl3_numeric.py',
                   'fwl2_numeric.py', 'suite_data.py', 'suite_resources.py'))
    with run_directory(args.out, config, inputs) as (out, identity):
        _warmup()
        records = []
        scheduled = sorted(sizes, key=sizes.get, reverse=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'),
                                 initializer=_limit_worker) as executor:
            jobs = [executor.submit(_evaluate, pid, out, args.numerical_rounds, candidates[pid])
                    for pid in scheduled]
            for job in as_completed(jobs):
                records.append(job.result())
                n = len(records)
                bar = '='*(20*n//len(sizes))
                print(f'4-FWL encoding [{bar:.<20}] {n}/{len(sizes)} candidates', flush=True)
        records.sort(key=lambda row: row['pair_id'])
        target = [row for row in records if row['reference_4fwl']['different']]
        summary = {'identity_sha256': identity, 'candidate_ids': sorted(sizes),
                   'target_pair_ids': [row['pair_id'] for row in target], 'records': records,
                   'different': sum(row['different'] for row in target),
                   'outside_reference_cohort': [r['pair_id'] for r in records if r not in target],
                   'official_reliability_pairs_passed': sum(len(r['official_reliability_pairs_passed'])
                                                           for r in records),
                   **_progressive_summary(target, args.numerical_rounds),
                   'peak_worker_rss_gib': max(r['worker_peak_rss_gib'] for r in records),
                   'wall_seconds_through_computation': perf_counter()-started}
        atomic_json(out/'summary.json', summary)


if __name__ == '__main__':
    main()
