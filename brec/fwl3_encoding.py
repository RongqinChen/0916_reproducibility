"""Progressive exact and numerical 3-FWL encoding on the 3-FWL-only BREC cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from .determinism import ensure_startup, prepare_environment

prepare_environment()

import numpy as np
import pynauty

from . import fwl3_numeric as numeric
from .fwl2_encoding import _progressive_summary, compare_encodings, encode_pair_colours
from .fwl2_progressive import check_control as check_polynomial_banks
from .suite_data import EXPECTED_SHA256, nauty_graph
from .theory import group_for_pair
from rebuttal_experiments.brec_data import (
    BLOCK_SIZE, OFFICIAL_RELIABILITY, PAIR_COUNT, load_brec_data, main_graph_pair,
    official_reliability_pairs, reliability_mapping,
)
from rebuttal_experiments.common import atomic_json, atomic_npz, run_directory
from rebuttal_experiments.encodings import descriptors
from rebuttal_experiments.triple_exact import TripleKernel, clear_caches, weighted_counts


COHORT = tuple(pid for pid in range(PAIR_COUNT) if group_for_pair(pid) == '3-FWL-only')
BASELINE_KEYS = tuple(key for key in descriptors(1, 3, 'joint')
                      if key[0] == 'constant' or (key[0] == 'pair' and key[2] < key[3])
                      or key == ('joint', 1, 1, 1))


def exact_three_fwl(orbits, atomic):
    """Exact shared-pivot reference, retaining the entire sorted joint profile."""
    colours = tuple(np.asarray(side, dtype=np.uint64) for side in atomic)
    round_ = 0
    while True:
        counts = [weighted_counts(side.tolist(), orbit.weights)
                  for side, orbit in zip(colours, orbits, strict=True)]
        if counts[0] != counts[1]:
            return {'different': True, 'round': round_}
        radix = max(counts[0]) + 1
        if radix**3 - 1 > np.iinfo(np.uint64).max:
            raise OverflowError('Joint reference colour codes exceed uint64 capacity')
        vocabulary, updated = {}, []
        for side, orbit in zip(colours, orbits, strict=True):
            profiles = numeric.joint_profiles(orbit.rows, orbit.lookup, side, np.uint64(radix))
            profiles.sort(axis=1)
            labels = [vocabulary.setdefault((int(previous), row.tobytes()), len(vocabulary))
                      for previous, row in zip(side, profiles, strict=True)]
            updated.append(np.asarray(labels, dtype=np.uint64))
        round_ += 1
        if len(vocabulary) == len(counts[0]):
            return {'different': False, 'round': None, 'stable_after_round': round_}
        colours = updated


def compare_progressively(graphs, orbits, atomic, max_rounds):
    """Compare the exact baseline plus every appended numerical channel jointly."""
    if type(max_rounds) is not int or max_rounds < 0:
        raise ValueError('Maximum numerical depth must be a nonnegative integer')
    # Symmetry supplies the reversed pair channels. A zero-degree joint
    # channel fixes the pivot to that coordinate, leaving ordinary pair
    # products and equality indicators already present in the same row.
    rows = []
    for graph, orbit in zip(graphs, orbits, strict=True):
        kernel = TripleKernel(graph, 1, 'monomial')
        rows.append(list(zip(*(kernel.column(orbit.rows, key) for key in BASELINE_KEYS), strict=True)))
    prefixes = [None, None]
    for depth in range(max_rounds + 1):
        if depth:
            for side, orbit in enumerate(orbits):
                prefix = atomic[side][:, None] if depth == 1 else prefixes[side]
                prefixes[side] = numeric.extend_triples(orbit, prefix, depth)
                channels = prefixes[side][:, -2:].tolist()
                rows[side] = [key + tuple(values) for key, values in
                              zip(rows[side], channels, strict=True)]
        left, right = (weighted_counts(values, orbit.weights)
                       for values, orbit in zip(rows, orbits, strict=True))
        if left != right:
            witness = next(key for key in chain(left, right) if left[key] != right[key])
            width = len(BASELINE_KEYS)
            return {'status': 'certified_different', 'different': True, 'round': depth,
                    'computed_rounds': depth, 'witness': {
                        'baseline': [value.json() for value in witness[:width]],
                        'numerical': list(witness[width:]),
                        'multiplicities': [left[witness], right[witness]]}}, prefixes
    return {'status': 'unresolved', 'different': False, 'round': None,
            'computed_rounds': max_rounds}, prefixes


def _check_official(graphs, depth, cached_plans):
    first, second = graphs
    mapping = reliability_mapping(first, second)
    check_polynomial_banks(first, second, np.argsort(mapping), q=1, basis='monomial', depth=0)
    if depth == 0:
        return
    # Reuse only the pivot lookup for the canonical control topology. Each
    # representative is read from its own adjacency and contracted separately.
    order = np.asarray(pynauty.canon_label(nauty_graph(first)))
    canonical = first[np.ix_(order, order)]
    key = canonical.tobytes()
    if key not in cached_plans:
        cached_plans[key] = numeric.triple_orbits(canonical)
    orbits = cached_plans[key]
    prefixes = []
    for graph, canonical_to_input in zip(graphs, (order, mapping[order]), strict=True):
        prefix = numeric.atomic_types(graph, canonical_to_input[orbits.rows])[:, None]
        for step in range(1, depth + 1):
            prefix = numeric.extend_triples(orbits, prefix, step)
        prefixes.append(prefix)
    if not np.array_equal(*prefixes):
        raise AssertionError('Numerical triple features changed on an official isomorphism')


_DATA = None


def _evaluate(pid, out, max_rounds):
    graphs = main_graph_pair(_DATA, pid)
    start = perf_counter()
    orbits = tuple(numeric.triple_orbits(graph) for graph in graphs)
    atomic = tuple(numeric.atomic_types(graph, orbit.rows)
                   for graph, orbit in zip(graphs, orbits, strict=True))
    reference2 = compare_encodings(*(encode_pair_colours(graph) for graph in graphs))
    reference3 = exact_three_fwl(orbits, atomic)
    reference_elapsed = perf_counter() - start
    if reference2['different'] or not reference3['different']:
        raise AssertionError(f'Pair {pid} is outside the exact 3-FWL-only cohort')
    decision, prefixes = compare_progressively(graphs, orbits, atomic, max_rounds)
    main_elapsed = perf_counter() - start - reference_elapsed

    control_started = perf_counter()
    controls, cached_plans = [], {}
    for indices, control_graphs in official_reliability_pairs(
            _DATA, pid, {'reliability_protocol': OFFICIAL_RELIABILITY}):
        try:
            _check_official(control_graphs, decision['computed_rounds'], cached_plans)
        except (AssertionError, ValueError) as error:
            raise AssertionError(f'BREC pair {pid}, reliability records {indices}: {error}') from error
        controls.append(list(indices))

    artifact = Path('encodings') / f'pair{pid:03d}.npz'
    arrays = {}
    for side, graph, orbit, prefix in zip(('left', 'right'), graphs, orbits, prefixes, strict=True):
        arrays[f'{side}_adjacency'] = graph.astype(np.uint8)
        arrays[f'{side}_representatives'] = orbit.rows
        arrays[f'{side}_weights'] = orbit.weights
        if prefix is not None:
            arrays[f'{side}_numerical_prefix'] = prefix
    record = {'pair_id': pid, **decision, 'artifact': str(artifact),
              'reference_2fwl': reference2, 'reference_3fwl': reference3,
              'official_reliability_pairs_passed': controls,
              'elapsed_seconds': {'reference': reference_elapsed, 'main_encoding': main_elapsed,
                                  'controls': perf_counter() - control_started}}
    atomic_npz(out / artifact, **arrays)
    atomic_json(out / 'records' / f'pair{pid:03d}.json', record)
    clear_caches()
    return record


def _warmup():
    empty = np.zeros((1, 1), dtype=np.int64)
    orbit = numeric.triple_orbits(empty)
    initial = numeric.atomic_types(empty, orbit.rows)
    exact_three_fwl((orbit, orbit), (initial, initial))
    numeric.extend_triples(orbit, initial[:, None], 1)


def main(argv=None):
    ensure_startup()
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--full', action='store_true', help='Evaluate all 110 3-FWL-only BREC pairs')
    scope.add_argument('--pair-ids', nargs='+', type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--numerical-rounds', default=10, type=int)
    args = parser.parse_args(argv)
    ids = sorted(COHORT if args.full else args.pair_ids)
    if len(set(ids)) != len(ids) or not set(ids).issubset(COHORT):
        parser.error('Select distinct IDs in the BREC 3-FWL-only cohort')
    if args.workers < 1 or args.numerical_rounds < 0:
        parser.error('Worker count must be positive and numerical depth nonnegative')
    global _DATA
    _DATA = load_brec_data(args.brec_file, EXPECTED_SHA256)
    config = {'encoding': 'progressive_3fwl_triple_features', 'executed_pair_ids': ids,
              'baseline': {'q': 1, 'basis': 'monomial', 'variant': 'joint'},
              'baseline_compact_descriptors': BASELINE_KEYS, 'readout': 'exact_row_multiset',
              'max_numerical_rounds': args.numerical_rounds, 'moduli': numeric.MODULI,
              'reliability_protocol': OFFICIAL_RELIABILITY, 'controls_depth': 'main_pair_depth',
              'expected_brec_sha256': EXPECTED_SHA256, 'workers': args.workers,
              'numba_version': numeric.NUMBA_VERSION,
              'arithmetic': 'cpu_exact_radical_and_modular_uint64'}
    inputs = [args.brec_file, Path(__file__), Path(numeric.__file__)]
    inputs += [Path(__file__).with_name(name) for name in
               ('fwl2_encoding.py', 'fwl2_numeric.py', 'fwl2_progressive.py', 'suite_data.py', 'theory.py')]
    with run_directory(args.out, config, inputs) as (out, identity):
        _warmup()
        records = []
        scheduled = sorted(ids, key=lambda pid: len(_DATA[BLOCK_SIZE * pid]), reverse=True)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ids)),
                                 mp_context=get_context('fork')) as executor:
            jobs = [executor.submit(_evaluate, pid, out, args.numerical_rounds) for pid in scheduled]
            for future in as_completed(jobs):
                row = future.result()
                records.append(row)
                n = len(records)
                bar = '=' * (20 * n // len(ids))
                print(f"3-FWL encoding [{bar:.<20}] {n}/{len(ids)} "
                      f"pair {row['pair_id']}, layer {row['round']}", flush=True)
        records.sort(key=lambda row: row['pair_id'])
        summary = {'identity_sha256': identity, 'pair_ids': ids, 'records': records,
                   'different': sum(row['different'] for row in records),
                   'reference_3fwl_only': len(records),
                   'reference_3fwl_first_difference': dict(Counter(row['reference_3fwl']['round']
                                                                  for row in records)),
                   'official_reliability_pairs_passed': sum(len(row['official_reliability_pairs_passed'])
                                                           for row in records),
                   **_progressive_summary(records, args.numerical_rounds),
                   'wall_seconds_through_computation': perf_counter() - started}
        atomic_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
