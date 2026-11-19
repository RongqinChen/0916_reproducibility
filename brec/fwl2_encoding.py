"""Compare structural pair encodings and their official BREC controls."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path

from .determinism import ensure_startup, prepare_environment

prepare_environment()

import numpy as np

from . import fwl2_numeric, fwl2_progressive
from rebuttal_experiments.brec_data import (
    BLOCK_SIZE, OFFICIAL_RELIABILITY, PAIR_COUNT, load_brec_data, main_graph_pair,
    official_reliability_pairs, reliability_mapping,
)
from rebuttal_experiments.common import (
    atomic_json, atomic_npz, run_directory, simple_adjacency,
)

FULL_PAIR_IDS = (*range(110), *range(160, 320))


@dataclass(frozen=True)
class ColourRound:
    profiles: np.ndarray
    counts: np.ndarray
    radix: int = 0


@dataclass(frozen=True)
class PairEncoding:
    colours: np.ndarray
    rounds: tuple[ColourRound, ...]


def _refine(colours: np.ndarray) -> tuple[np.ndarray, ColourRound]:
    n = len(colours)
    radix = int(colours.max()) + 1
    if radix * radix - 1 > np.iinfo(np.int64).max:
        raise OverflowError('Joint colour codes exceed int64 capacity')
    signatures = np.empty((n, n, n + 1), dtype=np.int64)
    signatures[:, :, 0] = colours
    joint = signatures[:, :, 1:]
    # Each column in the pivot dimension couples (u, w) with (w, v).
    joint[:] = radix * colours[:, None, :] + colours.T[None, :, :]
    joint.sort(axis=-1)
    profiles, inverse, counts = np.unique(signatures.reshape(-1, n + 1), axis=0,
                                           return_inverse=True, return_counts=True)
    return inverse.reshape(n, n), ColourRound(profiles, counts, radix)


def encode_pair_colours(adjacency: np.ndarray) -> PairEncoding:
    """Return stable pair colours with their canonical refinement dictionaries.

    Each graph is encoded independently. Lexicographically sorted signatures
    assign reproducible local IDs. Keep the dictionaries with these IDs to
    interpret colours or compare histograms across graphs.
    """
    adjacency = simple_adjacency(adjacency)
    n = len(adjacency)
    colours = adjacency + 2 * np.eye(n, dtype=np.int64)
    initial = np.bincount(colours.ravel(), minlength=3)
    rounds = [ColourRound(np.arange(3, dtype=np.int64)[:, None], initial)]
    previous = np.count_nonzero(initial)
    while True:
        colours, step = _refine(colours)
        rounds.append(step)
        if len(step.counts) == previous:
            return PairEncoding(colours, tuple(rounds))
        previous = len(step.counts)


def compare_encodings(first: PairEncoding, second: PairEncoding) -> dict:
    """Compare exact 2-FWL histograms and return the first differing count."""
    orders = [len(encoding.colours) for encoding in (first, second)]
    if orders[0] != orders[1]:
        return {'different': True, 'round': 0, 'vertex_counts': orders}
    for index, (left, right) in enumerate(zip(first.rounds, second.rounds, strict=True)):
        # Identical earlier rounds make the predecessor IDs comparable here.
        if (np.array_equal(left.profiles, right.profiles)
                and np.array_equal(left.counts, right.counts)):
            continue
        a, b = ({tuple(row.tolist()): int(count) for row, count in zip(step.profiles, step.counts)}
                for step in (left, right))
        profile = min(p for p in a.keys() | b.keys() if a.get(p, 0) != b.get(p, 0))
        witness = {'previous_colour': profile[0],
                   'multiplicities': [a.get(profile, 0), b.get(profile, 0)]}
        if index:
            witness['joint_colour_counts'] = [
                [*divmod(code, left.radix), count] for code, count in Counter(profile[1:]).items()
            ]
        return {'different': True, 'round': index, 'witness': witness}
    return {'different': False, 'round': None}


def _check_control(first, second, new_to_old, compare):
    a, b = (first.colours, second.colours) if isinstance(first, PairEncoding) else (first, second)
    if (compare(first, second)['different']
            or not np.array_equal(a[np.ix_(new_to_old, new_to_old)], b)):
        raise AssertionError('2-FWL encoding changed under an isomorphic relabeling')


def _artifact(encoding, name):
    arrays = {f'{name}_colours': encoding.colours,
              f'{name}_radices': np.asarray([step.radix for step in encoding.rounds])}
    for index, step in enumerate(encoding.rounds):
        arrays[f'{name}_r{index}_profiles'] = step.profiles
        arrays[f'{name}_r{index}_counts'] = step.counts
    return arrays


def _evaluate(data, out, pair_id, numerical_rounds, baseline):
    graphs = main_graph_pair(data, pair_id)
    artifact = Path('encodings') / f'pair{pair_id:03d}.npz'
    if baseline is not None:
        decision, prefixes = fwl2_progressive.compare_progressively(
            *graphs, **baseline, max_rounds=numerical_rounds)
        check_control = partial(fwl2_progressive.check_control, q=baseline['q'],
                                basis=baseline['basis'], depth=decision['computed_rounds'])
        arrays = {}
        for side, graph, prefix in zip(('left', 'right'), graphs, prefixes, strict=True):
            arrays[f'{side}_adjacency'] = graph.astype(np.uint8)
            if prefix is not None:
                arrays[f'{side}_numerical_prefix'] = prefix
    elif numerical_rounds is None:
        encode, compare = encode_pair_colours, compare_encodings
    else:
        encode = partial(fwl2_numeric.encode_numeric_pairs, rounds=numerical_rounds)
        compare = fwl2_numeric.compare_numeric_encodings

    if baseline is None:
        encodings = [encode(graph) for graph in graphs]
        decision = compare(*encodings)

        def check_control(a, b, permutation):
            _check_control(encode(a), encode(b), permutation, compare)

        if numerical_rounds is None:
            arrays = {**_artifact(encodings[0], 'left'), **_artifact(encodings[1], 'right')}
            decision['stable_colours'] = [len(e.rounds[-1].counts) for e in encodings]
        else:
            arrays = dict(left_features=encodings[0], right_features=encodings[1])
            decision['feature_width'] = encodings[0].shape[-1]

    controls = []
    for indices, (a, b) in official_reliability_pairs(data, pair_id,
                                                     {'reliability_protocol': OFFICIAL_RELIABILITY}):
        old_to_new = reliability_mapping(a, b)
        try:
            check_control(a, b, np.argsort(old_to_new))
        except AssertionError as error:
            raise AssertionError(f'BREC pair {pair_id}, reliability records {indices}: {error}') from error
        controls.append(list(indices))
    record = {'pair_id': pair_id, **decision, 'artifact': str(artifact),
              'official_reliability_pairs_passed': controls}
    atomic_npz(out / artifact, **arrays)
    atomic_json(out / 'records' / f'pair{pair_id:03d}.json', record)
    return record


_DATA = None


def _run_pair(pair_id, *, out, numerical_rounds, baseline):
    return _evaluate(_DATA, out, pair_id, numerical_rounds, baseline)


def _progressive_summary(records, max_rounds):
    computed = Counter(row['computed_rounds'] for row in records)
    first_distinctions = Counter(row['round'] for row in records if row['different'])
    evaluated, unresolved = len(records), len(records)
    layers = []
    for depth in range(max_rounds + 1):
        unresolved -= first_distinctions[depth]
        layers.append({'round': depth, 'evaluated': evaluated,
                       'newly_distinguished': first_distinctions[depth],
                       'remaining_unresolved': unresolved})
        evaluated -= computed[depth]
    return {'unresolved': unresolved, 'layers': layers,
            'numerical_pair_rounds_computed': sum(depth * pairs for depth, pairs in computed.items()),
            'numerical_pair_rounds_without_early_stop': len(records) * max_rounds}


def main(argv=None):
    ensure_startup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument('--pair-ids', type=int, nargs='+')
    cohort.add_argument('--full', action='store_true',
                        help='Evaluate the 270 standard 2-FWL-distinguishable BREC pairs progressively')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--progressive', action='store_true',
                        help='Start with the original encoding and append rounds only for unresolved pairs')
    parser.add_argument('--base-q', type=int, default=16, help='Polynomial degree of the original encoding')
    parser.add_argument('--basis', choices=('monomial', 'chebyshev'), default='monomial')
    parser.add_argument('--variant', choices=('polynomial', 'endpoint'), default='endpoint')
    parser.add_argument('--numerical-rounds', type=int,
                        help='Numerical depth, or maximum additional rounds in progressive mode (default 5)')
    args = parser.parse_args(argv)
    selected = FULL_PAIR_IDS if args.full else args.pair_ids
    ids = sorted(set(selected))
    if (len(ids) != len(selected) or min(ids) < 0 or max(ids) >= PAIR_COUNT
            or args.workers < 1):
        parser.error('Select distinct BREC pair IDs and a positive worker count')
    if args.numerical_rounds is not None and args.numerical_rounds < 0:
        parser.error('Numerical rounds must be nonnegative')
    if args.base_q < 1:
        parser.error('The baseline polynomial degree must be positive')
    expected = '6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562'
    global _DATA
    _DATA = load_brec_data(args.brec_file, expected)
    config = {'encoding': 'stable_2fwl_pair_colours', 'readout': 'exact_colour_histogram',
              'executed_pair_ids': ids, 'reliability_protocol': OFFICIAL_RELIABILITY,
              'expected_brec_sha256': expected, 'workers': args.workers}
    inputs = [args.brec_file, Path(__file__)]
    baseline = None
    numerical_rounds = args.numerical_rounds
    if args.progressive or args.full:
        if numerical_rounds is None:
            numerical_rounds = 5
        baseline = dict(q=args.base_q, basis=args.basis, variant=args.variant)
        config.update(encoding='progressive_2fwl_pair_features', readout='exact_row_multiset',
                      baseline=baseline, max_numerical_rounds=numerical_rounds,
                      moduli=fwl2_numeric.MODULI, controls_depth='evaluated_main_pair_depth')
        inputs.extend([Path(fwl2_progressive.__file__), Path(fwl2_numeric.__file__),
                       Path(__file__).with_name('suite_data.py')])
    elif numerical_rounds is not None:
        config.update(encoding='numerical_2fwl_pair_features', readout='exact_row_multiset',
                      numerical_rounds=numerical_rounds, moduli=fwl2_numeric.MODULI)
        inputs.append(Path(fwl2_numeric.__file__))
    with run_directory(args.out, config, inputs) as (out, identity):
        records = []
        # Larger graphs have the most expensive official control blocks.
        scheduled = sorted(ids, key=lambda pid: len(_DATA[BLOCK_SIZE * pid]), reverse=True)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ids)),
                                 mp_context=get_context('fork')) as executor:
            evaluate = partial(_run_pair, out=out, numerical_rounds=numerical_rounds, baseline=baseline)
            for future in as_completed([executor.submit(evaluate, pid) for pid in scheduled]):
                records.append(future.result())
                n = len(records)
                bar = '=' * (20 * n // len(ids))
                print(f'2-FWL encoding [{bar:.<20}] {n}/{len(ids)}', flush=True)
        records.sort(key=lambda row: row['pair_id'])
        summary = {'identity_sha256': identity, 'pair_ids': ids,
                   'different': sum(row['different'] for row in records),
                   'official_reliability_pairs_passed': sum(len(row['official_reliability_pairs_passed'])
                                                            for row in records),
                   'records': records}
        if baseline is not None:
            summary.update(_progressive_summary(records, numerical_rounds))
        atomic_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
