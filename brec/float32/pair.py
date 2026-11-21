"""Pair channels, official isomorphism checks, and 2-FWL result aggregation."""

from collections import Counter

import numpy as np

from rebuttal_experiments.brec_data import main_graph_pair, official_reliability_pairs, reliability_mapping
from rebuttal_experiments.common import atomic_json

from .numerics import DOT_BITS, DOT_DIGITS, chebyshev_pair, reproducible_product, walk_bank, weighted_gap
from .protocol import BRANCH_DEGREES, ROUNDS, TARGETS, TEST_PAIRS, TOLERANCE, describe_protocol
from .runtime import experiment_directory, official_checks, pair_ids_by_size, run_cpu, summarize_official


PROTOCOL = describe_protocol(
    branch_degrees=list(BRANCH_DEGREES[:2]),
    readout='maximum per-channel sorted difference',
    contraction='sum of shared-pivot products',
    reproducible_dot_fraction_bits=DOT_BITS*DOT_DIGITS,
)


def pair_channels(bank, rounds=ROUNDS):
    """Use a normalized walk matrix to bound both branches of each direct sum."""
    base = bank[0] + bank[1]
    square = reproducible_product(bank[1], bank[1])
    fourth = reproducible_product(square, square)
    eighth = reproducible_product(fourth, fourth)
    walk = reproducible_product(reproducible_product(eighth, eighth), eighth)
    state = base
    for _ in range(rounds):
        # With 0 <= base, state <= 1 entrywise, the input is also in [0,1].
        argument = 2*(base + (1-base)*state) - 1
        left, right = (walk*np.clip((1 + chebyshev_pair(argument, d)[0])*0.5, 0, 1)
                       for d in BRANCH_DEGREES[:2])
        # Both nonnegative branches are entrywise <= walk, whose spectral norm
        # is <= 1. Their contraction therefore has spectral norm <= 1 as well.
        state = reproducible_product(right, left)
        yield state


def pair_result(data, pid):
    graphs = main_graph_pair(data, pid)
    banks = [walk_bank(graph) for graph in graphs]
    baseline = max(weighted_gap(a, b) for a, b in zip(*banks, strict=True))
    gaps = [weighted_gap(a, b) for a, b in zip(*map(pair_channels, banks), strict=True)]
    return {'pair_id': pid, 'vertices': len(graphs[0]), 'walk_gap': baseline, 'recursive_gaps': gaps}


def official_result(data, pid):
    checks = []
    for indices, (left, right) in official_reliability_pairs(data, pid):
        mapping = reliability_mapping(left, right)
        aligned = np.ix_(mapping, mapping)
        a, b = (walk_bank(graph) for graph in (left, right))
        walk_gap = float(np.max(np.abs(a - b[:, mapping, :][:, :, mapping])))
        recursive_gap = max(float(np.max(np.abs(x - y[aligned])))
                            for x, y in zip(pair_channels(a), pair_channels(b), strict=True))
        checks.append({'indices': indices, 'walk_aligned_gap': walk_gap, 'recursive_aligned_gap': recursive_gap})
    return {'pair_id': pid, 'checks': checks}


def summarize(rows):
    targets = TARGETS[2]
    walk = {row['pair_id'] for row in rows if row['walk_gap'] > TOLERANCE}
    recursive = {row['pair_id'] for row in rows if max(row['recursive_gaps']) > TOLERANCE}
    added = (targets & recursive) - walk
    complete = walk | recursive
    round_counts = Counter(
        next(depth for depth, gap in enumerate(row['recursive_gaps'], 1) if gap > TOLERANCE)
        for row in rows if row['pair_id'] in added
    )
    return {
        'tested_pairs': len(rows),
        'reference_positive': len(targets),
        'walk_different': len(walk & targets),
        'recursive_only_different': len(recursive & targets),
        'added_by_recursion': len(added),
        'complete_different': len(complete & targets),
        'first_added_by_round': dict(round_counts),
        'unresolved_positive_ids': sorted(targets - complete),
        'reference_equal': len(TEST_PAIRS[2] - targets),
        'reference_equal_marked_different': sorted(complete - targets),
        'minimum_positive_gap': min(
            max(row['walk_gap'], *row['recursive_gaps']) for row in rows if row['pair_id'] in targets
        ),
    }


def run(data, out, workers):
    ids = pair_ids_by_size(data, TEST_PAIRS[2], largest_first=True)
    with experiment_directory(out, PROTOCOL):
        rows = run_cpu(out, 'main', ids, pair_result, data, workers)
        control_records = run_cpu(out, 'official', ids, official_result, data, workers)
        controls = official_checks(control_records, len(ids))
        summary = {
            **PROTOCOL,
            **summarize(rows),
            **summarize_official(controls, 'walk_aligned_gap', 'recursive_aligned_gap'),
        }
        atomic_json(out/'summary.json', summary)
    print(summary, flush=True)
