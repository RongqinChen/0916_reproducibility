"""Triple channels, official isomorphism checks, and 3-FWL result aggregation."""

from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
from numba import njit
import pynauty

from brec.fwl3_numeric import triple_orbits
from brec.suite_data import nauty_graph
from rebuttal_experiments.brec_data import main_graph_pair, official_reliability_pairs
from rebuttal_experiments.common import atomic_json

from .numerics import adjacent_walk_average, chebyshev_pair, walk_bank, weighted_gap
from .protocol import BRANCH_DEGREES, PIVOT_FRACTION_BITS, ROUNDS, TARGETS, TEST_PAIRS, TOLERANCE, describe_protocol
from .runtime import (
    experiment_directory, official_checks, pair_ids_by_size, run_cpu, summarize_official, verify_lower_order,
)


JOINT_DEGREE = 4
DEGREE_TRIPLES = tuple(combinations_with_replacement(range(1, JOINT_DEGREE+1), 3))
PIVOT_LOW_BITS = 16
PIVOT_SCALE = np.float32(2.0**PIVOT_FRACTION_BITS)
PIVOT_UNIT = np.float32(2.0**-PIVOT_FRACTION_BITS)
PIVOT_LOW_BASE = np.float32(2.0**PIVOT_LOW_BITS)
PIVOT_HIGH_UNIT = np.float32(2.0**(PIVOT_LOW_BITS-PIVOT_FRACTION_BITS))


PROTOCOL = describe_protocol(
    static_joint_degree=JOINT_DEGREE,
    static_channels=len(DEGREE_TRIPLES),
    branch_degrees=list(BRANCH_DEGREES[:3]),
    readout='maximum per-channel sorted difference',
    carrier='mean of two successive Walk powers, applied to the first two positions',
    float32_pivot_sum_fraction_bits=PIVOT_FRACTION_BITS,
    controls='all static channels and all recursive rounds',
)


@njit(inline='always')
def pivot_digits(term):
    # For nonnegative contractions bounded by one and n <= 256, both sums
    # are exact float32 integers. Their rounding is independent of pivot order.
    units = np.rint(term * PIVOT_SCALE)
    high = np.floor(units / PIVOT_LOW_BASE)
    return high, units - high*PIVOT_LOW_BASE


@njit(inline='always')
def pivot_sum(high, low):
    return high*PIVOT_HIGH_UNIT + low*PIVOT_UNIT


@njit(cache=True)
def static_channel(rows, pivots, a, b, c):
    values = np.empty(len(rows), dtype=np.float32)
    for index, (u, v, w) in enumerate(rows):
        high, low = np.float32(0), np.float32(0)
        for x in pivots:
            term = (a[u,x]*b[v,x])*c[w,x]
            h, l = pivot_digits(term)
            high += h
            low += l
        values[index] = pivot_sum(high, low)
    return values


@njit(cache=True)
def recursive_channel(rows, lookup, pivots, a, b, c, walk):
    values = np.empty(len(rows), dtype=np.float32)
    for index, (u, v, w) in enumerate(rows):
        high, low = np.float32(0), np.float32(0)
        for x in pivots:
            joint = (a[lookup[x,v,w]]*b[lookup[u,x,w]])*c[lookup[u,v,x]]
            term = (joint*walk[u,x])*walk[v,x]
            h, l = pivot_digits(term)
            high += h
            low += l
        values[index] = pivot_sum(high, low)
    return values


def triple_seed(uv, uw, vw):
    return ((uv + np.float32(3)*uw) + np.float32(9)*vw)/np.float32(13)


def triple_gates(base, state):
    argument = 2*(base+(1-base)*state)-1
    return tuple(np.clip((1+chebyshev_pair(argument, d)[0])*np.float32(0.5), 0, 1)
                 for d in BRANCH_DEGREES[:3])


def state_channels(bank, plan, pivots, rounds=ROUNDS):
    u, v, w = plan.rows.T
    pairs = bank[0]+bank[1]
    base = triple_seed(pairs[u, v], pairs[u, w], pairs[v, w])
    walk = adjacent_walk_average(bank)
    state = base
    for _ in range(rounds):
        a, b, c = triple_gates(base, state)
        # The three replacements interact for each x. Bounding two of the
        # factors by walk rows gives sum_x walk[u,x]*walk[v,x] <= 1.
        state = recursive_channel(plan.rows, plan.lookup, pivots, a, b, c, walk)
        yield state


def compare_main(data, pid):
    graphs = main_graph_pair(data, pid)
    banks = [walk_bank(graph) for graph in graphs]
    plans = [triple_orbits(graph) for graph in graphs]
    weights = [plan.weights for plan in plans]
    pivots = np.arange(len(graphs[0]), dtype=np.int32)
    static_gaps = []
    for degrees in DEGREE_TRIPLES:
        left, right = [static_channel(plan.rows, pivots, *bank[list(degrees)])
                       for bank, plan in zip(banks, plans, strict=True)]
        static_gaps.append(weighted_gap(left, right, weights))
        if static_gaps[-1] > TOLERANCE:
            break
    recursive_gaps = []
    if max(static_gaps) <= TOLERANCE:
        streams = [state_channels(bank, plan, pivots) for bank, plan in zip(banks, plans, strict=True)]
        for left, right in zip(*streams, strict=True):
            recursive_gaps.append(weighted_gap(left, right, weights))
            if recursive_gaps[-1] > TOLERANCE:
                break
    first = next((depth for depth, gap in enumerate(recursive_gaps, 1) if gap > TOLERANCE), None)
    return {
        'pair_id': pid,
        'vertices': len(graphs[0]),
        'static_gaps': static_gaps,
        'recursive_gaps': recursive_gaps,
        'first_recursive_round': first,
    }


def check_official(data, pid):
    checks = []
    previous = plan = None
    for indices, graphs in official_reliability_pairs(data, pid):
        orders = [np.asarray(pynauty.canon_label(nauty_graph(graph)),dtype=np.int32) for graph in graphs]
        aligned = [g[np.ix_(p,p)] for g,p in zip(graphs,orders,strict=True)]
        if not np.array_equal(*aligned):
            raise ValueError(f'Official control {indices} has unequal aligned adjacency matrices')
        if previous is None or not np.array_equal(previous, aligned[0]):
            previous = aligned[0]
            plan = triple_orbits(previous)
        banks = [walk_bank(g)[:, p, :][:, :, p] for g, p in zip(graphs, orders, strict=True)]
        pivots = [np.argsort(p).astype(np.int32) for p in orders]
        walk_gap = float(np.max(np.abs(banks[0] - banks[1])))
        static_gap = 0.0
        for degrees in DEGREE_TRIPLES:
            left, right = [static_channel(plan.rows, x, *bank[list(degrees)])
                           for bank, x in zip(banks, pivots, strict=True)]
            static_gap = max(static_gap, float(np.max(np.abs(left-right))))
        streams = [state_channels(bank, plan, x) for bank, x in zip(banks, pivots, strict=True)]
        gaps = [float(np.max(np.abs(left-right))) for left, right in zip(*streams, strict=True)]
        checks.append({
            'indices': indices,
            'walk_aligned_gap': walk_gap,
            'static_aligned_gap': static_gap,
            'recursive_aligned_gaps': gaps,
            'max_aligned_gap': max(walk_gap, static_gap, *gaps),
        })
    return {'pair_id': pid, 'checked_recursive_rounds': ROUNDS, 'checks': checks}


def summarize_main(records):
    rows = {row['pair_id']: row for row in records}
    targets = TARGETS[3]
    static = {pid for pid, row in rows.items() if max(row['static_gaps']) > TOLERANCE}
    recursive = {pid for pid, row in rows.items() if row['first_recursive_round'] is not None}
    added = recursive & targets
    complete = static | recursive
    peak = {pid: max((*row['static_gaps'], *row['recursive_gaps'])) for pid, row in rows.items()}
    return {
        'tested_pairs': len(rows),
        'reference_positive': len(targets),
        'walk_different': 0,
        'static_different': len(static & targets),
        'added_by_recursion': len(added),
        'complete_different': len(complete & targets),
        'minimum_positive_gap': min(peak[pid] for pid in targets),
        'first_added_by_round': dict(Counter(rows[pid]['first_recursive_round'] for pid in added)),
        'unresolved_positive_ids': sorted(targets - complete),
        'reference_equal': len(rows) - len(targets),
        'reference_equal_marked_different': sorted(complete - targets),
    }


def run(data, out, workers, lower_order_out=None):
    ids = pair_ids_by_size(data, TEST_PAIRS[3], largest_first=True)
    if lower_order_out:
        verify_lower_order(lower_order_out, ids, ('walk_gap', 'recursive_gaps'))
    with experiment_directory(out, PROTOCOL):
        rows = run_cpu(out, 'main', ids, compare_main, data, workers)
        summary = {**PROTOCOL, **summarize_main(rows)}
        atomic_json(out/'main_summary.json', summary)
        print(summary, flush=True)
        records = run_cpu(out, 'official', ids, check_official, data, workers)
        controls = official_checks(records, len(ids))
        if any(len(check['recursive_aligned_gaps']) != ROUNDS for check in controls):
            raise ValueError('Missing recursive rounds for an official check')
        summary.update(summarize_official(controls, 'max_aligned_gap'))
        atomic_json(out/'summary.json', summary)
    print(summary, flush=True)
