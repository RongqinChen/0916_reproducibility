"""Experiment 3: exact/interval BREC pair decisions and relabeling controls."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

from brec.determinism import prepare_environment, start_brec
prepare_environment()
import numpy as np

from .readouts import (DIFFERENT, EQUAL, UNRESOLVED, exact_multiset,
                       interval_multiset, mean_decision)
from .common import (Budget, ResourceLimit, atomic_json,
                     configuration, hard_wall_limit, permutations, run_directory,
                     sha256, simple_adjacency)
from .encodings import feature_matrix, moments


def load_brec_data(path, expected_sha256):
    if expected_sha256 and sha256(path) != expected_sha256:
        raise ValueError('BREC input hash differs from the frozen protocol')
    data = np.load(path, allow_pickle=False)
    if data.ndim != 1 or len(data) != 51200 or data.dtype.kind not in ('S', 'U'):
        raise ValueError('Expected the 51200-entry official graph6 byte/string array')
    return data


def graph_at(data, index):
    import networkx as nx
    value = data[index]
    value = value.encode() if isinstance(value, str) else bytes(value)
    return simple_adjacency(nx.to_numpy_array(nx.from_graph6_bytes(value.strip()), dtype='int64'))


def is_regular(graph):
    degrees = graph.sum(1)
    return bool(np.all(degrees == degrees[0]))


def decide(rows, readout, p, mode, budget, config):
    if readout == 'mean':
        return mean_decision(*(moments(x, p, budget) for x in rows))
    if mode == 'exact':
        return exact_multiset(*rows)
    return interval_multiset(*rows, budget, config['max_candidate_edges'])


def aligned_control(original, changed, new_to_old, mode):
    n = len(new_to_old)
    expected = original.reshape(n, n, -1)[np.ix_(new_to_old, new_to_old)].reshape(changed.shape)
    if mode == 'exact':
        if not np.array_equal(expected, changed):
            raise AssertionError('Exact same-graph aligned features changed')
        return {'aligned_exact_equal': True}
    if mode == 'interval':
        if not all(a.overlaps(b) for a, b in zip(expected.flat, changed.flat)):
            raise AssertionError('Same-graph valid enclosures are disjoint')
        return {'aligned_enclosures_overlap': True,
                'scope': 'Necessary enclosure consistency, not an equality certificate.'}
    return {'aligned_max_absolute_error': float(np.max(np.abs(expected - changed)))}


def relabel_controls(graphs, originals, q, basis, variant, mode, bits, config, pid, budget):
    result = []
    for graph_index, (graph, original) in enumerate(zip(graphs, originals)):
        for permutation in permutations(len(graph), config['relabels'], config['seed'] + 1009 * pid + graph_index):
            changed = feature_matrix(graph[np.ix_(permutation, permutation)], q, 2, variant,
                                     basis, mode, bits, budget)
            result.append({'graph': graph_index, 'new_index_to_old': permutation.tolist(),
                           **aligned_control(original, changed, permutation, mode)})
    return result


def reliability_control(data, pid, q, basis, variant, mode, bits, config, budget):
    import networkx as nx
    graphs = [graph_at(data, (pid + 400) * 64 + i) for i in range(2)]
    if mode == 'exact' and not all(is_regular(g) for g in graphs):
        mode = 'interval'
    matcher = nx.algorithms.isomorphism.GraphMatcher(nx.from_numpy_array(graphs[0]),
                                                    nx.from_numpy_array(graphs[1]))
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        raise ValueError('The official reliability representatives are not isomorphic')
    # The isomorphism is used ONLY to align a known-isomorphic control. It is
    # never a preprocessing step or input to the main graph comparison.
    new_to_old = np.argsort([mapping[i] for i in range(len(graphs[0]))])
    original = feature_matrix(graphs[0], q, 2, variant, basis, mode, bits, budget)
    changed = feature_matrix(graphs[1], q, 2, variant, basis, mode, bits, budget)
    result = [{'official_representatives': aligned_control(original, changed, new_to_old, mode),
               'control_old_to_new': [mapping[i] for i in range(len(graphs[0]))]}]
    for permutation in permutations(len(graphs[0]), config['reliability_relabels'], config['seed'] + 7919 * pid):
        relabelled = feature_matrix(graphs[0][np.ix_(permutation, permutation)], q, 2,
                                    variant, basis, mode, bits, budget)
        result.append({'new_index_to_old': permutation.tolist(),
                       **aligned_control(original, relabelled, permutation, mode)})
    return result


def evaluate_condition(data, pid, basis, variant, q, readout, p, config):
    budget = Budget(config['seconds_per_condition'], config['memory_gib'])
    graphs = [graph_at(data, 64 * pid + i) for i in range(2)]
    mode = 'exact' if all(is_regular(g) for g in graphs) else 'interval'
    precisions = [None] if mode == 'exact' else config['precision_bits']
    attempts = []
    for bits in precisions:
        rows = [feature_matrix(g, q, 2, variant, basis, mode, bits or 128, budget) for g in graphs]
        decision = decide(rows, readout, p, mode, budget, config)
        attempts.append({'bits': bits, 'status': decision['status'], 'reason': decision.get('reason')})
        if decision['status'] != UNRESOLVED:
            break
    # A detected difference cannot be promoted before its same-graph controls.
    relabels = relabel_controls(graphs, rows, q, basis, variant, mode, bits or 128, config, pid, budget)
    reliability = reliability_control(data, pid, q, basis, variant, mode, bits or 128, config, budget)
    return {**decision, 'arithmetic': mode, 'precision_attempts': attempts,
            'relabel_controls': relabels, 'official_reliability_controls': reliability,
            'width': rows[0].shape[1], 'elapsed_seconds': time.monotonic() - budget.started}


def experiment_settings(config):
    for basis, variant, q in itertools.product(config['bases'], config['variants'], config['q_values']):
        yield basis, variant, q, 'multiset', 1
        for p in config['mean_p_values']:
            yield basis, variant, q, 'mean', p


def summarize(out, config):
    summaries = []
    for basis, variant, q, readout, p in experiment_settings(config):
        stem = f'{basis}_{variant}_q{q}_{readout}_p{p}'
        counts = Counter()
        for pid in range(400):
            path = out / 'records' / f'{stem}_pair{pid:03d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                counts[record['status']] += 1
            else:
                counts['not_executed'] += 1
        assert sum(counts.values()) == 400
        summaries.append({'setting': stem, 'denominator': 400, 'counts': dict(counts),
                          'certified_detection_lower_bound': counts[DIFFERENT],
                          'detection_upper_bound': 400 - counts[EQUAL]})
    atomic_json(out / 'summary.json', {
        'settings': summaries,
        'scope': 'New order-2 protocol; unresolved/resource-limited items remain in denominator. Lower-bound rankings do not establish true expressivity rankings.'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rebuttal/brec.json')
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--pair-ids', nargs='+', type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = configuration(args.config)
    config = {**config, 'determinism': start_brec(config['seed'])}
    ids = args.pair_ids if args.pair_ids is not None else list(range(400))
    if not ids or len(set(ids)) != len(ids) or any(x < 0 or x >= 400 for x in ids):
        parser.error('Distinct pair IDs in 0..399 required')
    config = {**config, 'executed_pair_ids': ids}
    data = load_brec_data(args.brec_file, config['expected_brec_sha256'])
    with run_directory(args.out, config, [args.config, args.brec_file], args.resume) as (out, identity):
        for basis, variant, q, readout, p in experiment_settings(config):
            for pid in ids:
                stem = f'{basis}_{variant}_q{q}_{readout}_p{p}_pair{pid:03d}'
                target = out / 'records' / (stem + '.json')
                if target.exists():
                    old = json.loads(target.read_text())
                    if old['identity_sha256'] != identity:
                        raise ValueError('Record identity mismatch')
                    continue
                row = {'pair_id': pid, 'basis': basis, 'variant': variant,
                       'q': q, 'order': 2, 'readout': readout, 'p': p, 'identity_sha256': identity}
                try:
                    with hard_wall_limit(config['seconds_per_condition']):
                        result = evaluate_condition(data, pid, basis, variant, q, readout, p, config)
                    row.update(result)
                except (ResourceLimit, MemoryError) as exc:
                    row.update(status='resource_limited', reason=str(exc))
                except Exception as exc:
                    atomic_json(target, {**row, 'status': 'implementation_error', 'reason': repr(exc)})
                    summarize(out, config)
                    raise
                atomic_json(target, row)
                print(f'{stem}: {row["status"]}', flush=True)
        summarize(out, config)


if __name__ == '__main__':
    main()
