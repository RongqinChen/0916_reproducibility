"""Experiments 1--2: exact implementation controls for Appendix L.1/L.2."""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from .readouts import DIFFERENT, EQUAL, exact_multiset, mean_decision
from .common import Budget, atomic_json, configuration, hard_wall_limit, permutations, run_directory
from .encodings import constructed_graphs, descriptors, feature_matrix, moments


def basis_correspondence(before, after):
    """Check the invertible degree-one change, row by row, in both directions."""
    keys = descriptors(1, 3, 'joint')
    lookup = {key: i for i, key in enumerate(keys)}
    for row, changed in zip(before, after):
        for column, key in enumerate(keys):
            if key[0] == 'pair' and key[1] == 'B1':
                identity = lookup[('pair', 'I', key[2], key[3])]
                assert changed[column] == row[column] + row[identity]
                assert row[column] == changed[column] - changed[identity]
            elif key[0] == 'joint':
                indices = list(itertools.product(*(range(a + 1) for a in key[1:])))
                assert changed[column] == sum(row[lookup[('joint', *abc)]] for abc in indices)
                assert row[column] == sum((-1)**(sum(key[1:]) - sum(abc))
                                          * changed[lookup[('joint', *abc)]] for abc in indices)
            else:
                assert changed[column] == row[column]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rebuttal/mechanisms.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = configuration(args.config)
    graphs = constructed_graphs()
    records = []
    with run_directory(args.out, config, [args.config], args.resume) as (out, identity):
        budget = Budget(config['seconds'], config['memory_gib'])
        with hard_wall_limit(config['seconds']):
            for basis in config['bases']:
                base = [feature_matrix(g, 1, 3, 'endpoint', basis, 'exact', budget=budget) for g in graphs]
                joint = [feature_matrix(g, 1, 3, 'joint', basis, 'exact', budget=budget) for g in graphs]
                raw_base, raw_joint = exact_multiset(*base), exact_multiset(*joint)
                assert raw_base['status'] == EQUAL and raw_joint['status'] == DIFFERENT
                moment_records = []
                for power in config['moment_budgets']:
                    result = mean_decision(*(moments(x, power, budget) for x in joint))
                    assert result['status'] == EQUAL
                    moment_records.append({'p': power, **result})
                relabel_records = []
                for graph_index, (graph, original) in enumerate(zip(graphs, joint)):
                    for permutation in permutations(len(graph), config['relabels'], config['seed'] + graph_index):
                        changed = feature_matrix(graph[np.ix_(permutation, permutation)], 1, 3,
                                                 'joint', basis, 'exact', budget=budget)
                        check = exact_multiset(original, changed)
                        assert check['status'] == EQUAL
                        relabel_records.append({'graph': graph_index,
                                                'new_index_to_old': permutation.tolist(), 'status': EQUAL})
                records.append({'basis': basis, 'q': 1, 'order': 3,
                                'widths': [base[0].shape[1], joint[0].shape[1]],
                                'base': raw_base, 'joint': raw_joint,
                                'moments': moment_records, 'relabels': relabel_records})
            original = [feature_matrix(g, 1, 3, 'joint', 'monomial', 'exact', budget=budget) for g in graphs]
            shifted = [feature_matrix(g, 1, 3, 'joint', 'shifted', 'exact', budget=budget) for g in graphs]
            for left, right in zip(original, shifted):
                basis_correspondence(left, right)
            old_mean = mean_decision(*(moments(x, 3, budget) for x in original))
            new_mean = mean_decision(*(moments(x, 3, budget) for x in shifted))
            assert old_mean['status'] == EQUAL and new_mean['status'] == DIFFERENT
            atomic_json(out / 'summary.json', {
                'identity_sha256': identity, 'status': 'controls_passed',
                'graph_names': ['rook_4_4', 'shrikhande'], 'adjacencies': [g.tolist() for g in graphs],
                'finite_computational_controls': records,
                'basis_change': {'q': 1, 'p': 3, 'width': shifted[0].shape[1],
                                 'invertible_rowwise_correspondence_checked': True,
                                 'old_mean': old_mean, 'new_mean': new_mean},
                'scope': 'Implementation checks of existing exact results; not a proof for all Q/P, a full BREC audit, or prediction.'})
    print(f'Exact controls saved in {args.out}')


if __name__ == '__main__':
    main()
