"""Complete weighted row comparisons and exact cumulative marginal moments."""
from __future__ import annotations

from .encodings import descriptors
from .readouts import DIFFERENT, EQUAL
from .runtime import report_phase
from .triple_exact import weighted_counts, moment_from_counts


def compare_kernels(kernels, triples, weights, q, basis, p_values, budget=None):
    """Compare weighted representatives covering the full ordered triple domain."""
    baseline_keys = descriptors(q,3,'endpoint')
    keys = descriptors(q,3,'joint')
    raw_colors = [[0]*len(rows) for rows in triples]
    raw_difference = None
    differences = {}
    results = {}
    raw_keys = []
    pair_marginals = {}
    n = kernels[0].n
    if kernels[1].n != n:
        raise ValueError('Equal node counts required for this BREC comparison')

    def snapshot(variant, width):
        result = {'multiset': {'status': DIFFERENT if raw_difference else EQUAL,
                              'evidence': raw_difference or {
                                  'kind': 'exact_weighted_triple_multiset',
                                  'objects': [n**3,n**3], 'checked_channels': len(raw_keys)}}}
        for p in p_values:
            result[f'mean_p{p}'] = ({'status': DIFFERENT, 'evidence': differences[p]}
                                    if p in differences else {'status': EQUAL,
                                    'evidence': {'kind': 'exact_radical_moments', 'through_p': p,
                                                 'width': width, 'objects_per_graph': n**3}})
        results[variant] = result

    for column,key in enumerate(keys):
        if budget:
            budget.check()
        report_phase('triple_channels', q=q, basis=basis, channel=column+1, total=len(keys),
                     descriptor=list(key), orbit_rows=[len(rows) for rows in triples])
        # Zero-index joint channels are functions of the retained baseline
        # row. Omitting them from matching preserves its exact partition.
        raw_needed = raw_difference is None and not (key[0]=='joint' and 0 in key[1:])
        powers_needed = [power for power in range(2,max(p_values)+1)
                         if any(p>=power and p not in differences for p in p_values)]
        # Each ordered pair occurs n times at any two distinct triple
        # positions. Reuse its marginal sums across positions; raw rows
        # still need each position's value to preserve their joint distribution.
        marginal = pair_marginals.setdefault(key[:-2], {}) if key[0] in ('pair','endpoint') else {}
        new_powers = any(power not in marginal for power in powers_needed)
        values = [kernel.column(rows,key) for kernel,rows in zip(kernels,triples)] if (
            raw_needed or new_powers) else None
        moment_counts = [weighted_counts(side,w) for side,w in zip(values,weights)] if new_powers else None
        if raw_needed:
            vocabulary = {}
            raw_colors = [[vocabulary.setdefault((old,value),len(vocabulary))
                           for old,value in zip(colors,side)]
                          for colors,side in zip(raw_colors,values)]
            counts = [weighted_counts(side,w) for side,w in zip(raw_colors,weights)]
            raw_keys.append(key)
            if counts[0] != counts[1]:
                witness = next(color for group in counts for color in group
                               if counts[0][color] != counts[1][color])
                side = 0 if witness in counts[0] else 1
                index = raw_colors[side].index(witness)
                row = triples[side][index]
                raw_difference = {'kind': 'exact_weighted_multiset_projection',
                                  'channels': [list(k) for k in raw_keys],
                                  'values': [kernels[side].value(row,k).json() for k in raw_keys],
                                  'multiplicities': [counts[0][witness],counts[1][witness]]}
        if any(p not in differences for p in p_values):
            if 1 not in marginal:
                marginal[1] = [kernel.mean_sum(key) for kernel in kernels]
            left,right = marginal[1]
            if left != right:
                for p in p_values:
                    if p not in differences:
                        differences[p] = {'kind': 'exact_radical_mean_difference',
                            'channel': column, 'descriptor': list(key), 'power': 1,
                            'left_sum': left.json(), 'right_sum': right.json(),
                            'difference_sum': (left-right).json(), 'normalizer': n**3}
        for power in powers_needed:
            if not any(p>=power and p not in differences for p in p_values):
                continue
            if power not in marginal:
                marginal[power] = [moment_from_counts(counts,power) for counts in moment_counts]
            left,right = marginal[power]
            if left != right:
                for p in p_values:
                    if p>=power and p not in differences:
                        differences[p] = {'kind': 'exact_radical_mean_difference',
                            'channel': column, 'descriptor': list(key), 'power': power,
                            'left_sum': left.json(), 'right_sum': right.json(),
                            'difference_sum': (left-right).json(), 'normalizer': n**3}
        if column+1==len(baseline_keys):
            snapshot('baseline',column+1)
    snapshot('joint',len(keys))
    return results
