"""Exact bank alignment and sampled full-row relabeling controls."""
from __future__ import annotations

from functools import partial
import random

import numpy as np

from .brec_data import OFFICIAL_RELIABILITY, official_reliability_pairs, reliability_mapping
from .common import permutations
from .encodings import descriptors
from .runtime import report_phase
from .triple_exact import TripleKernel


def aligned_kernel_control(original, changed, mapping, samples, keys):
    inverse = np.argsort(mapping)
    if not np.array_equal(original.adjacency[np.ix_(inverse,inverse)],changed.adjacency):
        raise AssertionError('Control adjacency does not match its permutation')
    for first,second in zip(original.banks,changed.banks):
        if not np.array_equal(first[np.ix_(inverse,inverse)],second):
            raise AssertionError('Exact scaled polynomial bank changed under relabeling')
    for triple in samples:
        transformed = tuple(int(mapping[v]) for v in triple)
        for key in keys:
            if original.value(triple,key) != changed.value(transformed,key):
                raise AssertionError(f'Exact triple feature changed under relabeling: {key}')
    return {'all_bank_entries_exact_equal': True, 'sampled_full_rows_exact_equal': True,
            'sampled_triples': len(samples), 'channels_per_sample': len(keys)}


def sample_triples(n, seed, count):
    indices = random.Random(seed).sample(range(n**3), min(count, n**3))
    return [tuple(map(int, np.unravel_index(index, (n, n, n)))) for index in indices]


def controls(data,pid,kernels,q,basis,config,budget):
    make_kernel = partial(TripleKernel, q=q, basis=basis, budget=budget,
                          bank_cache=config.get('bank_cache'))
    keys = descriptors(q,3,'joint')
    result = []
    seed = config['seed']
    count = config['control_triples']
    if config.get('relabels', 0):
        for graph_index,original in enumerate(kernels):
            samples = sample_triples(original.n, seed + pid, count)
            for index,permutation in enumerate(permutations(original.n, config['relabels'],
                                                            seed + 1009 * pid + graph_index)):
                budget.check()
                report_phase('triple_relabel_control',q=q,basis=basis,graph=graph_index,relabel=index+1)
                changed = make_kernel(original.adjacency[np.ix_(permutation,permutation)])
                result.append({'graph':graph_index,'new_to_old':permutation.tolist(),
                               **aligned_kernel_control(original,changed,np.argsort(permutation),samples,keys)})
                del changed
    extra = config.get('reliability_relabels', 0)
    for indices, graphs in official_reliability_pairs(data, pid, config):
        budget.check()
        report_phase('triple_reliability_control',q=q,basis=basis,dataset_indices=indices)
        mapping = reliability_mapping(*graphs)
        original,changed = [make_kernel(graph) for graph in graphs]
        samples = sample_triples(original.n, seed + pid, count)
        control = {'official_reliability': True, 'old_to_new': mapping.tolist(),
                   **aligned_kernel_control(original,changed,mapping,samples,keys)}
        if config.get('reliability_protocol') == OFFICIAL_RELIABILITY:
            control['dataset_indices'] = list(indices)
        result.append(control)
        del changed
        permutations_to_check = permutations(original.n, extra, seed + 7919 * pid) if extra else ()
        for index,permutation in enumerate(permutations_to_check):
            budget.check()
            report_phase('triple_reliability_relabel',q=q,basis=basis,relabel=index+1)
            changed = make_kernel(original.adjacency[np.ix_(permutation,permutation)])
            result.append({'reliability_relabel':index+1,'new_to_old':permutation.tolist(),
                           **aligned_kernel_control(original,changed,np.argsort(permutation),samples,keys)})
            del changed
        del original
    return result
