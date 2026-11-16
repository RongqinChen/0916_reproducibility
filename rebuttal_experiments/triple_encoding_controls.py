"""Exact bank alignment and sampled full-row relabeling controls."""
from __future__ import annotations

import random

import numpy as np

from .brec import graph_at, reliability_mapping
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


def controls(data,pid,kernels,q,basis,config,budget):
    keys = descriptors(q,3,'joint')
    n = kernels[0].n
    rng = random.Random(config['seed']+pid)
    indices = rng.sample(range(n**3),min(config['control_triples'],n**3))
    samples = [tuple(map(int,np.unravel_index(index,(n,n,n)))) for index in indices]
    result = []
    for graph_index,original in enumerate(kernels):
        for index,permutation in enumerate(permutations(n,config['relabels'],
                                              config['seed']+1009*pid+graph_index)):
            budget.check()
            report_phase('triple_relabel_control',q=q,basis=basis,graph=graph_index,relabel=index+1)
            changed = TripleKernel(original.adjacency[np.ix_(permutation,permutation)],q,basis,budget)
            result.append({'graph':graph_index,'new_to_old':permutation.tolist(),
                           **aligned_kernel_control(original,changed,np.argsort(permutation),samples,keys)})
    graphs = [graph_at(data,64*(pid+400)+i) for i in (0,1)]
    mapping = reliability_mapping(*graphs)
    original,changed = [TripleKernel(graph,q,basis,budget) for graph in graphs]
    report_phase('triple_reliability_control',q=q,basis=basis)
    result.append({'official_reliability':True,'old_to_new':mapping.tolist(),
                   **aligned_kernel_control(original,changed,mapping,samples,keys)})
    for index,permutation in enumerate(permutations(n,config['reliability_relabels'],
                                                   config['seed']+7919*pid)):
        budget.check()
        report_phase('triple_reliability_relabel',q=q,basis=basis,relabel=index+1)
        changed = TripleKernel(original.adjacency[np.ix_(permutation,permutation)],q,basis,budget)
        result.append({'reliability_relabel':index+1,'new_to_old':permutation.tolist(),
                       **aligned_kernel_control(original,changed,np.argsort(permutation),samples,keys)})
    return result
