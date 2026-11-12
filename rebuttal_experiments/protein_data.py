"""Prepare protein datasets, triple features and normalized model inputs."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import (Budget, ResourceLimit, atomic_json, atomic_npy,
                     configuration, hard_wall_limit, run_directory)
from .encodings import descriptors, feature_matrix, normalized_matrix
from .tu_data import load_graph, load_tu, save_graph, stratified_splits

VARIANTS = ('base', 'joint', 'nonjoint')


def build_triple_features(adjacency, config, budget=None):
    """Return the shared 19 channels and each eight-channel augmentation.

    Rows cover all ordered triples, including repeated vertices. The nonjoint
    augmentation uses degree two; equal width does not mean equal degree.
    """
    n = len(adjacency)
    rows = n**3
    if rows * 35 > config['max_cached_scalars_per_graph']:
        raise ResourceLimit('Full triple cache exceeds the frozen per-graph scalar limit; no graph may be silently dropped')
    if budget:
        budget.check(rows * 35 * 8)
    start = time.monotonic()
    base = feature_matrix(adjacency, 1, 3, 'polynomial', mode='float64', budget=budget)
    matrix = normalized_matrix(adjacency, 'float64')
    base_seconds = time.monotonic() - start
    start = time.monotonic()
    squared = matrix @ matrix
    h = matrix.sum(1)
    nonjoint = np.empty((rows, 8), dtype='float64')
    positions = [(i, j) for i in range(3) for j in range(3) if i != j]
    chunk = config['feature_chunk_size']
    for offset in range(0, rows, chunk):
        if budget:
            budget.check()
        ids = np.arange(offset, min(rows, offset + chunk))
        coords = np.unravel_index(ids, (n, n, n))
        for column, (i, j) in enumerate(positions):
            nonjoint[offset:offset + len(ids), column] = squared[coords[i], coords[j]]
        nonjoint[offset:offset + len(ids), 6] = h[coords[0]]
        nonjoint[offset:offset + len(ids), 7] = h[coords[1]]
    nonjoint_seconds = time.monotonic() - start
    start = time.monotonic()
    joint = np.empty((rows, 8), dtype='float64')
    for offset in range(0, rows, chunk):
        if budget:
            budget.check()
        ids = np.arange(offset, min(rows, offset + chunk))
        u, v, w = np.unravel_index(ids, (n, n, n))
        block = joint[offset:offset + len(ids)]
        # Lexicographic (a,b,c). The seven identity-containing channels
        # simplify exactly; the last is the shared-pivot contraction.
        block[:, 0] = (u == v) & (v == w)
        block[:, 1] = (u == v) * matrix[w, u]
        block[:, 2] = (u == w) * matrix[v, u]
        block[:, 3] = matrix[v, u] * matrix[w, u]
        block[:, 4] = (v == w) * matrix[u, v]
        block[:, 5] = matrix[u, v] * matrix[w, v]
        block[:, 6] = matrix[u, w] * matrix[v, w]
        block[:, 7] = (matrix[u] * matrix[v] * matrix[w]).sum(1)
    joint_seconds = time.monotonic() - start
    if not all(np.isfinite(x).all() for x in (base, joint, nonjoint)):
        raise FloatingPointError('Nonfinite structural cache')
    return base, joint, nonjoint, {
        'nodes': n, 'ordered_triples_including_repeats': rows,
        'base_channels': descriptors(1, 3, 'polynomial'),
        'joint_channels': descriptors(1, 3, 'joint')[-8:],
        'nonjoint_channels': [('pair', 'M2', i, j) for i, j in positions] + [('h1', 0), ('h1', 1)],
        'base_seconds': base_seconds, 'joint_extra_seconds': joint_seconds,
        'nonjoint_extra_seconds': nonjoint_seconds,
        'cached_bytes': sum(x.nbytes for x in (base, joint, nonjoint)),
        'width': 27, 'arithmetic': 'float64',
        'budget_note': 'Joint Q=1; nonjoint extras include degree-2 seeds. Same width/backbone/updates, construction cost not matched.'}


def prepare(args):
    """Save native inputs, fixed splits and the three structural feature caches."""
    config = configuration(args.config)
    dataset = args.dataset or config['dataset']
    graphs, metadata = load_tu(args.data_root, dataset, args.download)
    if len(graphs) != config['expected_graphs'][dataset]:
        raise ValueError('Dataset record count differs from the frozen protocol')
    config = {**config, 'dataset': dataset, 'input_provenance': metadata['provenance']}
    with run_directory(args.out, config, [args.config], args.resume) as (out, identity):
        atomic_json(out / 'dataset.json', metadata)
        split = stratified_splits([g['y'] for g in graphs], config['folds'], config['validation_fraction'], config['split_seed'])
        atomic_json(out / 'splits.json', split)
        for graph in graphs:
            index = graph['graph_id']
            record_path = out / 'records' / f'graph{index:04d}.json'
            if record_path.exists():
                saved = json.loads(record_path.read_text())
                if saved['identity_sha256'] != identity:
                    raise ValueError('Prepared graph identity mismatch')
                if any(not (out / p).is_file() for p in saved['files']):
                    raise FileNotFoundError('Prepared graph cache is incomplete')
                continue
            with hard_wall_limit(config['prepare_seconds_per_graph']):
                budget = Budget(config['prepare_seconds_per_graph'], config['memory_gib'])
                base, joint, nonjoint, cost = build_triple_features(graph['adjacency'], config, budget)
            raw_path = Path('graphs') / f'graph{index:04d}.npz'
            save_graph(out / raw_path, graph)
            paths = [raw_path]
            for name, array in [('base', base), ('joint', joint), ('nonjoint', nonjoint)]:
                path = Path('features') / f'graph{index:04d}' / (name + '.npy')
                atomic_npy(out / path, array)
                paths.append(path)
            atomic_json(record_path, {'graph_id': index, 'identity_sha256': identity,
                                      'ordered_input_sha256': graph['ordered_input_sha256'],
                                      'files': [str(p) for p in paths], 'construction': cost})
            print(f'prepared graph {index + 1}/{len(graphs)}', flush=True)
        atomic_json(out / 'summary.json', {'status': 'prepared', 'graph_count': len(graphs),
                                          'identity_sha256': identity, 'dataset': dataset,
                                          'split_policy': 'record-stratified outer folds; fixed inner validation; no homology grouping claim',
                                          'budget_note': 'Equal 27-slot model inputs and update budgets; construction degree/cost not matched.'})


def load_prepared(path):
    """Read the identity and configuration of a completed preparation run."""
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    if summary['status'] != 'prepared':
        raise ValueError('Prepare must complete before training')
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['status'] != 'execution_finished':
        raise ValueError('Preparation manifest is incomplete')
    if summary['identity_sha256'] != manifest['identity_sha256']:
        raise ValueError('Prepared summary and manifest do not match')
    return summary, manifest


class ProteinInputs:
    """Load one graph and stream its structural rows into the common model."""

    def __init__(self, prepared, variant, config):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.root, self.variant, self.config = Path(prepared), variant, config
        self.metadata = json.loads((self.root / 'dataset.json').read_text())

    @lru_cache(maxsize=8)
    def graph(self, index):
        return load_graph(self.root / 'graphs' / f'graph{index:04d}.npz')

    def arrays(self, index):
        directory = self.root / 'features' / f'graph{index:04d}'
        base = np.load(directory / 'base.npy', mmap_mode='r', allow_pickle=False)
        extra = None if self.variant == 'base' else np.load(directory / f'{self.variant}.npy', mmap_mode='r', allow_pickle=False)
        return base, extra

    def structural_chunks(self, index):
        base, extra = self.arrays(index)
        for start in range(0, len(base), self.config['structural_chunk_size']):
            first = np.asarray(base[start:start + self.config['structural_chunk_size']])
            second = (np.zeros((len(first), 8), dtype='float64') if extra is None
                      else np.asarray(extra[start:start + len(first)]))
            yield np.concatenate((first, second), 1)

    def normalization(self, train_ids):
        sums, squares, total = np.zeros(27), np.zeros(27), 0
        for index in train_ids:
            for block in self.structural_chunks(index):
                sums += block.sum(0)
                squares += np.square(block).sum(0)
                total += len(block)
        mean = sums / total
        std = np.sqrt(np.maximum(0, squares / total - mean**2))
        std[std < 1e-12] = 1
        result = {'structural_mean': mean.tolist(), 'structural_std': std.tolist()}
        for key, columns in [('x', self.metadata['continuous_node_columns']),
                             ('edge_attr', self.metadata['continuous_edge_columns'])]:
            values = np.concatenate([self.graph(i)[key][:, columns] for i in train_ids], 0)
            if values.size:
                average, spread = values.mean(0), values.std(0)
                spread[spread < 1e-12] = 1
            else:
                average, spread = np.zeros(len(columns)), np.ones(len(columns))
            result[key + '_columns'] = columns
            result[key + '_mean'], result[key + '_std'] = average.tolist(), spread.tolist()
        result['fitted_graph_ids'] = list(train_ids)
        result['policy'] = 'Training records only; structural rows pooled; continuous native attributes standardized'
        return result

    def forward(self, model, index, normalization, device):
        import torch
        graph = self.graph(index)
        values = {}
        for key in ('x', 'edge_attr'):
            array = graph[key].astype('float64').copy()
            columns = normalization[key + '_columns']
            if columns:
                array[:, columns] = ((array[:, columns] - np.asarray(normalization[key + '_mean']))
                                      / np.asarray(normalization[key + '_std']))
            values[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
        mean, std = np.asarray(normalization['structural_mean']), np.asarray(normalization['structural_std'])
        chunks = (torch.as_tensor((block - mean) / std, dtype=torch.float32, device=device)
                  for block in self.structural_chunks(index))
        return model(values['x'], torch.as_tensor(graph['edge_index'], dtype=torch.long, device=device),
                     values['edge_attr'], chunks, len(graph['adjacency'])**3)
