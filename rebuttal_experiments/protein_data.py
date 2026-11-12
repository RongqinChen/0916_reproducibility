"""Prepare protein graphs and stream full-triple encoding-grid inputs."""
from __future__ import annotations

import json
import shutil
import time
from contextlib import ExitStack
from functools import lru_cache, partial
from itertools import product
from pathlib import Path

import numpy as np

from .common import (Budget, ResourceLimit, atomic_json, atomic_output,
                     configuration, hard_wall_limit, run_directory)
from .encodings import descriptors, feature_chunks, normalized_matrix
from .protein_encodings import (ENCODINGS, PROTOCOL, VARIANTS, WIDTH,
                                canonical_rows, channel_projection,
                                project_rows, validate_config)
from .tu_data import load_graph, load_tu, save_graph, stratified_splits


def write_triple_features(adjacency, directory, config, budget):
    """Cache 25 seed lifts and 27 joint channels; derive endpoints on demand."""
    started = time.monotonic()
    n, rows = len(adjacency), len(adjacency)**3
    start = time.monotonic()
    matrix = normalized_matrix(adjacency, 'float64')
    bank = np.stack((np.eye(n), matrix, matrix @ matrix))
    h1 = matrix.sum(1)
    chunk = config['feature_chunk_size']
    base_blocks = feature_chunks(adjacency, bank, 2, 3, 'polynomial',
                                chunk_size=chunk, budget=budget)
    base_seconds, joint_seconds = time.monotonic() - start, 0.0

    @lru_cache(maxsize=1)
    def joint_plane(vertex):
        budget.check(n*n*27*8 + chunk*WIDTH*8)
        plane = np.empty((n, n, 27), dtype='float64')
        for column, (a, b, c) in enumerate(product(range(3), repeat=3)):
            if a == 0:
                plane[:, :, column] = bank[b, :, vertex, None] * bank[c, None, :, vertex]
            elif b == 0:
                plane[:, :, column] = bank[a, vertex, :, None] * bank[c].T
            elif c == 0:
                plane[:, :, column] = bank[a, vertex, None, :] * bank[b]
            else:
                support = np.flatnonzero(bank[a, vertex])
                left = bank[b][:, support] * bank[a, vertex, support]
                plane[:, :, column] = left @ bank[c][:, support].T
        return plane

    sums, gram = np.zeros(WIDTH), np.zeros((WIDTH, WIDTH))
    with ExitStack() as stack:
        streams = {}
        for name, width in (('base', 25), ('joint', 27)):
            temporary = stack.enter_context(atomic_output(directory / (name + '.npy')))
            streams[name] = stack.enter_context(temporary.open('wb'))
            np.lib.format.write_array_header_2_0(streams[name], {
                'descr': np.dtype('float64').str, 'fortran_order': False,
                'shape': (rows, width)})
        for offset in range(0, rows, chunk):
            start = time.monotonic()
            _, base = next(base_blocks)
            coords = np.unravel_index(np.arange(offset, offset + len(base)), (n, n, n))
            base_seconds += time.monotonic() - start
            u, v, w = coords
            start = time.monotonic()
            joint = np.empty((len(base), 27), dtype='float64')
            for vertex in np.unique(u):
                selected = u == vertex
                joint[selected] = joint_plane(int(vertex))[v[selected], w[selected]]
            joint_seconds += time.monotonic() - start
            full = canonical_rows(base, joint, h1, coords)
            if not np.isfinite(full).all():
                raise FloatingPointError('Nonfinite structural cache')
            sums += full.sum(0)
            gram += full.T @ full
            streams['base'].write(base.tobytes(order='C'))
            streams['joint'].write(joint.tobytes(order='C'))
    return {
        'nodes': n, 'ordered_triples_including_repeats': rows,
        'base_channels': descriptors(2, 3, 'polynomial'),
        'endpoint_channels': descriptors(2, 3, 'endpoint')[25:],
        'joint_channels': descriptors(2, 3, 'joint')[-27:],
        'h1': h1.tolist(), 'channel_sum': sums.tolist(), 'channel_gram': gram.tolist(),
        'base_seconds': base_seconds, 'joint_extra_seconds': joint_seconds,
        'total_preparation_seconds': time.monotonic() - started,
        'cached_bytes': rows * 52 * 8,
        'width': WIDTH, 'arithmetic': 'float64',
        'budget_note': 'Shared Q=2 bank. All recipes use 64 model slots; preprocessing cost is reported separately.'}


def prepare(args):
    config = configuration(args.config)
    validate_config(config)
    dataset = args.dataset or config['dataset']
    graphs, metadata = load_tu(args.data_root, dataset, args.download)
    if len(graphs) != config['expected_graphs'][dataset]:
        raise ValueError('Dataset record count differs from the frozen protocol')
    config = {**config, 'dataset': dataset, 'input_provenance': metadata['provenance']}
    with run_directory(args.out, config, [args.config], args.resume) as (out, identity):
        atomic_json(out / 'dataset.json', metadata)
        split = stratified_splits([g['y'] for g in graphs], config['folds'], config['validation_fraction'], config['split_seed'])
        atomic_json(out / 'splits.json', split)
        remaining = [g for g in graphs if not (out / 'records' / f"graph{g['graph_id']:04d}.json").exists()]
        needed = sum(len(g['adjacency'])**3 * 52 * 8 + 4096 for g in remaining)
        free = shutil.disk_usage(out).free
        print(f'{dataset}: additional feature cache {needed / 2**30:.2f} GiB; free disk {free / 2**30:.2f} GiB', flush=True)
        if needed + int(config['cache_disk_reserve_gib'] * 2**30) > free:
            raise ResourceLimit('Insufficient disk for full triple caches plus reserve; choose another output disk')
        for graph in graphs:
            index = graph['graph_id']
            record_path = out / 'records' / f'graph{index:04d}.json'
            if record_path.exists():
                saved = json.loads(record_path.read_text())
                if saved['identity_sha256'] != identity or any(not (out / p).is_file() for p in saved['files']):
                    raise ValueError('Prepared graph identity mismatch or incomplete cache')
                continue
            with hard_wall_limit(config['prepare_seconds_per_graph']):
                budget = Budget(config['prepare_seconds_per_graph'], config['memory_gib'])
                directory = Path('features') / f'graph{index:04d}'
                cost = write_triple_features(graph['adjacency'], out / directory, config, budget)
            raw_path = Path('graphs') / f'graph{index:04d}.npz'
            save_graph(out / raw_path, graph)
            paths = [raw_path, directory / 'base.npy', directory / 'joint.npy']
            atomic_json(record_path, {'graph_id': index, 'identity_sha256': identity,
                                      'ordered_input_sha256': graph['ordered_input_sha256'],
                                      'files': [str(p) for p in paths], 'construction': cost})
            print(f'prepared graph {index + 1}/{len(graphs)}', flush=True)
        atomic_json(out / 'summary.json', {'status': 'prepared', 'graph_count': len(graphs),
                                          'identity_sha256': identity, 'dataset': dataset,
                                          'encoding_protocol': PROTOCOL, 'structural_width': WIDTH,
                                          'encodings': {name: ENCODINGS[name] for name in config['variants']},
                                          'split_policy': 'record-stratified outer folds; fixed inner validation; no homology grouping claim',
                                          'budget_note': 'Equal input width/backbone/readout/updates; no equal preprocessing-cost claim.'})


def load_prepared(path):
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    if summary.get('encoding_protocol') != PROTOCOL or summary.get('structural_width') != WIDTH:
        raise ValueError('Legacy 27-slot caches cannot be mixed into the 64-slot encoding grid; use protein_encodings outputs')
    manifest = json.loads((path / 'manifest.json').read_text())
    if summary['status'] != 'prepared' or manifest['status'] != 'execution_finished':
        raise ValueError('Preparation must finish before training')
    if summary['identity_sha256'] != manifest['identity_sha256']:
        raise ValueError('Prepared summary and manifest do not match')
    return summary, manifest


class ProteinInputs:
    """Load small graph records and one structural block at a time."""

    def __init__(self, prepared, variant, config):
        validate_config(config)
        if variant not in config['variants']:
            raise ValueError(variant)
        self.root, self.variant, self.config = Path(prepared), variant, config
        self.projection = channel_projection(variant)
        self.uses_joint = bool(np.any(self.projection[:, 37:]))
        self.metadata = json.loads((self.root / 'dataset.json').read_text())

    @lru_cache(maxsize=8)
    def graph(self, index):
        return load_graph(self.root / 'graphs' / f'graph{index:04d}.npz')

    @lru_cache(maxsize=8)
    def construction(self, index):
        return json.loads((self.root / 'records' / f'graph{index:04d}.json').read_text())['construction']

    def structural_block(self, index, start):
        n = self.construction(index)['nodes']
        stop = min(n**3, start + self.config['structural_chunk_size'])
        coords = np.unravel_index(np.arange(start, stop), (n, n, n))
        if self.variant == 'none':
            return np.zeros((stop-start, WIDTH)), coords
        directory = self.root / 'features' / f'graph{index:04d}'
        base = np.load(directory / 'base.npy', mmap_mode='r', allow_pickle=False)[start:stop]
        joint = (np.load(directory / 'joint.npy', mmap_mode='r', allow_pickle=False)[start:stop]
                 if self.uses_joint else None)
        full = canonical_rows(base, joint, np.asarray(self.construction(index)['h1']), coords)
        return project_rows(full, self.projection), coords

    def normalization(self, train_ids):
        sums, gram, total = np.zeros(WIDTH), np.zeros((WIDTH, WIDTH)), 0
        for index in train_ids:
            record = self.construction(index)
            sums += np.asarray(record['channel_sum'])
            gram += np.asarray(record['channel_gram'])
            total += record['ordered_triples_including_repeats']
        projection = self.projection
        mean, std = np.zeros(WIDTH), np.ones(WIDTH)
        if len(projection):
            mean[:len(projection)] = projection @ sums / total
            second = np.einsum('ij,jk,ik->i', projection, gram, projection) / total
            variance = np.maximum(0, second - mean[:len(projection)]**2)
            scale = np.einsum('ij,jk,ik->i', np.abs(projection), np.abs(gram), np.abs(projection)) / total
            constant = variance <= 64 * np.finfo('float64').eps * scale
            std[:len(projection)] = np.where(constant, 1, np.sqrt(variance))
        result = {'structural_mean': mean.tolist(), 'structural_std': std.tolist()}
        for key, columns in [('x', self.metadata['continuous_node_columns']),
                             ('edge_attr', self.metadata['continuous_edge_columns'])]:
            values = np.concatenate([self.graph(i)[key][:, columns] for i in train_ids], 0)
            average = values.mean(0) if values.size else np.zeros(len(columns))
            spread = values.std(0) if values.size else np.ones(len(columns))
            spread[spread < 1e-12] = 1
            result[key + '_columns'] = columns
            result[key + '_mean'], result[key + '_std'] = average.tolist(), spread.tolist()
        result['fitted_graph_ids'] = list(train_ids)
        result['policy'] = 'Training graphs only; full-triple sums/Gram; numerical variance floor relative to second moments'
        return result

    def tensor_block(self, index, start, mean, std, device):
        import torch
        block, coords = self.structural_block(index, start)
        return (torch.as_tensor((block - mean) / std, dtype=torch.float32, device=device),
                torch.as_tensor(np.stack(coords), dtype=torch.long, device=device))

    def forward(self, model, index, normalization, device):
        import torch
        graph, values = self.graph(index), {}
        for key in ('x', 'edge_attr'):
            array = graph[key].astype('float64').copy()
            columns = normalization[key + '_columns']
            if columns:
                array[:, columns] = ((array[:, columns] - np.asarray(normalization[key + '_mean']))
                                      / np.asarray(normalization[key + '_std']))
            values[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
        mean, std = np.asarray(normalization['structural_mean']), np.asarray(normalization['structural_std'])
        rows = len(graph['adjacency'])**3
        chunks = (partial(self.tensor_block, index, start, mean, std, device)
                  for start in range(0, rows, self.config['structural_chunk_size']))
        return model(values['x'], torch.as_tensor(graph['edge_index'], dtype=torch.long, device=device),
                     values['edge_attr'], chunks, rows)
