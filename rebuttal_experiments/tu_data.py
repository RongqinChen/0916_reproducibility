"""TU text/zip input preserving categorical and continuous attributes.

Only explicit prepare commands with --download may access the network.
No PyG/GraphGym dependency is required by the supplementary trainer.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .common import atomic_npz, sha256

NAMES = ('ENZYMES', 'PROTEINS', 'PROTEINS_full')


def input_files(root, name, download=False):
    if name not in NAMES:
        raise ValueError(f'Supported protein inputs: {NAMES}')
    root = Path(root)
    for directory in (root / name / 'raw', root / name, root):
        if (directory / f'{name}_A.txt').exists():
            files = {p.name: p.read_bytes() for p in directory.glob(f'{name}_*.txt')}
            return files, {'kind': 'local_tu_text', 'sha256': {
                filename: hashlib.sha256(data).hexdigest() for filename, data in files.items()}}
    archive = root / f'{name}.zip'
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f'Place {name}.zip or TU raw text under {root}, or use explicit --download on the server')
        root.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix=name + '.', suffix='.download', dir=root)
        os.close(descriptor)
        temporary = Path(filename)
        url = f'https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip'
        try:
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            if archive.exists():
                temporary.unlink()
            else:
                temporary.replace(archive)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    with zipfile.ZipFile(archive) as source:
        wanted = [item for item in source.namelist()
                  if Path(item).name.startswith(name + '_') and item.endswith('.txt')]
        names = [Path(item).name for item in wanted]
        if len(names) != len(set(names)):
            raise ValueError('Ambiguous duplicate TU members')
        # Read named members, never extract arbitrary paths.
        files = {Path(item).name: source.read(item) for item in wanted}
    return files, {'kind': 'tu_zip', 'archive_sha256': sha256(archive)}


def _table(files, filename, dtype, required=False):
    if filename not in files:
        if required:
            raise FileNotFoundError(filename)
        return None
    text = files[filename].decode('utf-8').strip()
    if not text:
        return np.empty((0, 0), dtype=dtype)
    result = np.loadtxt(io.StringIO(text.replace(',', ' ')), dtype=dtype, ndmin=2)
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite values in {filename}')
    return result


def _attributes(labels, continuous, count):
    blocks = [np.ones((count, 1), dtype='float64')]
    vocabularies = []
    if labels is not None:
        if len(labels) != count:
            raise ValueError('Categorical attribute row mismatch')
        for column in labels.T:
            vocabulary = sorted(set(column.tolist()))
            blocks.append((column[:, None] == np.asarray(vocabulary)[None]).astype('float64'))
            vocabularies.append(vocabulary)
    start = sum(block.shape[1] for block in blocks)
    if continuous is not None:
        if len(continuous) != count:
            raise ValueError('Continuous attribute row mismatch')
        blocks.append(continuous.astype('float64'))
    width = sum(block.shape[1] for block in blocks)
    return np.concatenate(blocks, 1), list(range(start, width)), vocabularies


def load_tu(root, name, download=False):
    files, provenance = input_files(root, name, download)
    indicator = _table(files, name + '_graph_indicator.txt', 'int64', True).reshape(-1) - 1
    labels = _table(files, name + '_graph_labels.txt', 'int64', True).reshape(-1)
    edges = _table(files, name + '_A.txt', 'int64', True)
    if edges.shape[1] != 2:
        raise ValueError('TU adjacency must have two columns')
    edges = edges.astype('int64') - 1
    n, total = len(indicator), len(labels)
    if n == 0 or indicator.min() < 0 or indicator.max() >= total or set(indicator) != set(range(total)):
        raise ValueError('Invalid graph indicators')
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Invalid global edge endpoint')
    x, continuous_x, node_vocab = _attributes(
        _table(files, name + '_node_labels.txt', 'int64'),
        _table(files, name + '_node_attributes.txt', 'float64'), n)
    edge_x, continuous_e, edge_vocab = _attributes(
        _table(files, name + '_edge_labels.txt', 'int64'),
        _table(files, name + '_edge_attributes.txt', 'float64'), len(edges))
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError('A classification task needs at least two classes')
    y = np.searchsorted(classes, labels)
    indexes = [np.flatnonzero(indicator == i) for i in range(total)]
    local_index = np.zeros(n, dtype='int64')
    for group in indexes:
        local_index[group] = np.arange(len(group))
    edge_maps = [dict() for _ in range(total)]
    removed_self_loops = 0
    for (u, v), attributes in zip(edges, edge_x):
        if indicator[u] != indicator[v]:
            raise ValueError('Edge connects different graph records')
        if u == v:
            removed_self_loops += 1
            continue
        graph = int(indicator[u])
        key = (int(local_index[u]), int(local_index[v]))
        old = edge_maps[graph].get(key)
        if old is not None and not np.array_equal(old, attributes):
            raise ValueError('Duplicate edge carries inconsistent native attributes')
        edge_maps[graph][key] = attributes
    graphs, ordered_duplicates = [], {}
    for graph_id, (nodes, edge_map) in enumerate(zip(indexes, edge_maps)):
        for (u, v), attributes in list(edge_map.items()):
            reverse = edge_map.get((v, u))
            if reverse is not None and not np.array_equal(reverse, attributes):
                raise ValueError('Undirected edge orientations carry different native attributes')
            edge_map[v, u] = attributes
        keys = sorted(edge_map)
        edge_index = np.asarray(keys, dtype='int64').reshape(-1, 2).T
        attributes = np.stack([edge_map[k] for k in keys]) if keys else np.empty((0, edge_x.shape[1]))
        adjacency = np.zeros((len(nodes), len(nodes)), dtype='int64')
        adjacency[edge_index[0], edge_index[1]] = 1
        record = {'graph_id': graph_id, 'adjacency': adjacency, 'x': x[nodes],
                  'edge_index': edge_index, 'edge_attr': attributes, 'y': int(y[graph_id])}
        digest = hashlib.sha256()
        for value in (adjacency, record['x'], edge_index, attributes):
            digest.update(np.ascontiguousarray(value).tobytes())
        record['ordered_input_sha256'] = digest.hexdigest()
        ordered_duplicates.setdefault(record['ordered_input_sha256'], []).append(graph_id)
        graphs.append(record)
    metadata = {'dataset': name, 'graph_count': total, 'classes': classes,
                'node_feature_dim': x.shape[1], 'edge_feature_dim': edge_x.shape[1],
                'continuous_node_columns': continuous_x, 'continuous_edge_columns': continuous_e,
                'node_label_vocabularies': node_vocab, 'edge_label_vocabularies': edge_vocab,
                'native_attribute_policy': 'constant + native categorical one-hot + all native continuous attributes',
                'categorical_vocab_policy': 'dataset input vocabulary; no target labels used for feature encoding',
                'removed_self_loops': removed_self_loops,
                'topology_policy': 'simple undirected; merge identical parallel edges; preserve native edge attributes',
                'ordered_duplicate_groups': [g for g in ordered_duplicates.values() if len(g) > 1],
                'duplicate_scope': 'Exact ordered-input bytes only; not a label-invariant duplicate or homology audit',
                'provenance': provenance}
    return graphs, metadata


def save_graph(path, graph):
    atomic_npz(path, adjacency=graph['adjacency'], x=graph['x'], edge_index=graph['edge_index'],
               edge_attr=graph['edge_attr'], y=np.asarray(graph['y']), graph_id=np.asarray(graph['graph_id']))


def load_graph(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def stratified_splits(labels, folds=5, validation_fraction=0.2, seed=20261112):
    import random
    labels = np.asarray(labels)
    rng = random.Random(seed)
    tests = [[] for _ in range(folds)]
    for label in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == label).tolist()
        if len(ids) < folds:
            raise ValueError('Every class needs at least as many records as outer folds')
        rng.shuffle(ids)
        for index, graph_id in enumerate(ids):
            tests[index % folds].append(graph_id)
    result = []
    for fold, test in enumerate(tests):
        train, validation = [], []
        heldout = set(test)
        inner = random.Random(seed + 1009 * (fold + 1))
        for label in sorted(set(labels.tolist())):
            ids = [int(i) for i in np.flatnonzero(labels == label) if i not in heldout]
            inner.shuffle(ids)
            count = max(1, int(round(len(ids) * validation_fraction)))
            if count >= len(ids):
                raise ValueError('Inner split leaves a class without training records')
            validation.extend(ids[:count])
            train.extend(ids[count:])
        assert not (set(train) & set(validation) or set(train) & heldout or set(validation) & heldout)
        assert sorted(train + validation + test) == list(range(len(labels)))
        result.append({'fold': fold, 'train': sorted(train), 'validation': sorted(validation), 'test': sorted(test)})
    return result
