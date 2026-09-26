#!/usr/bin/env python
r"""Dump graph topology from supported PyG/OGB/GNNPlus datasets to portable npz.

The discriminability protocol only needs the unweighted topology of each graph,
so this tool reuses GNNPlus's dataset access code to fetch a dataset and writes a
compact npz that `discriminability.run` reads with `--dataset npz`.

    python -m discriminability.dump_topology --format PyG-TUDataset --name NCI1 \
        --root data --max-nodes 60 --limit 500 --out nci1.npz

`--format`/`--name` use the identifiers accepted by the project loaders. Standard
PyG datasets are loaded directly, which avoids importing GraphGym just to read
topology; project-specific datasets use GNNPlus adapters lazily. The
output npz holds `edges` (2 x total_edges, concatenated), `counts` (nodes per
graph), `offsets` (edge offsets per graph), `splits` (split label per graph) and
a `provenance` record naming the source and any node cap. This is the whole
format contract; the reader is `discriminability.pipeline.load_npz`.

To reproduce the archived MUTAG baseline byte for byte, use `--split-label tud`:
the baseline predates the split handling and labelled every graph `tud`.

Datasets whose payload is hosted on Dropbox cannot be fetched from a restricted
network; the error message names the blocked host so the failure is unambiguous.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def import_gnnplus(root: Path):
    """Import the GNNPlus preformat functions from a checkout."""
    root = root.resolve()
    if not (root / 'GNNPlus' / 'loader' / 'master_loader.py').exists():
        raise SystemExit(f'no GNNPlus package under {root}')
    sys.path.insert(0, str(root))
    from GNNPlus.loader import master_loader as ml
    return ml


class JoinedSplits:
    """Small list-like view that preserves train/validation/test membership."""

    def __init__(self, splits):
        self.splits = splits
        lengths = [len(split) for split in splits]
        starts = np.cumsum([0] + lengths)
        self.split_idxs = [np.arange(starts[i], starts[i + 1])
                           for i in range(len(splits))]

    def __len__(self):
        return sum(len(split) for split in self.splits)

    def __getitem__(self, index):
        for split in self.splits:
            if index < len(split):
                return split[index]
            index -= len(split)
        raise IndexError(index)


def build_dataset(fmt: str, name: str, root: str, slic_compactness: int,
                  gnnplus: Path):
    """Instantiate one dataset without importing GNNPlus when PyG is sufficient."""
    if fmt == 'PyG-TUDataset':
        from torch_geometric.datasets import TUDataset
        return TUDataset(root, name)
    if fmt == 'PyG-ZINC':
        if name not in ('subset', 'full'):
            raise SystemExit('PyG-ZINC name must be "subset" or "full"')
        from torch_geometric.datasets import ZINC
        return JoinedSplits([
            ZINC(root=root, subset=name == 'subset', split=split)
            for split in ('train', 'val', 'test')])
    if fmt == 'PyG-QM9':
        if name != 'QM9':
            raise SystemExit('PyG-QM9 name must be "QM9"')
        from torch_geometric.datasets import QM9
        return QM9(root=root)
    # The remaining formats use GNNPlus adapters or project-local dataset
    # classes. Import the larger package only when one of them is requested.
    ml = import_gnnplus(gnnplus)
    if fmt == 'PyG-AQSOL':
        return ml.preformat_AQSOL(root)
    if fmt == 'PyG-GNNBenchmarkDataset':
        return ml.preformat_GNNBenchmarkDataset(root, name)
    if fmt == 'PyG-MalNetTiny':
        return ml.preformat_MalNetTiny(root, name)
    if fmt == 'PyG-VOCSuperpixels':
        return ml.preformat_VOCSuperpixels(root, name, slic_compactness)
    if fmt == 'PyG-COCOSuperpixels':
        return ml.preformat_COCOSuperpixels(root, name, slic_compactness)
    if fmt == 'OGB':
        if name.startswith('ogbg'):
            return ml.preformat_OGB_Graph(root, name.replace('_', '-'))
        if name.startswith('PCQM4Mv2-'):
            return ml.preformat_OGB_PCQM4Mv2(root, name.split('-', 1)[1])
        if name.startswith('peptides-'):
            return ml.preformat_Peptides(root, name)
        raise SystemExit(f'unsupported OGB dataset: {name}')
    raise SystemExit(f'unsupported format: {fmt}')


def split_labels(dataset, total: int, override: str) -> np.ndarray:
    """Per-graph split name: an explicit override, else GNNPlus's `split_idxs`."""
    if override:
        return np.array([override] * total, dtype=str)
    labels = np.array(['all'] * total, dtype=object)
    idxs = getattr(dataset, 'split_idxs', None)
    if idxs is None:
        return labels.astype(str)
    for split, index in zip(('train', 'val', 'test'), idxs):
        for i in np.asarray(index, dtype=np.int64):
            if 0 <= i < total:
                labels[i] = split
    return labels.astype(str)


def topology_of(data):
    """(edge_index, num_nodes) as a simple undirected topology.

    The protocol's `adjacency_matrix` contract requires each undirected edge to
    appear in both orientations, exactly once each, so the unique undirected
    pairs are mirrored back after de-duplication.
    """
    edge_index = getattr(data, 'edge_index', None)
    n = int(data.num_nodes if getattr(data, 'num_nodes', None) is not None
            else (data.x.size(0) if getattr(data, 'x', None) is not None else 0))
    if edge_index is None or edge_index.numel() == 0 or n == 0:
        return np.zeros((2, 0), dtype=np.int32), n
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    keep = (src != dst) & (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    src, dst = src[keep], dst[keep]
    undirected = np.unique(np.stack([np.minimum(src, dst),
                                     np.maximum(src, dst)]), axis=1)
    both = np.concatenate([undirected, undirected[::-1]], axis=1)
    return both.astype(np.int32), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--format', required=True,
                        help='GNNPlus format id, e.g. PyG-TUDataset, OGB')
    parser.add_argument('--name', required=True,
                        help='GNNPlus dataset name, e.g. NCI1, ogbg-molhiv')
    parser.add_argument('--root', default='data', help='dataset cache directory')
    parser.add_argument('--gnnplus', default=str(REPO / 'gnnplus_official'),
                        help='GNNPlus checkout used only by custom dataset adapters')
    parser.add_argument('--slic-compactness', type=int, default=10)
    parser.add_argument('--max-nodes', type=int, default=0,
                        help='skip graphs larger than this (0 = no cap)')
    parser.add_argument('--limit', type=int, default=0,
                        help='keep at most this many graphs (0 = all)')
    parser.add_argument('--split-label', default='',
                        help="label every graph with this split name instead of "
                             "reading GNNPlus's split_idxs; the archived MUTAG "
                             "baseline was dumped with --split-label tud")
    parser.add_argument('--out', required=True, help='output npz path')
    args = parser.parse_args()

    dataset = build_dataset(args.format, args.name, args.root,
                            args.slic_compactness, Path(args.gnnplus))
    total = len(dataset)
    print(f'[dump] {args.format}/{args.name}: {total} graphs', flush=True)

    edges, counts, offsets, kept, skipped = [], [], [0], [], 0
    splits_all = split_labels(dataset, total, args.split_label)
    for i in range(total):
        if args.limit and len(kept) >= args.limit:
            break
        undirected, n = topology_of(dataset[i])
        if args.max_nodes and n > args.max_nodes:
            skipped += 1
            continue
        edges.append(undirected)
        counts.append(n)
        offsets.append(offsets[-1] + undirected.shape[1])
        kept.append(i)

    if not edges:
        raise SystemExit('no graph survived the node cap; raise --max-nodes')

    matrix = np.concatenate(edges, axis=1)
    # Carry the provenance with the topology: a node cap changes which graphs the
    # protocol numbers describe, so the run must be able to state it.
    provenance = json.dumps({
        'tool': 'dump_topology.py', 'format': args.format, 'name': args.name,
        'data_root': str(args.root), 'max_nodes': args.max_nodes,
        'limit': args.limit, 'split_label': args.split_label,
        'total_graphs': total, 'kept_graphs': len(counts),
        'skipped_over_max_nodes': skipped,
    })
    np.savez_compressed(args.out, edges=matrix,
                        counts=np.array(counts, dtype=np.int32),
                        offsets=np.array(offsets, dtype=np.int64),
                        splits=splits_all[kept],
                        provenance=np.array(provenance))
    nodes = np.array(counts)
    print(f'[dump] kept {len(counts)} graphs, skipped {skipped} over '
          f'--max-nodes {args.max_nodes}')
    print(f'[dump] nodes: min={nodes.min()} mean={nodes.mean():.1f} '
          f'max={nodes.max()}  edges={matrix.shape[1]}')
    print(f'[dump] wrote {args.out}')


if __name__ == '__main__':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
