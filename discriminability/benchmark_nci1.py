"""Prepare a deterministic NCI1 test subset and verify shared-mask equivalence.

Reads existing TU raw topology without torch. This is an engineering benchmark,
not a full-dataset scientific result. No graphs are selected by outcome.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from discriminability.features import graph_features, select_route, ROUTES
from discriminability.evaluation import separation_mask, shared_route_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.raw)
    labels = np.loadtxt(root / 'NCI1_graph_indicator.txt', dtype=np.int64)
    edges = np.loadtxt(root / 'NCI1_A.txt', dtype=np.int64, delimiter=',').T - 1
    counts, offsets, blocks, measurements = [], [0], [], []
    dims = [6, 8, 10, 12, 14]
    with threadpool_limits(limits=1):
        for graph_id in range(1, min(args.limit, int(labels.max())) + 1):
            vertices = np.flatnonzero(labels == graph_id)
            mapping = np.full(len(labels), -1, dtype=np.int64)
            mapping[vertices] = np.arange(len(vertices))
            block = mapping[edges[:, labels[edges[0]] == graph_id]]
            block = np.unique(block, axis=1)
            n = len(vertices)
            counts.append(n)
            blocks.append(block)
            offsets.append(offsets[-1] + block.shape[1])
            features = graph_features(block, n, max(dims))
            baseline_seconds = optimized_seconds = 0.0
            for kind, (p, e) in features.items():
                start = time.perf_counter()
                baseline = {(d, route): separation_mask(select_route(route, p, e, d),
                            1e-8, 1e-6, 128) for d in dims for route in ROUTES}
                baseline_seconds += time.perf_counter() - start
                start = time.perf_counter()
                optimized = shared_route_masks(p, e, dims, ROUTES, 1e-8, 1e-6, 128)
                optimized_seconds += time.perf_counter() - start
                for key, mask in baseline.items():
                    if not np.array_equal(np.packbits(mask), optimized[key]):
                        raise AssertionError((graph_id, kind, key))
            measurements.append(dict(graph_id=graph_id, nodes=n,
                                     baseline_seconds=baseline_seconds,
                                     optimized_seconds=optimized_seconds))
            print(measurements[-1], flush=True)
    np.savez_compressed(out / 'topology.npz', edges=np.concatenate(blocks, axis=1),
                        counts=np.array(counts), offsets=np.array(offsets),
                        provenance=np.array(json.dumps(dict(dataset='NCI1',
                        selection='first graphs in TU order', limit=args.limit,
                        raw_path=str(root.resolve()), purpose='engineering benchmark'))))
    (out / 'benchmark.json').write_text(json.dumps(measurements, indent=2))


if __name__ == '__main__':
    main()
