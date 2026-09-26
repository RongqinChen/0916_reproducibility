"""BREC-first, encoding-only seed/explicit screening (not learned RPC).

Reads the official graph6 NPY ordering: 400 main blocks followed by 400
reliability blocks, each containing 32 alternating left/right relabelings.
No legacy project imports. Numerical signatures are rounded, not exact proofs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def polynomial_bank(adj, q, basis):
    if q < 1 or basis not in ("monomial", "chebyshev"):
        raise ValueError("Positive Q and a supported basis are required")
    adj = np.asarray(adj, dtype=np.float32)
    if not np.allclose(adj, adj.T) or np.any(adj < 0):
        raise ValueError("Expected undirected nonnegative adjacency")
    degree = adj.sum(1)
    inv = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inv, where=degree > 0)
    matrix = inv[:, None] * adj * inv[None, :]
    bank = [np.eye(len(adj), dtype=np.float32), matrix]
    for _ in range(2, q + 1):
        bank.append(matrix @ bank[-1] if basis == "monomial"
                    else 2 * matrix @ bank[-1] - bank[-2])
    return np.stack(bank)


def encode_bank(adj, bank, q, p=1, variant="seed"):
    if p < 1 or variant not in ("seed", "explicit"):
        raise ValueError("Invalid expansion or variant")
    adj = np.asarray(adj, dtype=np.float32)
    n = len(adj)
    # Indexed channels: I, J, A, S_1,...,S_Q.
    channels = [np.eye(n, dtype=np.float32), np.ones((n, n), dtype=np.float32), adj, *bank[1:q+1]]
    if variant == "explicit":
        h = bank[:q+1].sum(2)
        channels.extend(np.outer(h[a], h[b])
                        for a in range(q) for b in range(q-a)
                        if (a, b) != (0, 0))
    base = np.stack(channels, axis=-1)
    return np.stack([base ** power for power in range(1, p+1)],
                    axis=-1).reshape(n, n, -1)


def encode(adj, q, p=1, variant="seed", basis="monomial"):
    return encode_bank(adj, polynomial_bank(adj, q, basis), q, p, variant)


def signature(features, decimals):
    rows = np.round(features.reshape(-1, features.shape[-1]), decimals)
    if not np.isfinite(rows).all():
        raise FloatingPointError("Nonfinite encoding")
    rows[rows == 0] = 0  # Canonicalize signed zero before byte comparison.
    rows = rows[np.lexsort(rows.T[::-1])]
    return rows.shape, rows.tobytes()


def category(pid):
    for label, stop in (("Basic", 60), ("Regular", 160), ("Extension", 260),
                        ("CFI", 360), ("4-Vertex", 380), ("Distance", 400)):
        if pid < stop:
            return label
    raise ValueError(pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brec-file", type=Path, required=True)
    parser.add_argument("--trust-pickle", action="store_true",
                        help="Allow object NPY only for a trusted dataset file")
    parser.add_argument("--pair-ids", nargs="+", type=int, default=list(range(400)))
    parser.add_argument("--q-values", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--p-values", nargs="+", type=int, default=[1])
    parser.add_argument("--bases", nargs="+", choices=["monomial", "chebyshev"],
                        default=["monomial", "chebyshev"])
    parser.add_argument("--decimals", nargs="+", type=int, default=[8])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new path")
    if min(args.q_values + args.p_values) < 1 or any(not 0 <= i < 400 for i in args.pair_ids):
        parser.error("Invalid budget or pair ID")
    import networkx as nx
    raw = np.load(args.brec_file, allow_pickle=args.trust_pickle)
    if raw.ndim != 1 or len(raw) != 51200:
        parser.error("Expected the official 51200-entry graph6 NPY layout")

    def adjacency(index):
        value = raw[index]
        encoded = value.encode() if isinstance(value, str) else bytes(value)
        return nx.to_numpy_array(nx.from_graph6_bytes(encoded.strip()), dtype=np.float32)

    started = time.perf_counter()
    records = []
    # One representative pair per main/reliability block. This is a screening
    # test, NOT the full 32-relabel RPC reliability protocol.
    for pid in args.pair_ids:
        graphs = [adjacency(pid*64), adjacency(pid*64+1),
                  adjacency((pid+400)*64), adjacency((pid+400)*64+1)]
        for basis in args.bases:
            bank_started = time.perf_counter()
            banks = [polynomial_bank(a, max(args.q_values), basis) for a in graphs]
            bank_seconds = time.perf_counter() - bank_started
            for q in args.q_values:
                for p in args.p_values:
                    for variant in ("seed", "explicit"):
                        tick = time.perf_counter()
                        features = [encode_bank(a, b, q, p, variant)
                                    for a, b in zip(graphs, banks)]
                        encoding_seconds = time.perf_counter() - tick
                        for decimals in args.decimals:
                            sigs = [signature(f, decimals) for f in features]
                            records.append(dict(
                                pair_id=pid, category=category(pid), basis=basis,
                                q=q, p=p, variant=variant, decimals=decimals,
                                separated=sigs[0] != sigs[1],
                                reliability_representative_failed=sigs[2] != sigs[3],
                                channels=features[0].shape[-1],
                                feature_bytes=sum(f.nbytes for f in features),
                                encoding_seconds=encoding_seconds,
                                max_q_bank_seconds=bank_seconds))
        print(f"screened pair {pid}", flush=True)
    digest = hashlib.sha256()
    with args.brec_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    payload = dict(experiment="brec_pair_encoding_screen", records=records,
                   configuration={k: str(v) if isinstance(v, Path) else v
                                  for k, v in vars(args).items()},
                   dataset_sha256=digest.hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   numpy_version=np.__version__, networkx_version=nx.__version__,
                   elapsed_seconds=time.perf_counter()-started,
                   note="Rounded pair-multiset screening; no training, no exact "
                        "arithmetic claim, and no full RPC reliability claim.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
