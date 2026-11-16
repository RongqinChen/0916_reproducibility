"""Floating-point diagnostics. None of these decisions is an exact certificate."""
from collections import Counter

import numpy as np

from .suite_data import graphs
from .suite_exact import descriptors


def bank(graph, q, basis, dtype):
    a = graph.astype(dtype)
    degree = a.sum(1)
    inv = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=inv, where=degree > 0)
    m = inv[:, None]*a*inv[None, :]
    values = [np.eye(len(a), dtype=dtype), m]
    for _ in range(2, q+1):
        product = m @ values[-1]
        values.append(product if basis == "monomial" else 2*product-values[-2])
    return np.stack(values)


def column(graph, values, key, order, chunk=512):
    n = len(graph)
    output = np.empty(n**order, dtype=values.dtype)
    h = values.sum(2)
    for start in range(0, len(output), chunk):
        end = min(start+chunk, len(output))
        row = np.unravel_index(np.arange(start, end), (n,)*order)
        if key[0] == "constant":
            result = np.ones(end-start, dtype=values.dtype)
        elif key[0] == "pair":
            _, name, i, j = key
            matrix = values[0] if name == "I" else graph if name == "A" else values[name]
            result = matrix[row[i], row[j]]
        elif key[0] == "endpoint":
            _, a, b, i, j = key
            result = h[a, row[i]]*h[b, row[j]]
        else:
            a, b, c = key[1:]
            result = (values[a, row[0], :]*values[b, row[1], :]*values[c, row[2], :]).sum(1)
        output[start:end] = result
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite diagnostic feature column")
    return output


def diagnostic_compare(graph_pair, q, order, variant, basis, dtype, scales, decimals, atols, powers):
    kernels = [bank(g, q, basis, dtype) for g in graph_pair]
    keys = descriptors(q, order, variant)
    n = len(graph_pair[0])
    raw_states = {(scale, digits): [np.zeros(n**order, dtype=np.int64) for _ in range(2)]
                  for scale in scales for digits in decimals}
    raw_different = {key: False for key in raw_states}
    mean_different = {(scale, p, atol, rounded): False for scale in scales for p in powers
                      for atol in atols for rounded in (None, 8)}
    # Full tuple rows are evaluated here: orbit compression would hide rounding-dependent differences.
    for key in keys:
        originals = [column(g, k, key, order) for g, k in zip(graph_pair, kernels)]
        for scale in scales:
            divided = scale in ("K_over_n", "restored") and key[0] == "joint"
            values = [v/n if divided else v for v in originals]
            for digits in decimals:
                state_key = scale, digits
                if raw_different[state_key]:
                    continue
                # Restore before raw comparison, but after arithmetic in the scaled units.
                raw_values = [v*n if scale == "restored" and divided else v for v in values]
                raw_values = [v if digits is None else np.round(v, digits) for v in raw_values]
                vocab, new = {}, []
                for old, vals in zip(raw_states[state_key], raw_values):
                    new.append(np.fromiter((vocab.setdefault((int(c), float(v)), len(vocab))
                                            for c, v in zip(old, vals)), dtype=np.int64, count=len(vals)))
                raw_states[state_key] = new
                if Counter(new[0]) != Counter(new[1]):
                    raw_different[state_key] = True
                    raw_states[state_key] = None
            for power in range(1, max(powers)+1):
                means = [np.mean(v**power, dtype=np.dtype(dtype)) for v in values]
                if scale == "restored" and divided:
                    means = [m*n**power for m in means]
                if not np.isfinite(means).all():
                    raise FloatingPointError("Nonfinite diagnostic moment")
                for rounding in (None, 8):
                    a, b = means if rounding is None else np.round(means, rounding)
                    for atol in atols:
                        if abs(float(a)-float(b)) > atol:
                            for limit in powers:
                                if limit >= power:
                                    mean_different[scale, limit, atol, rounding] = True
    raw = [{"scale": scale, "decimals": digits, "different": value}
           for (scale, digits), value in raw_different.items()]
    means = [{"scale": scale, "p": p, "atol": atol, "rtol": 0,
              "moment_decimals": rounded, "different": value}
             for (scale, p, atol, rounded), value in mean_different.items()]
    return {"basis": basis, "dtype": dtype, "q": q, "order": order, "variant": variant,
            "multiset": raw, "moments": means, "arithmetic_scope": "floating_point_diagnostic"}


def run_diagnostic(data, job, config):
    main = graphs(data, job["pair_id"])
    reliability = graphs(data, job["pair_id"], True)
    rng = np.random.default_rng(config["seed"]+job["pair_id"])
    samples = [("main", main), ("official_reliability", reliability)]
    # Compare each graph against its own independently recomputed relabeling.
    for index in range(config["diagnostic_relabels"]):
        for side, graph in enumerate(main):
            p = rng.permutation(len(graph))
            samples.append((f"self_relabel_{side}_{index}", [graph, graph[np.ix_(p, p)]]))
        perms = [rng.permutation(len(g)) for g in main]
        samples.append((f"cross_relabel_{index}", [g[np.ix_(p, p)] for g, p in zip(main, perms)]))
    results = []
    for label, pair in samples:
        result = diagnostic_compare(pair, job["q"], job["order"], job["variant"], job["basis"],
                                    job["dtype"], ["K", "K_over_n", "restored"] if job["variant"] == "joint" else ["K"],
                                    config["decimals"], config["atols"], config["powers"])
        results.append({"sample": label, **result})
    return {"diagnostics": results, "scope": "diagnostic_only; retains every observed discrepancy"}
