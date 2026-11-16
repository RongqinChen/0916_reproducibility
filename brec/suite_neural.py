"""Per-pair fitting, exact input baselines, saved embeddings, and independent scores."""
from functools import lru_cache
import random

import numpy as np

from .suite_data import graphs, isomorphism, tuple_orbits
from .suite_exact import Kernel, compare


def rpc_score(embeddings):
    """Float64 covariance/pseudoinverse and centered-SVD with a shared cutoff."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.shape != (64, 16) or not np.isfinite(x).all():
        raise ValueError("RPC needs 64 finite 16-dimensional embeddings")
    delta = x[::2]-x[1::2]
    mean = delta.mean(0)
    centered = delta-mean
    covariance = centered.T@centered/31
    eigenvalues, vectors = np.linalg.eigh(covariance)
    cutoff = 16*np.finfo(np.float32).eps*max(0., float(eigenvalues[-1]))
    keep = eigenvalues > cutoff
    inverse = (vectors[:, keep]/eigenvalues[keep])@vectors[:, keep].T
    primary = float(mean@inverse@mean)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    keep_svd = singular**2/31 > cutoff
    second = float(31*np.sum((vh[keep_svd]@mean/singular[keep_svd])**2))
    agreement = bool(np.isclose(primary, second, rtol=1e-3, atol=1e-5))
    degenerate = not keep.any() and np.linalg.norm(mean) > 1e-7
    valid = agreement and not degenerate and np.isfinite([primary, second]).all() and primary >= 0
    return {"score": primary, "svd_score": second, "rank": int(keep.sum()),
            "cutoff": cutoff, "mean_norm": float(np.linalg.norm(mean)),
            "scorers_agree": agreement, "degenerate_nonzero_mean": bool(degenerate),
            "valid": bool(valid), "scoring_dtype": "float64", "cutoff_epsilon_dtype": "float32"}


def exact_input_bank(graph, q):
    kernel = Kernel(graph, q)
    # Convert each canonical exact value by the same operation sequence.
    # This avoids label-order-dependent matrix accumulation in neural inputs.
    convert = lru_cache(maxsize=None)(float)
    return np.asarray([[[convert(kernel.pair(power, u, v)) for v in range(len(graph))]
                        for u in range(len(graph))] for power in range(q+1)], dtype=np.float32)


def input_tensor(values, order):
    import torch
    if order == 2:
        return torch.from_numpy(values.copy())
    n, c = values.shape[1], values.shape[0]
    # Three plane blocks (u,v), (u,w), (v,w), each containing powers 0..Q.
    planes = [np.broadcast_to(values[:, :, :, None], (c, n, n, n)),
              np.broadcast_to(values[:, :, None, :], (c, n, n, n)),
              np.broadcast_to(values[:, None, :, :], (c, n, n, n))]
    return torch.from_numpy(np.concatenate(planes, axis=0))


def aggregate_consistency(groups, atol, rtol):
    """All pairwise graph embeddings in each isomorphism group must be close.

    Symmetric tolerance: abs(a-b) <= atol + rtol*max(abs(a),abs(b)).
    A group is never compared to the other member of the nonisomorphic pair.
    """
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Finite nonnegative aggregate tolerances are required")
    reports = []
    for label, values in groups.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Invalid graph embeddings in an isomorphism group")
        passed, max_error, excess, comparisons = True, 0., 0., 0
        for i in range(len(values)):
            for j in range(i):
                difference = np.abs(values[i]-values[j])
                tolerance = atol+rtol*np.maximum(np.abs(values[i]), np.abs(values[j]))
                passed = passed and bool(np.all(difference <= tolerance))
                max_error = max(max_error, float(difference.max()))
                excess = max(excess, float((difference-tolerance).max()))
                comparisons += 1
        reports.append({"group": label, "embeddings": len(values), "pairwise_comparisons": comparisons,
                        "passed": passed, "max_absolute_error": max_error, "max_tolerance_excess": excess})
    return {"passed": all(r["passed"] for r in reports), "atol": atol, "rtol": rtol,
            "comparison": "symmetric_elementwise_tolerance", "groups": reports,
            "scope": "tested_graph_embeddings; numerical_consistency_not_exact_equality"}


def extra_relabel_embeddings(main, reliability, config, pair_id, embed, q=8):
    """Fresh integer relabelings and independently reconstructed exact input banks."""
    rng = np.random.default_rng(config["seed"]+pair_id*1009+701)
    outputs, permutations = {}, {}
    for label, graph, count in [(f"main_{s}", main[s], config["relabels"]) for s in (0, 1)]+[
            (f"reliability_{s}", reliability[s], config["reliability_relabels"]) for s in (0, 1)]:
        perms = [rng.permutation(len(graph)) for _ in range(count)]
        # Construct one bank at a time; repeated input tensors are not cached.
        banks = [exact_input_bank(graph[np.ix_(p, p)], q) for p in perms]
        outputs[label] = embed(banks) if banks else np.empty((0, 16), dtype=np.float32)
        permutations[label] = [p.tolist() for p in perms]
    return outputs, permutations


def run_neural(data, job, config, artifacts):
    import torch
    from .suite_model import PPGN
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    seed, order, pid = job["seed"], job["order"], job["pair_id"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    main = graphs(data, pid, count=64)
    reliability = graphs(data, pid, True, count=64)
    n, q = len(main[0]), 8
    # Validate official inputs before using mapped exact banks as a cache.
    representatives = [main[0], main[1], reliability[0]]
    banks = [exact_input_bank(g, q) for g in representatives]
    main_banks, reliability_banks = [], []
    for i, graph in enumerate(main):
        inv = np.argsort(isomorphism(representatives[i % 2], graph))
        main_banks.append(banks[i % 2][:, inv][:, :, inv])
    for graph in reliability:
        inv = np.argsort(isomorphism(representatives[2], graph))
        reliability_banks.append(banks[2][:, inv][:, :, inv])
    # Check the cache transformation against independent exact reconstruction.
    for index in (2, 3):
        if not np.array_equal(main_banks[index], exact_input_bank(main[index], q)):
            raise AssertionError("Neural input bank does not align under relabeling")
    positions = [(0, 1)] if order == 2 else [(0, 1), (0, 2), (1, 2)]
    keys = [("pair", "I" if p == 0 else p, i, j) for i, j in positions for p in range(q+1)]
    baseline = compare([Kernel(g, q) for g in main[:2]], [tuple_orbits(g, order) for g in main[:2]],
                       q, order, "neural_input", [1], keys_override=keys)
    # Select a fixed physical batch before initialization/training; do not alter it mid-run.
    per_graph = n**order*max(27 if order == 3 else 9, 16)*4
    batch = job["physical_batch"]
    if per_graph*batch*20 >= job["memory_gib"]*2**30*.7:
        raise MemoryError("The fixed physical batch exceeds this job's memory reservation")
    model = PPGN(order)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0)
    accumulation = 32//batch
    history = []

    def batch_tensor(values, start):
        return torch.stack([input_tensor(v, order) for v in values[start:start+batch]])

    for epoch in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for step, start in enumerate(range(0, 64, batch), 1):
            y = model(batch_tensor(main_banks, start))
            loss = loss_fn(y[::2], y[1::2], -torch.ones(len(y)//2))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite neural training loss")
            (loss/accumulation).backward()
            total += float(loss.detach())*len(y)/2
            if step % accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        average = total/32
        scheduler.step(average)
        history.append({"epoch": epoch, "loss": average})
        if average < .2:
            break

    @torch.no_grad()
    def embed(values):
        model.eval()
        return np.concatenate([model(batch_tensor(values, start)).numpy() for start in range(0, 64, batch)])

    embeddings, controls = embed(main_banks), embed(reliability_banks)
    repeated = embed(main_banks)
    if not np.array_equal(embeddings, repeated):
        raise AssertionError("Repeated neural evaluation changed embeddings")
    extra, permutations = extra_relabel_embeddings(main, reliability, config, pid, embed, q)
    consistency = aggregate_consistency({
        "main_left": np.concatenate([embeddings[::2], extra["main_0"]]),
        "main_right": np.concatenate([embeddings[1::2], extra["main_1"]]),
        "official_reliability": np.concatenate([controls, extra["reliability_0"], extra["reliability_1"]])},
        config["neural_invariance_atol"], config["neural_invariance_rtol"])
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifacts/"embeddings.npz", main=embeddings, reliability=controls,
                        **{f"extra_{name}": value for name, value in extra.items()})
    torch.save(model.state_dict(), artifacts/"model.pt")
    primary, control = rpc_score(embeddings), rpc_score(controls)
    valid = primary["valid"] and control["valid"] and control["score"] < 72.34 and consistency["passed"]
    separated = primary["score"] > 72.34 and not np.isclose(primary["score"], control["score"], rtol=1e-5, atol=1e-6)
    return {"input_baseline": baseline, "main_score": primary, "reliability_score": control,
            "numerical_decision": "unresolved" if not valid else "different" if separated else "not_detected",
            "raw_separated": bool(separated), "history": history, "physical_batch": batch,
            "aggregate_relabel_consistency": consistency, "extra_new_to_old": permutations,
            "effective_batch": 32, "parameters": sum(p.numel() for p in model.parameters()),
            "input_arithmetic": "canonical_exact_values_converted_to_float32",
            "model_dtype": "float32", "device": "cpu", "repeat_embeddings_equal": True,
            "scope": "per_pair_fitting; not held_out_prediction; not an exact certificate"}
