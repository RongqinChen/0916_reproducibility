"""Bounded-memory pair/triple features and cumulative moment readouts."""
from collections import Counter
import hashlib
import numpy as np
try:
    from .budget_sweep import polynomial_bank
except ImportError:  # standalone module inside a frozen BREC snapshot
    from budget_sweep import polynomial_bank


def signatures_equal(left, right, readout, tolerance):
    """Compare two signatures, allowing an absolute tolerance for sum/mean.

    Byte equality is tried first. Multiset signatures stay exact: their rows
    are already stable digests, and a tolerant multiset match would need an
    assignment problem rather than an equality test.
    """
    if left == right:
        return True
    if tolerance <= 0 or readout == "multiset":
        return False
    a = np.frombuffer(left, dtype="<f4")
    b = np.frombuffer(right, dtype="<f4")
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=0.0, atol=tolerance))


def channel_count(order, q, variant):
    if order not in (2, 3) or variant not in ("polynomial", "endpoint", "joint"):
        raise ValueError("Invalid encoding")
    if order == 2 and variant == "joint":
        raise ValueError("Joint is a triple-only condition")
    pair = q + 3
    # Strict total-degree budget: a,b >= 0 and a+b < q, excluding B_00=J.
    endpoint = q * (q + 1) // 2 - 1
    if order == 2:
        return pair + (endpoint if variant == "endpoint" else 0)
    return 1 + 6*(pair - 1) + (6*endpoint if variant != "polynomial" else 0) + (
        (q+1)**3 if variant == "joint" else 0)


def feature_chunks(adj, bank, q, order, variant, chunk_size=256, joint_normalise=None,
                   columns=None):
    """Ordered tuples, all six ordered coordinate lifts, one constant.

    Joint channels are evaluated by batched contractions without constructing
    a pivot-by-all-tuples array. Channel indices are lexicographic.

    The pair/lift gather is issued as one advanced-index call per matrix and
    the endpoint block as one broadcast product per ordered position; values
    and column order are unchanged, only the number of numpy calls drops.

    ``joint_normalise="n"`` divides every K_abc channel by the vertex count.
    That is a fixed positive rescaling of one channel block: it is a bijection
    on the feature space (no separation power is lost in exact arithmetic) and
    it keeps the joint rows at O(1) so post-aggregation rounding stays stable.
    """
    width = channel_count(order, q, variant)
    selected=np.arange(width) if columns is None else np.asarray(columns,dtype=int)
    if np.any(selected<0) or np.any(selected>=width) or len(np.unique(selected))!=len(selected):
        raise ValueError('Invalid channel selection')
    lookup=np.full(width,-1,dtype=int)
    lookup[selected]=np.arange(len(selected))
    n = len(adj)
    h = bank[:q+1].sum(2)
    pair_matrices = [np.eye(n, dtype=np.float32), adj, *bank[1:q+1]]
    positions = [(0,1)] if order == 2 else [
        (i,j) for i in range(3) for j in range(3) if i != j]
    endpoint_degrees = [(a,b) for a in range(q) for b in range(q-a)
                        if (a,b) != (0,0)]
    endpoint_width = len(endpoint_degrees)
    for start in range(0, n**order, chunk_size):
        ids = np.arange(start, min(start+chunk_size, n**order))
        coordinates = np.unravel_index(ids, (n,)*order)
        result = np.empty((len(ids), len(selected)), dtype=np.float32)
        if lookup[0]>=0:
            result[:, lookup[0]] = 1.
        offset = 1
        if order == 2:
            u, v = coordinates
            for matrix in pair_matrices:
                if lookup[offset]>=0:
                    result[:,lookup[offset]] = matrix[u,v]
                offset += 1
        else:
            left = np.stack([coordinates[i] for i,_ in positions])
            right = np.stack([coordinates[j] for _,j in positions])
            span = len(positions)
            for matrix in pair_matrices:
                take=lookup[offset:offset+span]>=0
                if take.any():
                    result[:,lookup[offset:offset+span][take]] = matrix[left[take],right[take]].T
                offset += span
        if variant != "polynomial":
            if order == 2:
                u, v = coordinates
                for a,b in endpoint_degrees:
                    if lookup[offset]>=0:
                        result[:,lookup[offset]] = h[a,u]*h[b,v]
                    offset += 1
            else:
                u, v, w_ = coordinates
                gathered = [h.T[u], h.T[v], h.T[w_]]
                for i, j in positions:
                    take=np.flatnonzero(lookup[offset:offset+endpoint_width]>=0)
                    if len(take):
                        degrees=np.asarray([endpoint_degrees[index] for index in take])
                        a,b=degrees[:,0],degrees[:,1]
                        result[:,lookup[offset+take]]=gathered[i][:,a]*gathered[j][:,b]
                    offset += endpoint_width
        if variant == "joint":
            scale = float(n) if joint_normalise == "n" else 1.0
            # Per (a,b), contract all c channels together; temporary O(chunk*n).
            u,v,w = coordinates
            for a in range(q+1):
                for b in range(q+1):
                    if not np.any(lookup[offset:offset+q+1]>=0):
                        offset+=q+1
                        continue
                    product = bank[a,u,:] * bank[b,v,:]
                    for c in range(q+1):
                        if lookup[offset]>=0:
                            result[:,lookup[offset]] = np.einsum("it,it->i", product, bank[c,w,:]) / scale
                        offset += 1
        assert offset == width
        yield result


def fingerprints(adj, bank, q, order, variant, decimals, chunk_size):
    """Multiset of 256-bit row digests after rounding.

    Cryptographic hashing bounds memory, but is not a mathematical exactness
    claim. Report numerical stability checks alongside all decisions.
    """
    result = readout_signatures(adj, bank, q, order, variant, decimals,
                                chunk_size, ("multiset",))
    return {d: result[("multiset", d, 1)] for d in decimals}


def readout_signatures(adj, bank, q, order, variant, decimals, chunk_size,
                       readouts=("mean", "multiset"), p_values=(1,),
                       joint_normalise=None):
    """Single-variant signatures; see readout_signatures_shared."""
    return readout_signatures_shared(adj, bank, q, order, decimals, chunk_size,
                                     readouts, p_values, (variant,),
                                     joint_normalise)[variant]


def readout_signatures_shared(adj, bank, q, order, decimals, chunk_size,
                              readouts=("mean", "multiset"), p_values=(1,),
                              variants=("polynomial", "endpoint"),
                              joint_normalise=None):
    """Compute both readouts for several prefix-related variants in one pass.

    ``polynomial`` channels are a prefix of ``endpoint``, which is a prefix of
    ``joint``, so a single streamed pass over the widest requested variant
    serves every variant by slicing the same rows.  Values, column order and
    rounding are unchanged; only the duplicated feature generation is removed.

    Mean uses float32 compensated chunk accumulation, divides by tuple count,
    then rounds the final graph vector, NOT the tuple vectors. No learned map.
    Explicit sum remains supported for legacy callers only.
    """
    if not set(readouts) <= {"multiset", "sum", "mean"}:
        raise ValueError("Unsupported readout")
    if not p_values or min(p_values)<1:
        raise ValueError("P must be positive")
    variants = tuple(variants)
    if not variants:
        raise ValueError("At least one variant is required")
    accumulate = ("sum" in readouts) or ("mean" in readouts)
    widths = {v: channel_count(order, q, v) for v in variants}
    widest = max(variants, key=lambda v: widths[v])
    counters = {v: ({d: Counter() for d in decimals} if "multiset" in readouts else {})
                for v in variants}
    totals = {v: np.zeros((max(p_values), widths[v]), dtype=np.float32) for v in variants}
    corrections = {v: np.zeros_like(totals[v]) for v in variants}
    # Mean readout divides the ordered-tuple sum by the tuple count, which keeps
    # the magnitude O(1) and therefore keeps post-aggregation rounding stable.
    normaliser = float(len(adj) ** order)
    for rows in feature_chunks(adj,bank,q,order,widest,chunk_size,joint_normalise):
        if not np.isfinite(rows).all():
            raise FloatingPointError("Nonfinite encoding")
        for variant in variants:
            sub = rows if widths[variant] == rows.shape[1] else rows[:, :widths[variant]]
            if accumulate:
                powered=sub.copy()
                with np.errstate(over="raise",invalid="raise"):
                    for exponent in range(max(p_values)):
                        if exponent:
                            powered *= sub
                        increment=(powered.sum(axis=0,dtype=np.float32)
                                   -corrections[variant][exponent])
                        updated=totals[variant][exponent]+increment
                        corrections[variant][exponent]=(updated-totals[variant][exponent])-increment
                        totals[variant][exponent]=updated
            for d in counters[variant]:
                rounded = np.round(sub,d)
                rounded[rounded == 0] = 0.
                for row in rounded:
                    counters[variant][d][hashlib.sha256(
                        row.astype("<f4",copy=False).tobytes()).digest()] += 1
    results = {}
    for variant in variants:
        result = {("multiset", d, 1): value for d, value in counters[variant].items()}
        if accumulate:
            if not np.isfinite(totals[variant]).all():
                raise FloatingPointError("Nonfinite aggregated readout")
            for p in p_values:
                for d in decimals:
                    if "sum" in readouts:
                        rounded=np.round(totals[variant][:p],d)
                        rounded[rounded==0]=0.
                        result[("sum",d,p)]=rounded.astype("<f4",copy=False).tobytes()
                    if "mean" in readouts:
                        rounded=np.round(totals[variant][:p]/normaliser,d)
                        rounded[rounded==0]=0.
                        result[("mean",d,p)]=rounded.astype("<f4",copy=False).tobytes()
        results[variant] = result
    return results


def exact_fwl_pair(left, right, k, max_rounds=100, max_work=2_000_000):
    """Synchronized exact integer colors; joint coordinate replacement FWL.

    Returns True for a separation, False ONLY after stable equality, and None
    for resource/round limits. Intended for small graphs, not unrestricted 3-FWL.
    """
    if len(left) != len(right):
        return True
    n = len(left)
    if n**(k+1) > max_work:
        return None
    tuples = list(np.ndindex((n,)*k))
    def intern(profiles):
        vocabulary = {}
        return [np.array([vocabulary.setdefault(p,len(vocabulary)) for p in side])
                for side in profiles]
    profiles = []
    for adj in (left,right):
        profiles.append([tuple(x for i in range(k) for j in range(i+1,k)
                               for x in (int(t[i] == t[j]),int(adj[t[i],t[j]])))
                         for t in tuples])
    colors = intern(profiles)
    for step in range(max_rounds+1):
        if Counter(colors[0]) != Counter(colors[1]):
            return True
        if step == max_rounds:
            return None
        old_classes = len(set(colors[0]) | set(colors[1]))
        profiles = []
        for color in colors:
            grid = color.reshape((n,)*k)
            side = []
            for index,t in enumerate(tuples):
                joint = []
                for pivot in range(n):
                    joint.append(tuple(int(grid[t[:i]+(pivot,)+t[i+1:]])
                                       for i in range(k)))
                side.append((int(color[index]),tuple(sorted(joint))))
            profiles.append(side)
        colors = intern(profiles)
        if len(set(colors[0]) | set(colors[1])) == old_classes:
            return False
