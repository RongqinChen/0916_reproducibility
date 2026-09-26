"""CPU endpoint-summary products of symmetric normalized adjacency powers.

The legacy function name is retained for configuration compatibility.
These are endpoint products, not shared-pivot joint counts.
"""
import torch


def endpoint_degree_pairs(qsteps, add_identity=True, channel_budget=0,
                          exclude_constant=False, symmetric_channels=False):
    """Ordered degrees with strict total-degree budget a+b < qsteps."""
    first = 0 if add_identity else 1
    pairs = [(a, total-a) for total in range(2*first, qsteps)
             for a in range(first, total-first+1)
             if not symmetric_channels or a <= total-a]
    if exclude_constant:
        pairs = [pair for pair in pairs if pair != (0, 0)]
    if channel_budget:
        if channel_budget < 1 or channel_budget > len(pairs):
            raise ValueError('Endpoint channel_budget exceeds the available dictionary')
        pairs = pairs[:channel_budget]
    if not pairs:
        raise ValueError('Endpoint budget has no channels: require Q>=3 (or Q>=1 with identity)')
    return pairs


@torch.no_grad()
def add_single_pivot_contraction(data, qsteps=8, add_identity=True,
                                 node_attr_name="pestat_ContractionNode",
                                 edge_attr_name="pestat_ContractionEdge",
                                 same_degree=False, injection="both",
                                 channel_budget=0, exclude_constant=False,
                                 symmetric_channels=False):
    if qsteps < 1:
        raise ValueError("qsteps must be positive")
    n = int(data.num_nodes)
    index = data.edge_index.detach().cpu()
    src, dst = index
    edge_weight = getattr(data, "edge_weight", None)
    weight = (edge_weight.detach().cpu().to(torch.float32) if edge_weight is not None
              else torch.ones(src.numel(), dtype=torch.float32))
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Expected finite nonnegative edge weights")
    degree = torch.zeros(n, dtype=weight.dtype).index_add_(0, src, weight)
    inv = torch.zeros_like(degree)
    mask = degree > 0
    inv[mask] = degree[mask].rsqrt()
    normalized = torch.sparse_coo_tensor(
        index, weight * inv[src] * inv[dst], (n, n)).coalesce()
    reverse = normalized.transpose(0, 1).coalesce()
    if not (torch.equal(normalized.indices(), reverse.indices())
            and torch.allclose(normalized.values(), reverse.values())):
        raise ValueError("Endpoint encoding requires an undirected weighted graph")
    pairs = endpoint_degree_pairs(qsteps, add_identity, channel_budget,
                                  exclude_constant, symmetric_channels)
    if same_degree:
        pairs = [(a, b) for a, b in pairs if a == b]
    vectors = [torch.ones(n, dtype=weight.dtype)]
    for _ in range(max(max(pair) for pair in pairs)):
        vectors.append(torch.sparse.mm(normalized, vectors[-1][:, None]).squeeze(1))
    h = torch.stack(vectors)

    if injection not in ("node","edge","both"):
        raise ValueError("Expected node or edge injection")
    a, b = torch.tensor(pairs, dtype=torch.long).T
    if injection in ("node","both"):
        data[node_attr_name] = (h[a] * h[b]).T.contiguous()
    if injection in ("edge","both"):
        data[edge_attr_name] = (h[a][:, src] * h[b][:, dst]).T.contiguous()
    return data
