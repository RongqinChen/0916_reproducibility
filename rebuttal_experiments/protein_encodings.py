"""Encoding-only replacements for Co-Sparsify's native Polynomials transform."""
from __future__ import annotations

ENCODINGS = ('rrwp', 'mono', 'cheb', 'none', 'base', 'endpoint', 'powers', 'joint', 'joint_mix')
COMPARISONS = (('mono', 'rrwp'), ('cheb', 'mono'), ('base', 'none'),
               ('endpoint', 'base'), ('powers', 'endpoint'), ('joint', 'endpoint'),
               ('joint', 'powers'), ('joint', 'mono'), ('joint_mix', 'joint'))


def layout(width):
    base = width // 2
    endpoint = (width - base) // 2
    return base, endpoint, width - base - endpoint


def joint_pair_channels(poly, count):
    """J_q(u,v)=A_uv sum_w A_uw A_vw sum_t Bq_ut Bq_vt Bq_wt.

    This fixed readout uses a three-coordinate shared-pivot channel without
    adding a trainable readout or changing the upstream pair-based backbone.
    Only one common-neighbor plane is held at a time; no n^3 cache is stored.
    """
    import torch
    adjacency = (poly[:, :, 1] > 0).to(poly.dtype)
    result = poly.new_zeros((len(poly), len(poly), count))
    for degree in range(1, count + 1):
        matrix = poly[:, :, degree]
        for u in range(len(poly)):
            neighbors = adjacency[u].nonzero(as_tuple=False).flatten()
            if not len(neighbors):
                continue
            rows = matrix[neighbors]
            joint = (rows * matrix[u]) @ rows.T
            triangle = adjacency[neighbors[:, None], neighbors[None, :]]
            result[u, neighbors, degree - 1] = (joint * triangle).sum(1)
    return result


def make_transform(upstream, poly_method, poly_dim, **kwargs):
    """Keep the native transform, including metadata, for upstream baselines."""
    if poly_method not in ENCODINGS:
        raise ValueError(poly_method)
    if poly_method in ('rrwp', 'mono', 'cheb'):
        return upstream(poly_method=poly_method, poly_dim=poly_dim, **kwargs)

    class EncodingTransform(upstream):
        def __init__(self):
            super().__init__(poly_method='mono', poly_dim=poly_dim, **kwargs)

        def forward(self, data):
            import math
            import torch
            data = super().forward(data)
            poly = data['pair_x'].reshape(data.num_nodes, data.num_nodes, poly_dim)
            values = torch.zeros_like(poly)
            base, endpoints, extra = layout(poly_dim)
            if poly_method != 'none':
                values[:, :, :base] = poly[:, :, :base]
            if poly_method in ('endpoint', 'powers', 'joint', 'joint_mix'):
                h1, h2 = poly[:, :, 1].sum(1), poly[:, :, 2].sum(1)
                channels = (h1[:, None], h1[None, :], h2[:, None], h2[None, :])
                for column in range(endpoints):
                    values[:, :, base + column] = channels[column]
            if poly_method == 'powers':
                values[:, :, base + endpoints:] = poly[:, :, 1:extra+1].square()
            if poly_method in ('joint', 'joint_mix'):
                values[:, :, base + endpoints:] = joint_pair_channels(poly, extra)
            if poly_method == 'joint_mix':
                indices = torch.arange(poly_dim, dtype=poly.dtype)
                change = torch.cos(math.pi / poly_dim * indices[:, None] * (indices[None, :] + 0.5))
                change[0] *= math.sqrt(1 / poly_dim)
                change[1:] *= math.sqrt(2 / poly_dim)
                values = values @ change.T
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite structural encoding')
            data['pair_x'] = values.flatten(0, 1)
            data['loop_x'] = torch.diagonal(values).permute(1, 0)
            return data

        def __repr__(self):
            return f'{poly_method}_{poly_dim}'

    return EncodingTransform()
