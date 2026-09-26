"""PPGN pair/triple backbone following the CSDGSE BREC reference implementation.

Reference (kept in sync by hand, see README section 2):
    /home/chenrq/CSDGSE/run_brec.py
    /home/chenrq/CSDGSE/configs/brec.ppgn.poly.yaml
    /home/chenrq/CSDGSE/models/network/ppgn_brec.py
    /home/chenrq/CSDGSE/models/layer/ppgn_layers.py

Components, matched to the reference:

* input: rrwp powers ``[I, A, A^2, ...]`` of the symmetric-normalised adjacency
  (``poly_method: rrwp`` is the reference default), one channel per power for a
  pair and three lifted planes ``(u,v), (u,w), (v,w)`` per power for a triple;
* block: parallel 1x1 channel convolutions -> pivot contraction -> signed root
  ``rho_{1/k}`` -> concat skip projection (the reference ``RegularBlock``), with
  the reference's signed square root at ``k = 2``;
* normalisation: ``BatchNorm2d``/``BatchNorm3d`` after every block;
* readout: diagonal/off-diagonal pooling (pair) or diagonal/planes/marginal
  pooling (triple) -> ``Linear`` per block -> summed over blocks -> node mean ->
  ``Linear`` classifier (the reference head);
* capacity: width 32, five blocks, one 1x1 convolution per branch
  (``mlp_depth: 1`` in the reference config).

The realisation runner's resource ladder (physical batch 32 -> 16 -> 8 -> 4)
and its input caches are deliberately independent of this architecture.
"""
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from torch import nn

from .budget_sweep import polynomial_bank


class _Contracted3(torch.autograd.Function):
    """Memory-safe three-branch pivot contraction for k=3 J1.

    Forward and backward both loop over the pivot t and never keep one
    term per pivot alive: activations stay O(N^3 * P * batch) instead of the
    O(N^4 * P * batch) retention of the naive ``raw = raw + term`` chain.
    """

    @staticmethod
    def forward(ctx, b1, b2, b3):
        b, n, _, _, p = b1.shape
        raw1 = torch.zeros((b, n, n, n, p), device=b1.device, dtype=b1.dtype)
        for t in range(n):
            term = (
                b1[:, t].unsqueeze(1)
                * b2[:, :, t].unsqueeze(2)
                * b3[:, :, :, t].unsqueeze(3)
            )
            raw1 = raw1 + term
        ctx.save_for_backward(b1, b2, b3)
        return raw1

    @staticmethod
    def backward(ctx, grad_raw):
        b1, b2, b3 = ctx.saved_tensors
        b, n, _, _, p = b1.shape
        g1 = torch.zeros_like(b1)
        g2 = torch.zeros_like(b2)
        g3 = torch.zeros_like(b3)
        if not (b1.requires_grad or b2.requires_grad or b3.requires_grad):
            return g1, g2, g3
        for t in range(n):
            b1t = b1[:, t]  # [B, v, w, P]
            b2t = b2[:, :, t]  # [B, u, w, P]
            b3t = b3[:, :, :, t]  # [B, u, v, P]
            # g1[t,v,w] = sum_u grad[u,v,w] b2[u,t,w] b3[u,v,t]
            m3 = b2t.unsqueeze(2) * b3t.unsqueeze(3)  # [B, u, v, w, P]
            g1[:, t] += (grad_raw * m3).sum(dim=1)
            # g2[u,t,w] = sum_v grad[u,v,w] b1[t,v,w] b3[u,v,t]
            m2 = b1t.unsqueeze(1) * b3t.unsqueeze(3)  # [B, u, v, w, P]
            g2[:, :, t] += (grad_raw * m2).sum(dim=2)
            # g3[u,v,t] = sum_w grad[u,v,w] b1[t,v,w] b2[u,t,w]
            m1 = b1t.unsqueeze(1) * b2t.unsqueeze(2)  # [B, u, v, w, P]
            g3[:, :, :, t] += (grad_raw * m1).sum(dim=3)
        return g1, g2, g3


def signed_root(x, alpha, eps=0.0):
    """``rho_alpha``: the reference signed root applied after every contraction.

    ``alpha = 1/2`` with ``eps = 0`` is exactly the reference pair block's
    ``sqrt(relu(x)) - sqrt(relu(-x))``.
    """
    if eps <= 0:
        return torch.sign(x) * x.abs().pow(alpha)
    return torch.sign(x) * ((x.abs() + eps) ** alpha - eps ** alpha)


def input_channels(order, q):
    """Channels of the rrwp input: ``q + 1`` powers, lifted per tuple order."""
    if order not in (2, 3):
        raise ValueError('Supported tuple orders are 2 and 3')
    return (q + 1) if order == 2 else 3 * (q + 1)


class FixedInput:
    """CSDGSE rrwp input: powers of the symmetric-normalised adjacency.

    ``order == 2`` gives ``[q+1]`` channels per pair; ``order == 3`` stacks the
    three lifted planes ``(u,v), (u,w), (v,w)``, i.e. ``3(q+1)`` channels per
    triple. Feature arrays and stacked batches are cached because the training
    loop revisits the same graph list for every epoch and every seed.
    """

    def __init__(self, q=8, chunk_size=128, cache_bytes=4*1024**3,
                 tensor_cache_bytes=6*1024**3):
        self.q, self.chunk_size = q, chunk_size
        self.banks = OrderedDict()
        self.cache_bytes = int(cache_bytes)
        self.arrays = OrderedDict()
        self.cached_bytes = 0
        self.tensor_cache_bytes = int(tensor_cache_bytes)
        self.tensors = OrderedDict()
        self.tensor_bytes = 0

    def bank(self, adj):
        adj = np.asarray(adj, dtype=np.float32)
        key = hashlib.sha256(adj.tobytes()).digest()
        stored = self.banks.get(key)
        if stored is None:
            stored = polynomial_bank(adj, self.q, 'monomial')
            self.banks[key] = stored
            if len(self.banks) > 128:
                self.banks.popitem(last=False)
        return key, stored

    def array(self, adj, order):
        key, bank = self.bank(adj)
        cache_key = (key, order)
        cached = self.arrays.get(cache_key)
        if cached is not None:
            self.arrays.move_to_end(cache_key)
            return cached
        powers = np.moveaxis(bank, 0, -1)  # (n, n, q+1)
        if order == 2:
            array = np.ascontiguousarray(powers, dtype=np.float32)
        else:
            n, channels = powers.shape[0], powers.shape[-1]
            shape = (n, n, n, channels)
            array = np.concatenate([
                np.broadcast_to(powers[:, :, None, :], shape),
                np.broadcast_to(powers[:, None, :, :], shape),
                np.broadcast_to(powers[None, :, :, :], shape),
            ], axis=-1).astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise FloatingPointError('Nonfinite fixed model input')
        if self.cache_bytes:
            while self.arrays and self.cached_bytes + array.nbytes > self.cache_bytes:
                _, evicted = self.arrays.popitem(last=False)
                self.cached_bytes -= evicted.nbytes
            if array.nbytes <= self.cache_bytes:
                self.arrays[cache_key] = array
                self.cached_bytes += array.nbytes
        return array

    def batch(self, graphs, order, device, dtype=None):
        if len({len(g) for g in graphs}) != 1:
            raise ValueError('Equal node counts required within each model batch')
        key = tuple(hashlib.sha256(np.asarray(g, dtype=np.float32).tobytes()).digest()
                    for g in graphs)
        cache_key = (key, order, str(device), str(dtype))
        cached = self.tensors.get(cache_key)
        if cached is not None:
            self.tensors.move_to_end(cache_key)
            return cached
        values = np.stack([self.array(g, order) for g in graphs])
        tensor = torch.from_numpy(values).movedim(-1, 1).contiguous()
        if dtype is not None:
            tensor = tensor.to(dtype)
        tensor = tensor.to(device)
        if self.tensor_cache_bytes:
            nbytes = tensor.numel()*tensor.element_size()
            while self.tensors and self.tensor_bytes + nbytes > self.tensor_cache_bytes:
                _, evicted = self.tensors.popitem(last=False)
                self.tensor_bytes -= evicted.numel()*evicted.element_size()
            if nbytes <= self.tensor_cache_bytes:
                self.tensors[cache_key] = tensor
                self.tensor_bytes += nbytes
        return tensor


def channel_mlp(order, ins, outs, depth=1):
    """1x1 channel convolutions; the reference config uses ``mlp_depth: 1``."""
    conv = nn.Conv2d if order == 2 else nn.Conv3d
    layers = []
    for _ in range(depth):
        layers.append(conv(ins, outs, 1, bias=True))
        ins = outs
    return nn.Sequential(*layers)


class RegularBlock(nn.Module):
    """Reference ``RegularBlock``: branches -> contraction -> signed root -> skip."""

    def __init__(self, order, ins, width, mlp_depth=1):
        super().__init__()
        self.order = order
        self.alpha = 1.0/order
        # The reference pair block uses the bare signed square root, whose
        # gradient diverges at zero; on small graphs that made ~6% of the
        # order-2 realisation runs produce a nonfinite loss. The stabilised
        # rho form (same shift the triple block always used) removes that.
        self.eps = 1e-6
        self.branches = nn.ModuleList([
            channel_mlp(order, ins, width, mlp_depth) for _ in range(order)])
        conv = nn.Conv2d if order == 2 else nn.Conv3d
        self.skip = conv(ins + width, width, 1, bias=True)

    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        if self.order == 2:
            raw = branches[0] @ branches[1]
        else:
            raw = _Contracted3.apply(
                *(branch.movedim(1, -1) for branch in branches)).movedim(-1, 1)
        joint = signed_root(raw, self.alpha, self.eps)
        return self.skip(torch.cat([x, joint], dim=1))


class DiagOffdiagAvgPooling(nn.Module):
    """Pair node readout: self entry vs mean off-diagonal (reference pooling)."""

    @staticmethod
    def forward(pair_h):
        diagonal = torch.diagonal(pair_h, dim1=-2, dim2=-1)
        offdiag = (pair_h.sum(dim=-1) - diagonal) / max(pair_h.shape[-1] - 1, 1)
        return torch.cat([diagonal, offdiag], dim=1).permute(0, 2, 1).contiguous()


class TripleNodePooling(nn.Module):
    """Triple node readout: main diagonal, three diagonal planes, marginal."""

    @staticmethod
    def forward(pair_h, node_counts=None):
        if node_counts is None:
            node_counts = torch.full((pair_h.shape[0],), pair_h.shape[2],
                                     device=pair_h.device, dtype=pair_h.dtype)
        b, c, n = pair_h.shape[:3]
        idx = torch.arange(n, device=pair_h.device, dtype=torch.long)
        counts = node_counts.to(pair_h).clamp(min=1.0).view(b, 1, 1)
        diag = pair_h[:, :, idx, idx, idx]
        line1 = pair_h.diagonal(dim1=3, dim2=4).sum(dim=-1)
        line2 = pair_h.diagonal(dim1=2, dim2=4).sum(dim=-1)
        line3 = pair_h.diagonal(dim1=2, dim2=3).sum(dim=-1)
        marginal = (pair_h.sum(dim=(3, 4)) + pair_h.sum(dim=(2, 4)) +
                    pair_h.sum(dim=(2, 3))) / (3.0*counts.pow(2))
        node_h = torch.cat([diag, line1/counts, line2/counts, line3/counts, marginal],
                           dim=1)
        return node_h.permute(0, 2, 1).contiguous()


class RealizationPPGN(nn.Module):
    """Reference PPGN backbone: batch norm, per-block readout sum, node-mean head."""

    def __init__(self, order, q=8, width=32, layers=5, output_dim=16, mlp_depth=1,
                 grad_checkpoint=False):
        super().__init__()
        self.order = order
        # Recompute block activations in backward; only used by the memory
        # fallback for pairs whose float32 ladder cannot fit.
        self.grad_checkpoint = grad_checkpoint
        ins = input_channels(order, q)
        self.blocks = nn.ModuleList([
            RegularBlock(order, ins if index == 0 else width, width, mlp_depth)
            for index in range(layers)])
        norm = nn.BatchNorm2d if order == 2 else nn.BatchNorm3d
        self.norms = nn.ModuleList([norm(width) for _ in range(layers)])
        node_dim = 2*width if order == 2 else 5*width
        self.readout_fc = nn.ModuleList([nn.Linear(node_dim, width) for _ in range(layers)])
        self.pooling = DiagOffdiagAvgPooling() if order == 2 else TripleNodePooling()
        self.classifier = nn.Linear(width, output_dim)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        node_h = None
        for index, block in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            x = self.norms[index](x)
            node_x = self.pooling(x)
            contribution = self.readout_fc[index](node_x)
            node_h = contribution if node_h is None else node_h + contribution
        return self.classifier(node_h.mean(dim=1))
