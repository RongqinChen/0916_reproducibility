"""PPGN regular matrix-product blocks and their shared-pivot triple extension.

Pair block structure follows GraphPKU/BREC's RegularBlock (two channel MLPs,
channelwise matrix product, concatenation with input, linear skip projection).
Inputs and equality-pattern mean readouts are adapted to this experiment.
https://github.com/GraphPKU/BREC/blob/Release/ProvablyPowerfulGraphNetworks_torch/layers/modules.py
"""
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from torch import nn
from brec.budget_sweep import polynomial_bank
from brec.encodings import feature_chunks, channel_count
from brec.ppgn_models import _Contracted3, signed_root as rho_signed
from torch.utils.checkpoint import checkpoint


class FixedInput:
    """Use the exact fixed-brec NumPy feature kernel; never recompute in torch."""
    def __init__(self, q=8, chunk_size=128, cache_bytes=4*1024**3,
                 tensor_cache_bytes=6*1024**3):
        self.q, self.chunk_size = q, chunk_size
        self.banks = OrderedDict()
        # Feature arrays are rebuilt for every epoch and every seed otherwise; one
        # pass over a pair then 60 epochs of identical work. Cache them under a
        # byte budget (arrays are immutable and bit-identical when rebuilt).
        self.cache_bytes = int(cache_bytes)
        self.arrays = OrderedDict()
        self.cached_bytes = 0
        # Batching repeats the same graph lists for every epoch and seed; caching
        # the stacked channel-first tensor skips the stack/transpose copies too.
        self.tensor_cache_bytes = int(tensor_cache_bytes)
        self.tensors = OrderedDict()
        self.tensor_bytes = 0

    def array(self, adj, order):
        adj = np.asarray(adj, dtype=np.float32)
        key = hashlib.sha256(adj.tobytes()).digest()
        if key not in self.banks:
            self.banks[key] = polynomial_bank(adj, self.q, 'chebyshev')
            if len(self.banks) > 128:
                self.banks.popitem(last=False)
        cache_key = (key, order)
        cached = self.arrays.get(cache_key)
        if cached is not None:
            self.arrays.move_to_end(cache_key)
            return cached
        bank = self.banks[key]
        n = len(adj)
        array = np.empty((n**order, channel_count(order,self.q,'polynomial')),dtype=np.float32)
        offset = 0
        for chunk in feature_chunks(adj,bank,self.q,order,'polynomial',self.chunk_size):
            array[offset:offset+len(chunk)] = chunk
            offset += len(chunk)
        if not np.isfinite(array).all():
            raise FloatingPointError('Nonfinite fixed model input')
        array = array.reshape((n,)*order+(-1,))
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
        key=tuple(hashlib.sha256(np.asarray(g,dtype=np.float32).tobytes()).digest()
                  for g in graphs)
        cache_key=(key,order,str(device),str(dtype))
        cached=self.tensors.get(cache_key)
        if cached is not None:
            self.tensors.move_to_end(cache_key)
            return cached
        values = np.stack([self.array(g,order) for g in graphs])
        tensor = torch.from_numpy(values).movedim(-1,1).contiguous()
        if dtype is not None:
            tensor = tensor.to(dtype)
        tensor = tensor.to(device)
        if self.tensor_cache_bytes:
            nbytes=tensor.numel()*tensor.element_size()
            while self.tensors and self.tensor_bytes + nbytes > self.tensor_cache_bytes:
                _, evicted = self.tensors.popitem(last=False)
                self.tensor_bytes -= evicted.numel()*evicted.element_size()
            if nbytes <= self.tensor_cache_bytes:
                self.tensors[cache_key]=tensor
                self.tensor_bytes += nbytes
        return tensor


def channel_mlp(order, ins, outs, depth=2):
    conv = nn.Conv2d if order == 2 else nn.Conv3d
    layers=[]
    for _ in range(depth):
        layers.extend([conv(ins,outs,1),nn.ReLU()])
        ins=outs
    return nn.Sequential(*layers)


class RegularBlock(nn.Module):
    def __init__(self, order, ins, width):
        super().__init__()
        self.order=order
        self.branches=nn.ModuleList([channel_mlp(order,ins,width) for _ in range(order)])
        conv=nn.Conv2d if order==2 else nn.Conv3d
        self.skip=conv(ins+width,width,1)

    def forward(self, x):
        branches=[f(x) for f in self.branches]
        if self.order==2:
            joint=branches[0] @ branches[1]
        else:
            joint=_Contracted3.apply(*(b.movedim(1,-1) for b in branches)).movedim(-1,1)
            # Retain the historical triple model's signed cube-root stabilization.
            joint=rho_signed(joint,1./3.,1e-6)
        return self.skip(torch.cat([x,joint],dim=1))


def pattern_means(x, order):
    """Means within equality patterns; empty patterns contribute zero."""
    n=x.shape[-1]
    coord=torch.meshgrid(*([torch.arange(n,device=x.device)]*order),indexing='ij')
    if order==2:
        equal=coord[0]==coord[1]
        masks=[equal,~equal]
    else:
        uv,uw,vw=coord[0]==coord[1],coord[0]==coord[2],coord[1]==coord[2]
        masks=[uv&uw,uv&~uw,uw&~uv,vw&~uv,~uv&~uw&~vw]
    flat=x.flatten(2)
    weights=torch.stack([m.flatten().to(x.dtype) for m in masks])
    weights=weights/weights.sum(1,keepdim=True).clamp(min=1)
    return torch.einsum('bct,pt->bcp',flat,weights).flatten(1)


class RealizationPPGN(nn.Module):
    def __init__(self, order, q=8, width=16, layers=4, output_dim=16,
                 grad_checkpoint=False, input_width=None):
        super().__init__()
        self.order=order
        # Recompute block activations in backward; only used by the memory
        # fallback for pairs whose float32 ladder cannot fit.
        self.grad_checkpoint=grad_checkpoint
        ins=channel_count(order,q,'polynomial') if input_width is None else int(input_width)
        self.blocks=nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(RegularBlock(order,ins,width))
            ins=width
        patterns=2 if order==2 else 5
        self.head=nn.Sequential(nn.Linear(patterns*width*layers,width),nn.ReLU(),
                                nn.Linear(width,output_dim))
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Conv2d,nn.Conv3d)):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        pooled=[]
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x=checkpoint(block,x,use_reentrant=False)
            else:
                x=block(x)
            pooled.append(pattern_means(x,self.order))
        return self.head(torch.cat(pooled,dim=1))

# Borrowed backbone provenance. Input width is configurable for matched augmentations.
LEGACY_SOURCE_SHA256 = 'ab8ef95f8ba8eeaadb144438e00482aab0283b10c0f937a7ebcdc51604291526'
