"""The two-block, width-16 pair/triple model specified in Appendix P.

The architecture follows the manuscript and the previous local implementation;
this module is self-contained. Triple contraction uses an explicit backward to
avoid retaining one full activation tensor for each pivot.
"""
import torch
from torch import nn


class TripleContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second, third):
        out = torch.zeros_like(first)
        for t in range(first.shape[2]):
            out += first[:, :, t].unsqueeze(2)*second[:, :, :, t].unsqueeze(3)*third[:, :, :, :, t].unsqueeze(4)
        ctx.save_for_backward(first, second, third)
        return out

    @staticmethod
    def backward(ctx, grad):
        a, b, c = ctx.saved_tensors
        da, db, dc = torch.zeros_like(a), torch.zeros_like(b), torch.zeros_like(c)
        for t in range(a.shape[2]):
            at, bt, ct = a[:, :, t], b[:, :, :, t], c[:, :, :, :, t]
            da[:, :, t] = (grad*bt.unsqueeze(3)*ct.unsqueeze(4)).sum(2)
            db[:, :, :, t] = (grad*at.unsqueeze(2)*ct.unsqueeze(4)).sum(3)
            dc[:, :, :, :, t] = (grad*at.unsqueeze(2)*bt.unsqueeze(3)).sum(4)
        return da, db, dc


class Block(nn.Module):
    def __init__(self, order, ins, width):
        super().__init__()
        self.order = order
        conv = nn.Conv2d if order == 2 else nn.Conv3d
        self.branches = nn.ModuleList([conv(ins, width, 1) for _ in range(order)])
        self.projection = conv(ins+width, width, 1)
        self.norm = (nn.BatchNorm2d if order == 2 else nn.BatchNorm3d)(width)

    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        z = branches[0] @ branches[1] if self.order == 2 else TripleContraction.apply(*branches)
        z = z.sign()*((z.abs()+1e-6)**(1/self.order)-1e-6**(1/self.order))
        return self.norm(self.projection(torch.cat([x, z], dim=1)))


def pool(x, order):
    n = x.shape[2]
    if order == 2:
        diagonal = x.diagonal(dim1=2, dim2=3)
        return torch.cat([diagonal, (x.sum(3)-diagonal)/max(n-1, 1)], dim=1).movedim(1, 2)
    index = torch.arange(n, device=x.device)
    diagonal = x[:, :, index, index, index]
    planes = [x.diagonal(dim1=a, dim2=b).sum(-1)/n for a, b in ((3, 4), (2, 4), (2, 3))]
    marginal = (x.sum((3, 4))+x.sum((2, 4))+x.sum((2, 3)))/(3*n*n)
    return torch.cat([diagonal, *planes, marginal], dim=1).movedim(1, 2)


class PPGN(nn.Module):
    def __init__(self, order, q=8, width=16, layers=2):
        super().__init__()
        self.order = order
        ins = (q+1)*(1 if order == 2 else 3)
        self.blocks = nn.ModuleList([Block(order, ins if i == 0 else width, width) for i in range(layers)])
        self.heads = nn.ModuleList([nn.Linear(width*(2 if order == 2 else 5), width) for _ in range(layers)])
        self.output = nn.Linear(width, 16)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        result = None
        for block, head in zip(self.blocks, self.heads):
            x = block(x)
            contribution = head(pool(x, self.order))
            result = contribution if result is None else result+contribution
        return self.output(result.mean(1))
