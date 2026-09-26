import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg


class GATv2ConvLayer(nn.Module):
    """GATv2 layer that consumes encoded edge features.

    Wrapped exactly like GINEConvLayer: message passing, dropout, optional
    residual connection and FFN block, with the same BatchNorm placement.
    Edge features enter the attention mechanism through `edge_dim`, so the
    encoded edge width must equal the hidden width (as required by endpoint
    edge injection).
    """

    def __init__(self, dim_in, dim_out, dropout, residual, ffn):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = True
        self.ffn = ffn

        # Not a registered config key: keeping it out of the dump leaves the
        # setting hash of other backbones untouched.
        heads = int(getattr(cfg.gnn, 'attn_heads', 4))
        if dim_out % heads != 0:
            raise ValueError(
                f"dim_inner={dim_out} must be divisible by "
                f"gnn.attn_heads={heads}")
        self.model = pyg_nn.GATv2Conv(dim_in, dim_out // heads, heads=heads,
                                      concat=True, edge_dim=dim_in,
                                      dropout=0.0, add_self_loops=False)

        if self.ffn:
            # Feed Forward block, same as GINEConvLayer.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out * 2)
            self.ff_linear2 = nn.Linear(dim_out * 2, dim_out)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x

        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)

            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch
