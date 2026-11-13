"""GNNPlus GINE model adapter for the protein encoding grid."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch import nn


GNNPLUS_ROOT = Path(__file__).resolve().parents[1] / 'gnnplus_official'
if str(GNNPLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(GNNPLUS_ROOT))

import GNNPlus  # noqa: E402,F401 -- register official layers, heads and pooling
from GNNPlus.head.san_graph import SANGraphHead  # noqa: E402
from GNNPlus.layer.gine_conv_layer import GINEConvLayer  # noqa: E402
from torch_geometric.graphgym.config import cfg as graph_cfg, set_cfg  # noqa: E402


class IdentityTransform:
    """Skip a transform that is only needed by the replaced triple backbone."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        return data


class GNNPlusGINE(nn.Module):
    """Official GNNPlus GINE layers with structural pair channels as edge input."""

    def __init__(self, node_dim, edge_dim, hidden_dim, classes, *, layers,
                 dropout, residual, ffn, head_layers, activation, pooling):
        super().__init__()
        set_cfg(graph_cfg)
        graph_cfg.gnn.act = activation
        graph_cfg.gnn.dropout = dropout
        graph_cfg.gnn.layers_post_mp = head_layers
        graph_cfg.model.graph_pooling = pooling

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINEConvLayer(hidden_dim, hidden_dim, dropout, residual, ffn)
            for _ in range(layers)
        ])
        self.head = SANGraphHead(hidden_dim, classes, L=head_layers)

    @staticmethod
    def _edge_struct(batch):
        source, target = batch.edge_index
        graph = batch.batch[source]
        if not torch.equal(graph, batch.batch[target]):
            raise ValueError('Edge crosses graph boundaries')
        sizes = batch.ptr[1:] - batch.ptr[:-1]
        local_source = source - batch.ptr[graph]
        local_target = target - batch.ptr[graph]
        rows = (batch.pair_x_ptr[graph]
                + local_source * sizes[graph] + local_target)
        return batch.pair_x[rows]

    def forward(self, batch):
        edge_struct = self._edge_struct(batch)
        native_x = batch.x
        if native_x is None:
            native_x = edge_struct.new_zeros((batch.num_nodes, 0))
        node_input = torch.cat((native_x, batch.loop_x), dim=1)

        native_edge = getattr(batch, 'edge_attr', None)
        if (native_edge is None or native_edge.ndim != 2
                or native_edge.size(0) != batch.edge_index.size(1)):
            native_edge = edge_struct.new_zeros((edge_struct.size(0), 0))
        edge_input = torch.cat((native_edge.to(edge_struct.dtype), edge_struct), dim=1)

        batch.x = self.node_encoder(node_input)
        batch.edge_attr = self.edge_encoder(edge_input)
        for layer in self.layers:
            batch = layer(batch)
        prediction, _ = self.head(batch)
        batch.graph_pred = prediction
        return batch


def register_model(config):
    """Install a model factory at the Co-Sparsify model-construction boundary."""
    import models
    import utils

    module = types.ModuleType('models.model_construction')
    module.__package__ = 'models'

    def make_model():
        current = utils.cfg
        settings = config['gnnplus']
        poly_dim = current.dataset.poly_dim
        return GNNPlusGINE(
            current.dataset.node_attr_dim + poly_dim,
            current.dataset.edge_attr_dim + poly_dim,
            settings['hidden_dim'],
            current.model.num_tasks,
            layers=settings['layers'],
            dropout=settings['dropout'],
            residual=settings['residual'],
            ffn=settings['ffn'],
            head_layers=settings['head_layers'],
            activation=settings['activation'],
            pooling=settings['pooling'],
        )

    module.make_model = make_model
    sys.modules['models.model_construction'] = module
    models.model_construction = module
    return module
