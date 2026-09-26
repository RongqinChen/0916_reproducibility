import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from GNNPlus.layer.gatedgcn_layer import GatedGCNLayer
from GNNPlus.layer.gine_conv_layer import GINEConvLayer
from GNNPlus.encoder.endpoint_addition import EndpointAddition
from GNNPlus.encoder.structural_edge import StructuralEdgeAddition
from GNNPlus.transform.single_pivot_contraction import endpoint_degree_pairs

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        if cfg.posenc_StructuralEdge.enable:
            param = cfg.posenc_StructuralEdge
            if cfg.posenc_RWSE.enable or cfg.posenc_Endpoint.enable:
                raise ValueError('Disable legacy RWSE and Endpoint for matched edge experiments')
            if cfg.gnn.layer_type != 'gine':
                raise ValueError('Matched structural experiments currently require GINE')
            if param.route == 'mixed' and param.width % 2:
                raise ValueError('Mixed requires an even total width')
            width = param.width * (2 if param.route == 'wider_powers' else 1)
            self.structural_edge = StructuralEdgeAddition(
                width, dim_in, param.raw_norm_type, param.injection)

        if cfg.posenc_Endpoint.enable:
            param=cfg.posenc_Endpoint
            if param.qsteps<1:
                raise ValueError("Endpoint qsteps must be positive")
            if param.injection=="edge" and cfg.gnn.layer_type=="gcn":
                raise ValueError("Use an edge-aware backbone for endpoint edge injection")
            self.endpoint=EndpointAddition(
                len(endpoint_degree_pairs(param.qsteps, param.add_identity,
                                          param.channel_budget,
                                          param.exclude_constant,
                                          param.symmetric_channels)),
                dim_in,param.injection,param.raw_norm_type)

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual,ffn=cfg.gnn.ffn))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcn':
            return GatedGCNLayer
        elif model_type == 'gine':
            return GINEConvLayer
        elif model_type == 'gatv2':
            from GNNPlus.layer.gatv2_conv_layer import GATv2ConvLayer
            return GATv2ConvLayer
        elif model_type == 'gcn':
            from GNNPlus.layer.gcn_conv_layer import GCNConvLayer
            return GCNConvLayer
        elif model_type == 'gcne':
            from GNNPlus.layer.gcn_conv_layer_e import GCNConvLayer
            return GCNConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
