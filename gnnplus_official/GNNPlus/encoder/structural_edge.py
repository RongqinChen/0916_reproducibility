"""Common single-domain adapter; legacy class name is retained for configs."""
from GNNPlus.encoder.endpoint_addition import EndpointAddition


class StructuralEdgeAddition(EndpointAddition):
    def __init__(self, input_width, hidden_width, raw_norm_type, injection='edge'):
        super().__init__(input_width, hidden_width, injection, raw_norm_type)

    def forward(self, batch):
        extra = self.encode(batch.pestat_StructuralEdge)
        target = batch.x if self.injection == 'node' else batch.edge_attr
        if target is None or target.shape != extra.shape:
            raise ValueError('Matched structural encoding requires native embeddings at hidden width')
        if self.injection == 'node':
            batch.x = target + extra
        else:
            batch.edge_attr = target + extra
        return batch
