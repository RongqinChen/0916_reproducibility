"""Add cross-degree endpoint channels after the existing feature encoder."""
from torch import nn
from torch.nn import functional as F


class EndpointAddition(nn.Module):
    def __init__(self, input_width, hidden_width, injection, raw_norm_type="BatchNorm"):
        super().__init__()
        if injection not in ("node","edge"):
            raise ValueError("Endpoint injection must be node or edge")
        self.injection=injection
        if raw_norm_type not in ("BatchNorm","none"):
            raise ValueError("Endpoint raw_norm_type must be BatchNorm or none")
        self.raw_norm=nn.BatchNorm1d(input_width) if raw_norm_type=="BatchNorm" else nn.Identity()
        self.project=nn.Linear(input_width,hidden_width,bias=False)

    def encode(self,values):
        # Empty edge sets and singleton batches cannot estimate batch variance.
        if len(values)==0:
            return self.project(values)
        if isinstance(self.raw_norm,nn.BatchNorm1d) and self.training and len(values)<2:
            values=F.batch_norm(values,self.raw_norm.running_mean,self.raw_norm.running_var,
                                self.raw_norm.weight,self.raw_norm.bias,training=False,
                                eps=self.raw_norm.eps)
        else:
            values=self.raw_norm(values)
        return self.project(values)

    def forward(self,batch):
        if self.injection=="node":
            batch.x=batch.x+self.encode(batch.pestat_EndpointNode)
        else:
            extra=self.encode(batch.pestat_EndpointEdge)
            if batch.edge_attr is None or batch.edge_attr.shape!=extra.shape:
                raise ValueError("Endpoint edge addition requires encoded edges at hidden width")
            batch.edge_attr=batch.edge_attr+extra
        return batch
