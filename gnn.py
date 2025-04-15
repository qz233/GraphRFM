import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.utils import to_undirected, scatter


class ReadoutHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch=None):
        if batch is not None:
            x = scatter(x, batch, dim=0, reduce='mean')
        x = self.linear(x)
        return x


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kind="gcn", dropout_rate=0.5, id_initialize=False):
        super().__init__()
        if kind == "gcn":
            self.conv = GCNConv(input_dim, output_dim)
        elif kind == "gin":
            self.conv = GINConv(nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)))
        elif kind == "gat":
            self.conv = GATConv(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout_rate=0.5,
        id_initialize=False
    ):
        super().__init__()
        if not hidden_dim:
            hidden_dim = 2 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim)
        if id_initialize:
            self.scale = nn.Parameter(torch.tensor([1e-6]))

    def forward(self, x):
        x = self.norm(x)
        out = self.w2(F.silu(self.w1(x)))
        out = self.dropout(out)
        return out


class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False):
        super().__init__()
        assert num_layers >= 1
        #self.in_conv = GCNConv(input_dim, hidden_dim)
        self.layers = nn.ModuleList([])
        #for i in range(num_layers):
        #    self.layers.append(GCNLayer(hidden_dim, hidden_dim, dropout_rate, id_initialize=False))
        self.readout = ReadoutHead(hidden_dim, output_dim)
        self.agg = agg
        self.residual = residual

    def forward(self, inputs):
        x, edge_index = inputs.x, inputs.edge_index
        # In conv
        x = self.in_conv(x, edge_index)
        for layer in self.layers:
            if self.residual:
                x = x + layer(x, edge_index)
            else:
                x = layer(x, edge_index)
        if self.agg:
            return self.readout(x, inputs.batch)
        else:
            return self.readout(x)

    def feature(self, inputs):
        x, edge_index = inputs.x, inputs.edge_index
        # In conv
        features = []
        features.append(x.clone())
        x = self.in_conv(x, edge_index)
        features.append(x.clone())
        for layer in self.layers:
            if self.residual:
                x = x + layer(x, edge_index)
            else:
                x = layer(x, edge_index)
            features.append(x.clone())
        return features


class GCN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False)
        self.in_conv = GCNConv(input_dim, hidden_dim)
        for i in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, dropout_rate=dropout_rate, kind="gcn", id_initialize=False))

class GIN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False)
        self.in_conv = GCNConv(input_dim, hidden_dim)
        for i in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, dropout_rate=dropout_rate, kind="gin", id_initialize=False))

class GAT(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, residual=True, agg=False)
        self.in_conv = GCNConv(input_dim, hidden_dim)
        for i in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, dropout_rate=dropout_rate, kind="gat", id_initialize=False))
        self.readout = ReadoutHead(hidden_dim, output_dim)



class SMPNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5, id_initialize=False, agg=False):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, dropout_rate=dropout_rate, kind="gcn", id_initialize=id_initialize) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            FeedForward(hidden_dim, hidden_dim, dropout_rate, id_initialize=id_initialize) for _ in range(num_layers)
        ])
        self.in_proj = nn.Linear(input_dim, hidden_dim)#GCNConv(input_dim, hidden_dim)
        self.readout = ReadoutHead(hidden_dim, output_dim)#GCNConv(hidden_dim, output_dim)
        self.agg = agg

    def forward(self, inputs):
        x, edge_index = inputs.x, inputs.edge_index
        x = self.in_proj(x)
        for gcn, ff in zip(self.gcn_layers, self.ff_layers):
            x = gcn(x, edge_index) + x
            x = ff(x) + x
        if self.agg:
            return self.readout(x, inputs.batch)
        else:
            return self.readout(x)
    
    def feature(self, inputs):
        x, edge_index = inputs.x, inputs.edge_index
        # In conv
        features = []
        x = self.in_proj(x)
        features.append(x.clone())
        for gcn, ff in zip(self.gcn_layers, self.ff_layers):
            x = gcn(x, edge_index) + x
            x = ff(x) + x
            features.append(x.clone())
        return features
    



def build_model(config):
    if config.gnn_model == "gcn":
        model = GCN(config.num_features,  config.num_classes, config.hidden_dim, config.num_layers, dropout_rate=config.dropout_rate)
    elif config.gnn_model == "gin":
        model = GIN(config.num_features,  config.num_classes, config.hidden_dim, config.num_layers, dropout_rate=config.dropout_rate)
    elif config.gnn_model == "gat":
        model = GAT(config.num_features,  config.num_classes, config.hidden_dim, config.num_layers, dropout_rate=config.dropout_rate)
    elif config.gnn_model == "smpnn":
        model = SMPNN(config.num_features,  config.num_classes, config.hidden_dim, config.num_layers, dropout_rate=config.dropout_rate)
    else:
        raise NotImplementedError("not a valid model")
    if hasattr(config, "pretrain_path") and config.pretrain_path is not None:
        model.load_state_dict(torch.load(config.pretrain_path, map_location="cpu"))
    return model