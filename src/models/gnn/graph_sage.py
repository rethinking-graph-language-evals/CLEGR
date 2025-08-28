import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn.conv import GATConv, GCNConv, SAGEConv
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn.dense.linear import Linear


class SAGEConv(SAGEConv):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        aggr="mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        edge_dim=None,
        **kwargs
    ):
        super().__init__(
            in_channels,
            out_channels,
            aggr,
            normalize,
            root_weight,
            project,
            bias,
            **kwargs
        )
        if edge_dim is not None:
            print("use edge")
            self.lin_edge = Linear(
                edge_dim, in_channels, bias=False, weight_initializer="glorot"
            )
            self.lin_edge.reset_parameters()
        else:
            self.lin_edge = None
            print("not use edge")

    def forward(self, x, edge_index, edge_attr=None, size=None):

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, "lin"):
            x = (self.lin(x[0]).relu(), x[1])
            
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        
        if self.lin_edge is not None and edge_attr is not None:
            row, col = edge_index  # src, dst
            e = self.lin_edge(edge_attr)  # [E, out_channels]
            out = out + scatter(e, col, dim=0,
                                dim_size=out.size(0),
                                reduce=self.aggr)
        
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            return x_j + edge_attr
        else:
            return x_j


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        num_proj_hidden,
        dropout,
        activation=F.relu,
        graph_pooling="mean",
        edge_dim=None,
        gnn_type="sage",
        task_level="node_classification",
        glm='tea-glm'
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = hidden_channels
        self.n_classes = out_channels
        self.convs = torch.nn.ModuleList()
        self.task_level = task_level
        if gnn_type == "sage":
            gnn_conv = SAGEConv
        elif gnn_type == "gat":
            gnn_conv = GATConv
        elif gnn_type == "gcn":
            gnn_conv = GCNConv


        conv_kwargs = {}
        if gnn_type in ['sage', 'gat']:
            conv_kwargs['edge_dim'] = edge_dim

        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels, edge_dim=edge_dim))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels, edge_dim=edge_dim))
            self.convs.append(gnn_conv(hidden_channels, out_channels, edge_dim=edge_dim))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels, edge_dim=edge_dim))


        if glm == 'tea-glm':
            # non-linear layer for contrastive loss
            self.fc1 = torch.nn.Linear(out_channels, num_proj_hidden)
            self.fc2 = torch.nn.Linear(num_proj_hidden, out_channels)

        self.dropout = dropout
        self.activation = activation

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        
        print(self)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)

            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task_level == "graph_classification":
            x = self.pool(x, batch)
        
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description("Evaluating")

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[: batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

