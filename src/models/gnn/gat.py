import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import Literal

class GAT(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        task_level: Literal["node_classification", "graph_classification"] = "node_classification",
        edge_dim: int = None,
    ):
        super(GAT, self).__init__()
        
        self.task_level = task_level
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.out_channels = output_dim
        # Initial layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=num_heads, edge_dim=edge_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_dim))

        # Output layer
        self.layers.append(GATv2Conv(hidden_dim * num_heads, output_dim, heads=1, edge_dim=edge_dim))
        
        # print all the layers
        print(self.layers)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = F.elu(x)
        
        x = self.layers[-1](x, edge_index, edge_attr)  # No activation on final layer

        if self.task_level == "graph_classification":
            x = global_mean_pool(x, batch)  # Aggregate node embeddings into graph-level representation

        return x
