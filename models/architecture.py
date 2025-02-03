from torch_geometric.nn import GATv2Conv, GraphNorm
import torch.nn as nn
import torch.nn.functional as F


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim=101, output_dim=1,
                 dropout=0.3, heads=4, num_layers=2):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Message passing layers
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * heads
            self.layers.append(GATv2Conv(in_dim, hidden_dim,
                                         edge_dim=edge_dim, heads=heads))
            self.norms.append(GraphNorm(hidden_dim * heads))

        # MLP with safe initialization
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * heads, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        self.mlp.apply(self.init_weights_kaiming)

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def init_weights_kaiming(m):
        """For MLP layers using ReLU"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, x, edge_index, edge_attr):
        # Message passing
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Final MLP
        return self.mlp(x)


def init_weights_xavier(m):
    """
    Xavier initialization for weights, suitable for sigmoid or tanh activation functions.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, GATv2Conv):
        if hasattr(m, 'lin'):  # Initialize the linear layer in GATv2Conv
            nn.init.xavier_uniform_(m.lin.weight)
        if hasattr(m, 'att'):  # Initialize the attention weights
            nn.init.xavier_uniform_(m.att)
