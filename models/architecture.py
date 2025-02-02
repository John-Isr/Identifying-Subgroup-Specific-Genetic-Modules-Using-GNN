import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm, GATv2Conv

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim=101, output_dim=1, dropout=0.3, heads=4):
        super(GNNClassifier, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, edge_dim=edge_dim, heads=heads)
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, edge_dim=edge_dim, heads=heads)
        self.norm2 = GraphNorm(hidden_dim * heads)
        self.dropout = nn.Dropout(dropout)
        # MLP layers
        self.mlp_hidden1 = nn.Linear(hidden_dim * heads, 128)
        self.mlp_hidden2 = nn.Linear(128, 64)
        self.mlp_output = nn.Linear(64, output_dim)


    def forward(self, x, edge_index, edge_attr):
        # GNN forward pass
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        # x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        # x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # MLP forward pass
        x = self.mlp_hidden1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.mlp_hidden2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.mlp_output(x)
        return x


#TODO: copy the init_weights function from the previous notebook, xavier and kaiming