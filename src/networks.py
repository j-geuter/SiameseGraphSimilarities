import torch
import torch.nn as nn

from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

torch.manual_seed(42)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_node_features=38, num_edge_features=3):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.num_params = sum([p.numel() for p in self.parameters()])
        self.output_dim = hidden_channels
        self.type = "GNN"

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.relu()

        return x


class FCN(nn.Module):
    def __init__(self, input_dim=38, hidden_dim=128, output_dim=64):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.num_params = sum([p.numel() for p in self.parameters()])
        self.output_dim = output_dim
        self.type = "FCN"

    def forward(self, batch):
        x = batch.x
        x = global_mean_pool(x, batch.batch)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()

        return x


class SiameseNetwork(nn.Module):

    def __init__(self, twin: nn.Module):
        super(SiameseNetwork, self).__init__()
        self.twin = twin
        self.fc = nn.Linear(twin.output_dim, 1)
        self.num_params = sum([p.numel() for p in self.parameters()])
        self.type = twin.type

    def forward(self, batch_1, batch_2):
        out_1 = self.twin(batch_1)
        out_2 = self.twin(batch_2)
        out = out_1 * out_2
        out = out.sum(dim=1)
        out = self.fc(out)
        out = F.sigmoid(out)
        return out
