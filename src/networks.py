import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

torch.manual_seed(42)


class GNN(torch.nn.Module):
    def __init__(self, hidden_1=64, hidden_2=32, hidden_3=16, num_node_features=38):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden_1)
        self.conv2 = GraphConv(hidden_1, hidden_2)
        self.conv3 = GraphConv(hidden_2, hidden_3)
        self.num_params = sum([p.numel() for p in self.parameters()])
        self.output_dim = hidden_3
        self.type = "GNN"

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch.batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = x.tanh()

        return x


class FCN(nn.Module):
    def __init__(self, input_dim=38, hidden_dim=192, output_dim=16):
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

    def __init__(self, twin: nn.Module, NTN_dim=16, hidden_dim_1=8, hidden_dim_2=4):
        super(SiameseNetwork, self).__init__()
        self.twin = twin
        self.tensor = nn.Bilinear(twin.output_dim, twin.output_dim, NTN_dim, bias=False)
        self.lin = nn.Linear(2 * twin.output_dim, NTN_dim)
        self.fc1 = nn.Linear(NTN_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)
        self.num_params = sum([p.numel() for p in self.parameters()])
        self.type = twin.type

    def forward(self, batch_1, batch_2):
        out_1 = self.twin(batch_1)
        out_2 = self.twin(batch_2)
        out = self.tensor(out_1, out_2) + self.lin(torch.cat((out_1, out_2), dim=1))
        out = out.tanh()
        out = self.fc1(out)
        out = out.tanh()
        out = self.fc2(out)
        out = out.tanh()
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out
