import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

torch.manual_seed(42)


class CustomGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        node_feat = torch.tensor(graph["node_feat"], dtype=torch.float)
        edge_attr = torch.tensor(graph["edge_attr"], dtype=torch.float)
        idx = torch.tensor(idx, dtype=torch.long)
        return Data(edge_index=edge_index, x=node_feat, edge_attr=edge_attr, idx=idx)


def create_loader(dataset, batch_size, shuffle):
    dataset = CustomGraphDataset(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
