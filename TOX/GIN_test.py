import torch
from torch_geometric.nn import GINConv
import pandas as pd

class GINTest(torch.nn.Module):
    def __init__(self, embedding_size):
        super(GINTest, self).__init__()
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(79, embedding_size),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU()
        )
        self.ginconv1 = GINConv(self.mlp1, eps=0.00005, train_eps=True)
        self.out_mlp1 = torch.nn.Linear(embedding_size, embedding_size)