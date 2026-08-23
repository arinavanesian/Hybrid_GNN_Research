import torch
from torch import nn as nn
from torch.nn import functional as F

edge_index = torch.tensor(
    [[1, 2, 3, 4, 1, 2, 3, 4],
    [2, 3, 4, 1, 1, 2, 3, 4]], dtype=torch.long
)

X = torch.tensor([
    [1.5, 0.2, 1.0],  # Node 1 features
    [0.8, 1.1, 0.4],  # Node 2 features
    [0.1, 0.5, 2.3],  # Node 3 features
    [1.2, 0.9, 0.7]   # Node 4 features
], dtype=torch.float32)
F_in = 3
F_out = 5

W = nn.Parameter(torch.randn(F_out, F_in)) #F_out x F_in 5x3
Z = torch.matmul(X, W.T)  # NxF_in @ F_in x F_out = N x F_out
print("Transformed features Z (W * h):\n", Z)

print("\n--- Step 2 & 3: Attention Coefficients (e_ij and alpha_ij) ---")

for src, tgt in zip(edge_index[0], edge_index[1]):
    # [1, 2, 3, 4, 1, 2, 3, 4] Source j+1
    # [2, 3, 4, 1, 1, 2, 3, 4] Target i+1
    i, j = tgt.item() - 1, src.item() - 1   
    concat_features = torch.cat([Z[i], Z[j]], dim=0) 
    print(f'Concatenated features {concat_features}\n')
    a = nn.Parameter(torch.randn(2 * F_out, 1)) #  5*2 = 10 
    print(f"Attention vector a: {a}\n")
    e_ij = F.leaky_relu(torch.matmul(concat_features, a), negative_slope=0.2)
    print(f"Node {i+1} attending to Node {j+1}: e_ij = {e_ij.item():.4f}")
    