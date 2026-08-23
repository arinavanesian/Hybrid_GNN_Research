import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define the Graph Structure based on your description:
# 1 -> 2, 2 -> 3, 4 -> 3 (and let's include self-loops as GAT typically does)
# Edges format: [source_nodes, target_nodes]
edge_index = torch.tensor([
    [1, 2, 4, 1, 2, 3, 4], # Source nodes
    [2, 3, 3, 1, 2, 3, 4]  # Target nodes (includes self-loops for 1, 2, 3, 4)
], dtype=torch.long)

# 2. Define Input Node Features (4 nodes, F = 2 features each)
# e.g., Node 1, 2, 3, 4 have simple 2D feature vectors
X = torch.tensor([
    [1.0, 0.5],  # Node 1
    [0.5, 1.0],  # Node 2
    [0.2, 0.8],  # Node 3
    [0.9, 0.1]   # Node 4
], dtype=torch.float32)

N = X.size(0) # Number of nodes (4)
F_in = 2      # Input feature dimension
F_out = 3     # Output transformed feature dimension

print(f"--- Step 1: Linear Transformation W ({F_in} -> {F_out}) ---")
# Shared weight matrix W mapping F_in to F_out
W = nn.Parameter(torch.randn(F_out, F_in))
print("Weight matrix W:\n", W)
Z = torch.matmul(X, W.T) # Shape: [4, 3]
print("Transformed features Z (W * h):\n", Z)


print("\n--- Step 2 & 3: Attention Coefficients (e_ij and alpha_ij) ---")
# Attention vector a^T of size 2 * F_out
# Because we concatenate [W*hi || W*hj], the vector length is 2 * F_out
a = nn.Parameter(torch.randn(2 * F_out, 1))

# Let's manually compute attention for Node 1 and its neighbors
# Neighbors of Node 1 in our edge_index: Node 1 (self) and Node 2
# Let's look at all edges to see how e_ij is calculated for each connected pair
edge_scores = []
for src, tgt in zip(edge_index[0], edge_index[1]):
    # Note: PyG graphs are directed edges. If 1 -> 2, source is 1, target is 2.
    # In GAT formulation: node i attends to neighbor j. Here let's treat target as i and source as j, 
    # or vice versa depending on message passing direction. Let's do i=tgt, j=src (incoming messages to i).
    i, j = tgt.item() - 1, src.item() - 1  # 0-indexed
    
    # Concatenate transformed features: [W*h_i || W*h_j]
    concat_features = torch.cat([Z[i], Z[j]], dim=0) # Shape: [2 * F_out]
    
    # Apply attention vector and LeakyReLU
    e_ij = F.leaky_relu(torch.matmul(concat_features, a), negative_slope=0.2)
    edge_scores.append((i+1, j+1, e_ij.item()))

print("Raw attention scores (e_ij) for sample edges (Target, Source):")
for target, source, score in edge_scores:
    print(f"Node {target} attending to Node {source}: e_ij = {score:.4f}")


print("\n--- Using PyTorch Geometric's Built-in GATConv for Convenience ---")
from torch_geometric.nn import GATConv

# Initialize official GAT layer (input features=2, output features=3, heads=1)
gat_layer = GATConv(in_channels=2, out_channels=3, heads=1, concat=True)

# PyG expects edge_index to be 0-indexed (0 to 3 instead of 1 to 4)
pyg_edge_index = edge_index - 1

# Forward pass through GAT layer
out_features = gat_layer(X, pyg_edge_index)
print("Final output node features from GAT layer (Shape: 4x3):\n", out_features)