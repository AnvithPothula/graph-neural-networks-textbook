# Lecture 6: Graph Neural Networks — Foundations

## Overview

This is the heart of the course. GNNs learn node representations by iteratively aggregating information from neighbors — generalizing convolution to graphs.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/03-GNN1.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/06-GNN1.pdf

---

## Key Concepts

### Motivation
- Node embeddings (DeepWalk, node2vec) are transductive and don't use node features
- We want a function f: (A, X) → Z that:
  - Takes graph structure A and node features X
  - Outputs node embeddings Z
  - Can generalize to new nodes/graphs (inductive)

### The Core Idea: Neighborhood Aggregation
- **Computation graph:** for each node, define a computation tree using its neighborhood
- Node v at layer k uses info from its k-hop neighborhood
- **Two operations per layer:**
  1. **Aggregate:** collect messages from neighbors
  2. **Update (Combine):** combine aggregated message with current node embedding

### GNN Layer (General Form)
```
h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1) : u ∈ N(v)}))
```

### Graph Convolutional Network (GCN) — Kipf & Welling 2017
- Aggregation: mean of neighbor embeddings (normalized by degree)
- **GCN formula:**
  H^(k+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(k) W^(k))
  where Ã = A + I (self-loops), D̃ = degree matrix of Ã
- Intuition: each node's new embedding = weighted average of its own + neighbor embeddings, then linear transform + nonlinearity

### Training GNNs
- **Node classification:** cross-entropy loss on labeled nodes
- **Link prediction:** binary cross-entropy with negative sampling
- **Graph classification:** apply a readout function to pool all node embeddings
- **Supervised:** when labels are available
- **Unsupervised:** use structure-based loss (e.g., nearby nodes should have similar embeddings)

### Computation Graph = Receptive Field
- With K layers, each node's embedding reflects its K-hop neighborhood
- **Key insight:** GNNs do NOT need to share computation across different nodes — each node has its own computation graph

### Connection to Spectral Graph Theory
- GCN can be derived from spectral convolution using Chebyshev polynomial approximation
- Efficient first-order approximation leads to the GCN formula above

## Key Papers
- [GCN: Semi-Supervised Classification with GCN (Kipf & Welling, 2017)](https://arxiv.org/pdf/1609.02907.pdf)
- [Geometric Deep Learning (Bronstein et al., 2021)](https://arxiv.org/abs/2104.13478)

## Textbook Chapter Notes

Maps to **Chapter 6: Graph Neural Networks — Foundations**.
- Derivation of the GCN formula step by step
- Code: implement GCN from scratch with PyTorch; then with PyG
- MicroSim idea: interactive message passing visualization — click a node, watch messages flow in from neighbors across layers
- Connection back to node embeddings: GNNs subsume shallow embeddings

## Code Example (PyTorch Geometric)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```
