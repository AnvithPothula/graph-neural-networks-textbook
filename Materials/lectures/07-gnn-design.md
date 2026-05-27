# Lecture 7: GNN Design Space (GraphSAGE, GAT)

## Overview

There are many design choices in building a GNN — aggregation function, layer depth, skip connections. This lecture covers GraphSAGE, GAT, and a systematic view of the GNN design space.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/04-GNN2.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/07-GNN2.pdf

---

## Key Concepts

### GraphSAGE (Hamilton et al., 2017)
- **Key difference from GCN:** sample a fixed-size neighborhood (instead of using all neighbors)
- **General aggregator:** can use mean, max, LSTM, or pooling
- **Concatenation:** concatenates self embedding with aggregated neighbor embedding (vs. just averaging)
- **Inductive:** can generate embeddings for new, unseen nodes

**GraphSAGE update rule:**
```
h^k_N(v) = AGGREGATE_k({h^(k-1)_u : u ∈ N(v)})
h^k_v = σ(W^k · CONCAT(h^(k-1)_v, h^k_N(v)))
```

**Aggregator variants:**
- Mean: h̃ = mean({h_u : u ∈ N(v)})
- Max pooling: h̃ = max({ReLU(W · h_u) : u ∈ N(v)})
- LSTM: apply LSTM to shuffled sequence of neighbor embeddings

### Graph Attention Network (GAT, Veličković et al., 2018)
- **Key idea:** different neighbors should have different importance (learned weights)
- **Attention coefficient:** α_{ij} = how much node j attends to node i
- **Computation:**
  e_{ij} = a(W·h_i, W·h_j)  (attention score)
  α_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / Σ_{k ∈ N(i)} exp(e_{ik})
  h'_i = σ(Σ_{j ∈ N(i)} α_{ij} W h_j)
- **Multi-head attention:** run K independent attention heads, concatenate (or average) results
- Advantage over GCN: weights are data-dependent; more expressive

### General GNN Design Choices

| Component | Options |
|---|---|
| **Aggregation** | Mean, Sum, Max, LSTM, Attention |
| **Combination** | Concat, Add, MLP |
| **Layer depth** | Typically 2-3 (deeper → over-smoothing) |
| **Activation** | ReLU, ELU, Tanh |
| **Normalization** | Batch norm, Layer norm, Pair norm |
| **Skip connections** | Residual (add), Jumping Knowledge (concat) |

### Design Space Paper Key Findings
- No single architecture dominates all tasks
- Choice of aggregation is more important than choice of GNN type
- Skip connections are almost always beneficial
- 2-3 layers is optimal for most benchmarks

## Key Papers
- [GraphSAGE: Inductive Representation Learning (Hamilton et al., 2017)](https://arxiv.org/pdf/1706.02216.pdf)
- [GAT: Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/pdf/1710.10903.pdf)
- [Design Space of GNNs (You et al., 2020)](https://arxiv.org/pdf/2011.08843.pdf)

## Textbook Chapter Notes

Maps to **Chapter 7: GNN Design Space**.
- Compare GCN vs GraphSAGE vs GAT on Cora dataset
- MicroSim: visualize attention weights as edge thickness in a GAT computation graph

## Code Example

```python
from torch_geometric.nn import SAGEConv, GATConv

# GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

# GAT
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
```
