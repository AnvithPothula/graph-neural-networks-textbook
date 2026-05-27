# Lecture 8: GNN Augmentation and Training

## Overview

Practical issues in training GNNs: how to handle graphs without features, over-smoothing, graph data augmentation, and training best practices.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/05-GNN3.pdf

---

## Key Concepts

### Graph Feature Augmentation

When nodes/edges lack features:
- **Constant features:** h_v = 1 for all nodes (structure-only)
- **Degree features:** h_v = one-hot(degree(v))
- **Random features:** h_v ~ N(0, I) (different random init each run)
- **Structural features:** degree, clustering coefficient, PageRank, etc.
- **Learned positional embeddings:** position-aware encodings

When to augment graph structure:
- **Add virtual nodes:** super-node connected to all nodes (improves long-range communication)
- **Add virtual edges:** connect 2-hop neighbors (shortcut paths; used in bipartite graphs)
- **Remove edges:** randomly drop edges during training (DropEdge augmentation)

### Training on Large Graphs

Different settings for training GNNs:
1. **Full-batch training:** use entire graph adjacency matrix; only feasible for small graphs
2. **Mini-batch training with sampling:** sample k neighbors per node per layer (GraphSAGE sampling)
3. **Cluster-GCN:** partition graph into clusters; sample mini-batches of clusters
4. **GraphSAINT:** sample entire subgraphs as mini-batches

### Over-Smoothing Problem
- Deep GNNs (many layers) cause all node embeddings to converge to the same value
- Caused by iterative averaging (low-pass filter)
- **Solutions:**
  - Limit depth (2-3 layers usually sufficient)
  - Skip/residual connections
  - Jumping Knowledge Networks: aggregate from all layers
  - PairNorm: a normalization that prevents collapse
  - DropEdge: randomly remove edges during training

### Graph Pooling (Graph-Level Tasks)
To get a single embedding for the entire graph:
1. **Global mean/sum/max pooling:** simple, loses structural info
2. **Hierarchical pooling (DiffPool):** learn a differentiable soft clustering; pool iteratively
3. **MinCutPool:** learns partitioning with min-cut objective
4. **TopKPool:** retain top-k nodes by a learned score

### Batch Normalization in GNNs
- Apply to each node's embedding: normalize across the batch dimension
- Helps training stability but can remove structural information in some settings
- Layer normalization is an alternative

## Textbook Chapter Notes

Maps to **Chapter 8: GNN Training & Augmentation**.
- Key practical tips for making GNNs work
- Code example: compare deep GCN (over-smoothed) vs. shallow GCN with skip connections
