# Lecture 10: Designing Powerful Graph Encoders

## Overview

Building on GNN theory, this lecture explores practical architectures that go beyond 1-WL: position-aware GNNs, identity-aware GNNs, and other powerful encoder designs.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/07-theory2.pdf

---

## Key Concepts

### Recap: 1-WL Limitations
- Cannot count triangles or cycles of specific length
- Cannot distinguish nodes with same local structure but different global positions
- Cannot detect structural roles (e.g., two different bridge nodes in different parts of a graph)

### Position-aware GNNs (P-GNN)
- **Problem:** two nodes may have identical computation graphs but different graph positions
- **Solution:** add positional features from a set of anchor nodes
  - Select S anchor nodes; compute shortest-path distance from each node to each anchor
  - Use these distances as additional node features
- Enables position-sensitive computations while preserving permutation equivariance

### Identity-aware GNNs (ID-GNN)
- **Key idea:** when computing a node's embedding, mark that node distinctively
- Run GNN on an "ego graph" where the central node has a unique feature
- Preserves structure-sensitive computation without explicit position information
- Efficiently implemented by sampling and reusing computation graphs

### Structural Encodings
- **Random walk structural encoding (RWSE):** use landing probabilities of random walks as node features
- **Graphlet degree vectors:** count small subgraph patterns around each node
- **Laplacian eigenvectors:** use graph Laplacian eigenvectors as positional encodings (used in Graph Transformers)

### Graph-Level Encoders for Classification
- After getting node embeddings, need a graph-level summary
- Options:
  - Simple global pooling (mean/sum/max)
  - **DiffPool:** soft clustering into hierarchical coarsening
  - **Hierarchical message passing:** alternating coarsening and refinement

## Key Papers
- [Position-aware GNNs (You et al., 2019)](https://arxiv.org/pdf/1906.04817)
- [Identity-aware GNNs (You et al., 2021)](https://arxiv.org/pdf/2101.10320)
- [Counting Graph Substructures (Zhao et al., 2022)](https://openreview.net/pdf?id=qaJxPhkYtD)
- [DiffPool (Ying et al., 2018)](https://arxiv.org/pdf/1806.08804.pdf)

## Textbook Chapter Notes

Maps to **Chapter 10: Designing Powerful Graph Encoders**.
- Benchmark comparison: GCN vs. GraphSAGE vs. GAT vs. GIN on TU datasets
