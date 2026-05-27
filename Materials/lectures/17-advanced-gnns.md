# Lecture 17: Advanced Topics in GNNs

## Overview

This lecture covers recent advances pushing GNN capabilities: in-context learning on graphs, uncertainty quantification, scalable GNN training methods, and community structure detection.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/14-advanced_gnns.pdf

---

## Key Concepts

### In-Context Learning on Graphs (PRODIGY)
- **Problem:** can a single pre-trained GNN perform on new tasks without fine-tuning?
- **PRODIGY (2023):** prompting GNNs with in-context examples (like few-shot learning)
  - A "prompt graph" contains labeled examples of the new task
  - The model attends to these examples when making predictions
  - Enables zero/few-shot transfer across different graph tasks

### Conformalized GNNs (Uncertainty Quantification)
- Standard GNNs output point predictions without uncertainty estimates
- **Conformal prediction:** provides statistically valid prediction sets
  - Gives a set C(x) such that P(y ∈ C(x)) ≥ 1 - α (coverage guarantee)
- **DAPS (Diffusion Adaptive Prediction Sets):** applies conformal prediction to graph data
  - Accounts for graph topology in calibration
  - Provides tighter prediction sets than naive conformal prediction

### Scaling Up GNNs (from 2021 offering)
Key techniques for training on billion-scale graphs:

1. **Neighbor sampling (GraphSAGE-style):** sample k neighbors per layer
2. **Cluster sampling (Cluster-GCN):** partition into clusters, train on one cluster per batch
3. **Subgraph sampling (GraphSAINT):** sample connected subgraphs; unbiased estimation
4. **LADIES:** importance-based layer-dependent sampling
5. **Historical embeddings (SIGN):** precompute neighborhood features; fast inference

### Community Structure (from 2021 offering)
- **Communities:** groups of nodes densely connected internally, sparsely externally
- **Modularity Q:** measures quality of a partition
  Q = Σ_{s ∈ S} [edges in s - expected edges in s]
- **Louvain algorithm:** greedy modularity optimization; fast, scalable
- **Spectral clustering:** use eigenvectors of Laplacian to identify communities
- **Overlapping communities:** nodes can belong to multiple communities (BigCLAM)

## Key Papers
- [PRODIGY (Huang et al., 2023)](https://arxiv.org/abs/2305.12600)
- [Conformalized GNNs (Huang et al., 2023)](https://arxiv.org/abs/2305.14535)
- [Cluster-GCN (Chiang et al., 2019)](https://arxiv.org/abs/1905.07953)
- [GraphSAINT (Zeng et al., 2020)](https://arxiv.org/abs/1907.04931)

## Textbook Chapter Notes

Maps to **Chapter 17: Advanced GNN Topics**.
- Survey-style chapter covering multiple recent directions
- Code: GraphSAINT training on a large OGB dataset
