# Lecture 15: GNNs for Recommender Systems

## Overview

Recommender systems can be modeled as bipartite user-item graphs. GNNs propagate information through this graph to learn user and item embeddings for collaborative filtering.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/11-recsys.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/13-recsys.pdf

---

## Key Concepts

### Collaborative Filtering Setup
- **Bipartite graph:** users U, items I, edge (u,i) if user u interacted with item i
- **Goal:** predict missing interactions (ratings, clicks, purchases)
- **Matrix factorization:** learn U ∈ ℝ^{|U|×d}, V ∈ ℝ^{|I|×d} such that U·V^T ≈ R
- **Neural CF (NCF):** replace dot product with MLP for better expressiveness

### Why GNNs for Recommendation?
- Multi-hop user-item interactions carry signal:
  - User A liked Item 1 → Item 1 also liked by User B → User B liked Item 2 → recommend Item 2 to User A
- GNNs naturally propagate this higher-order collaborative signal
- Sparse graphs benefit from neighborhood aggregation

### NGCF (Neural Graph Collaborative Filtering, Wang et al., 2019)
- Builds user-item bipartite interaction graph
- GCN layers propagate embeddings across graph
- Concatenates embeddings from all layers before prediction
- Captures collaborative signals at multiple hops

### LightGCN (He et al., 2020)
- **Simplification of NGCF:** removes feature transformation and non-linear activation
- Only the neighborhood aggregation (linear) is retained
- **LightGCN update:**
  e_u^(k+1) = Σ_{i ∈ N_u} (1/√|N_u| · √|N_i|) e_i^(k)
- Final embedding: average over all layers
- **Empirically outperforms NGCF** despite being simpler — suggests those components hurt

### PinSage (Ying et al., 2018 — Pinterest)
- Industrial-scale GNN for web-scale recommendation (3B pins)
- **Key innovations:**
  - Importance-based sampling (not uniform): sample high-influence neighbors via random walk
  - Curriculum training: start with easy negatives, progress to harder ones
  - MapReduce-style efficient inference
- Deployed at Pinterest for home feed recommendations

### Graph-Based Collaborative Filtering Summary

| Method | Aggregation | Transformation | Performance |
|---|---|---|---|
| NCF | None (matrix) | MLP | Baseline |
| NGCF | Sum + concat | Linear + activation | Good |
| LightGCN | Weighted sum | None | State-of-the-art |
| PinSage | Importance-weighted | MLP | Industrial scale |

## Key Papers
- [NGCF: Neural Graph Collaborative Filtering (Wang et al., 2019)](https://arxiv.org/pdf/1905.08108.pdf)
- [LightGCN (He et al., 2020)](https://arxiv.org/pdf/2002.02126.pdf)
- [PinSage (Ying et al., 2018)](https://arxiv.org/pdf/1806.01973.pdf)

## Textbook Chapter Notes

Maps to **Chapter 15: GNNs for Recommender Systems**.
- Code: LightGCN on MovieLens-1M dataset
- MicroSim: bipartite user-item graph with multi-hop propagation visualization
