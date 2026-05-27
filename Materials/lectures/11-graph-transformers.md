# Lecture 11: Graph Transformers

## Overview

Graph Transformers combine the self-attention mechanism of Transformers with graph structure. They overcome some limitations of message-passing GNNs by enabling long-range interactions, and use positional/structural encodings to inject graph topology.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/08-graph-transformer1.pdf

---

## Key Concepts

### Motivation: Why Transformers on Graphs?
- Message-passing GNNs are limited to local neighborhoods (K-hop for K layers)
- Deep GNNs suffer from over-smoothing and over-squashing
- Transformers can directly attend to any node pair → capture long-range dependencies
- Challenge: graphs lack inherent positional order (unlike sequences/images)

### Vanilla Transformer Attention (Recap)
- Attention(Q, K, V) = softmax(QK^T / √d) · V
- For graphs: treat each node as a token; full attention = complete graph

### The Challenge: Positional Encodings for Graphs
Unlike sequences (sin/cos positions) or grids (2D coordinates), graphs lack canonical node ordering.

**Solutions:**
1. **Laplacian eigenvectors (LapPE):**
   - Compute eigenvectors of graph Laplacian L = D - A
   - Use k smallest eigenvectors as positional encoding
   - Sign ambiguity: must use sign-invariant/equivariant processing (SignNet, BasisNet)
2. **Random walk structural encoding (RWSE):**
   - Use k-step random walk landing probabilities as structural features
   - Captures local structure around each node
3. **Relative structural encoding:**
   - Encode distance between node pairs as attention bias

### Key Graph Transformer Architectures

**Graphormer (Ying et al., 2021):**
- Full self-attention over all node pairs
- 3 encodings: node centrality bias (degree), spatial bias (shortest path distance), edge encoding
- Achieves state-of-the-art on OGB molecular property prediction

**SAN (Spectral Attention Network, Kreuzer et al., 2021):**
- Uses full Laplacian spectrum for positional encodings
- Separate attention heads for connected vs. non-connected node pairs

**GPS (General, Powerful, Scalable GNN, Rampasek et al., 2022):**
- Framework combining local MPNN + global attention in each layer
- Modular: plug-in different MPNN types and Transformer types
- Best balance of expressiveness and scalability

**EGT (Edge-featured Graph Transformer):**
- Separate attention matrices for nodes and edges

### Sign and Basis Invariant Networks
- Laplacian eigenvectors have sign ambiguity (v and -v are both valid)
- **SignNet:** learns sign-invariant functions over eigenvectors
- **BasisNet:** learns rotation-equivariant functions over eigenspaces

### Scalability Concerns
- Full attention is O(N²) — expensive for large graphs
- Solutions: sparse attention (only neighbors + some random/long-range), Performer approximations

## Key Papers
- [Graphormer (Ying et al., 2021)](https://arxiv.org/pdf/2106.05234.pdf)
- [SAN (Kreuzer et al., 2021)](https://arxiv.org/pdf/2106.03893.pdf)
- [GPS (Rampasek et al., 2022)](https://arxiv.org/pdf/2205.12454.pdf)
- [SignNet/BasisNet (Lim et al., 2022)](https://arxiv.org/pdf/2202.13013.pdf)
- [Learning Efficient Positional Encodings with GNNs](https://openreview.net/pdf?id=AWg2tkbydO)
- [Attending to Graph Transformers (survey)](https://arxiv.org/pdf/2302.04181.pdf)

## Textbook Chapter Notes

Maps to **Chapter 11: Graph Transformers**.
- Mathematical derivation of Laplacian positional encodings
- Code: GPS model in PyG
- MicroSim idea: show how attention weights change with graph distance
