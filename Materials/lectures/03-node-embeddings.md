# Lecture 3: Node Embeddings

## Overview

Node embeddings map each node to a low-dimensional vector such that similar nodes (by graph structure) are close in the embedding space. This is unsupervised representation learning.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/02-nodeemb.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/03-nodeemb.pdf

---

## Key Concepts

### Encoder-Decoder Framework
- **Encoder:** ENC(v) = z_v ∈ ℝ^d (maps node to vector)
- **Decoder:** DEC(z_u, z_v) ≈ similarity(u, v) in original graph
- **Objective:** minimize ||DEC(z_u, z_v) − A[u,v]||² (or similar)
- **Shallow embeddings:** each node has a direct lookup embedding; no parameter sharing

### Random Walk Approaches
- **Random walk similarity:** P(v | u) = probability of reaching v on a random walk from u
- **DeepWalk (2014):**
  - Run short random walks from each node
  - Use skip-gram to predict context nodes
  - Maximize log P(N_R(u) | z_u)
  - N_R(u) = neighborhood of u via random walk
- **node2vec (2016):**
  - Biased random walk balancing BFS (local) and DFS (global) exploration
  - Parameters p (return parameter) and q (in-out parameter)
  - p < 1: more BFS-like (structural equivalence)
  - q < 1: more DFS-like (homophily)
  - Flexible: captures both local and global structure

### Optimization (Negative Sampling)
- Exact softmax over all nodes is expensive
- **Negative sampling:** for each positive pair (u, v), sample k negative nodes v_i
- Loss: log σ(z_u · z_v) - Σ log σ(-z_u · z_{v_i})

### Limitations of Shallow Embeddings
- Cannot generalize to unseen nodes (transductive only)
- O(|V|·d) parameters — doesn't scale
- No sharing of structural information across nodes
- Captures only walk-based similarity, not features
- **Solution:** GNNs

### Key Papers
- [DeepWalk (2014)](https://arxiv.org/pdf/1403.6652.pdf) — Perozzi, Al-Rfou, Skiena
- [node2vec (2016)](https://arxiv.org/pdf/1607.00653.pdf) — Grover, Leskovec
- [Network Embedding as Matrix Factorization (2018)](https://arxiv.org/pdf/1710.02971.pdf) — Qiu et al.

## Textbook Chapter Notes

Maps to **Chapter 3: Node Embeddings**.
- Visualization: 2D TSNE plot of embeddings colored by community
- Code: implement node2vec with the node2vec library
- MicroSim idea: interactive biased random walk visualization (BFS vs. DFS slider controlling p/q)

## Code Example

```python
from node2vec import Node2Vec
import networkx as nx

G = nx.karate_club_graph()

# Train node2vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embeddings
node_embedding = model.wv['0']  # embedding for node 0
print(node_embedding.shape)  # (64,)

# Find similar nodes
model.wv.most_similar('0')
```
