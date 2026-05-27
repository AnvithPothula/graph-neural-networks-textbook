# Lecture 2: Traditional Methods for ML on Graphs

## Overview

Before GNNs, graph ML relied on hand-engineered features. This lecture covers classical node, link, and graph-level features, and introduces graph kernels for comparing graphs.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/02-nodeemb.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/02-tradition-ml.pdf

---

## Key Concepts

### Node-Level Features
- **Node degree** — simplest structural feature
- **Node centrality** measures:
  - Eigenvector centrality: a node is important if its neighbors are important (recursive)
  - Betweenness centrality: fraction of shortest paths passing through a node
  - Closeness centrality: avg shortest path length to all other nodes
- **Clustering coefficient:** fraction of closed triangles among neighbors
- **Graphlets:** small subgraph patterns rooted at a node; Graphlet Degree Vector (GDV)

### Link-Level Features
- **Distance-based:** shortest path length between two nodes
- **Local neighborhood overlap:**
  - Common neighbors: |N(u) ∩ N(v)|
  - Jaccard coefficient: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
  - Adamic-Adar index: Σ 1/log(|N(w)|) for w ∈ N(u) ∩ N(v)
- **Global neighborhood overlap:**
  - Katz index: counts paths of all lengths between two nodes (weighted by β^l)
  - PageRank-based similarity

### Graph-Level Features (Kernels)
A **graph kernel** K(G, G') measures similarity between two graphs.
- **Graphlet kernel:** count graphlets in each graph, compare vectors
- **Weisfeiler-Lehman (WL) kernel:**
  1. Assign initial label = node degree
  2. Aggregate neighbor labels (hash)
  3. Repeat K times
  4. Count label occurrences → feature vector
  - WL kernel is efficient and expressive; directly related to WL graph isomorphism test

## Why Hand-Crafted Features Are Limiting

- Features must be designed for each task
- Cannot generalize to unseen graph structures
- Not learned from data
- **Solution:** representation learning — let the model learn features automatically (GNNs)

## Textbook Chapter Notes

Maps to **Chapter 2: Traditional Methods for ML on Graphs**.
- Code examples: computing centrality measures with NetworkX
- Table comparing different link prediction heuristics on benchmark datasets
- MicroSim idea: visualize the WL color refinement process step-by-step

## Code Example

```python
import networkx as nx
G = nx.karate_club_graph()

# Node-level features
degrees = dict(G.degree())
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)

# Link prediction features
preds_jaccard = list(nx.jaccard_coefficient(G))
preds_adamic = list(nx.adamic_adar_index(G))

# WL kernel: use grakel library
# from grakel import WeisfeilerLehman, graph_from_networkx
```
