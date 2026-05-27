# Lecture 4: Link Analysis & PageRank

## Overview

PageRank and related algorithms analyze the link structure of graphs to rank nodes by importance. This lecture covers the mathematics of PageRank, its variants (Personalized PageRank), and the HITS algorithm.

**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/04-pagerank.pdf
*(This topic is from the 2021 offering; not a separate lecture in 2025 but still important foundational material.)*

---

## Key Concepts

### PageRank (Google's Algorithm)
- **Idea:** a webpage is important if important pages link to it (recursive)
- **Flow formulation:** r_j = Σ_{i→j} r_i / d_i (where d_i = out-degree of i)
- **Matrix formulation:** r = M · r, where M is the column-stochastic adjacency matrix
- **Power iteration:** start with r⁰ = [1/N, ..., 1/N], iterate r^(t+1) = M · r^t until convergence
- **Eigenvector interpretation:** r is the principal eigenvector of M (eigenvalue = 1)

### Problems with Basic PageRank
1. **Spider traps:** groups of nodes with no outgoing edges trap the random walk
2. **Dead ends:** nodes with no outgoing edges cause rank to leak out of the graph

### Solution: Teleportation (Damping Factor)
- With probability β (typically 0.85), follow a random link
- With probability 1-β, teleport to a random page
- **Google matrix:** A = βM + (1-β)[1/N]_{N×N}
- This ensures a unique stationary distribution exists

### Personalized PageRank (PPR)
- Teleport to a **specific set** S of nodes instead of all nodes uniformly
- Biases the random walk toward a query node or set
- Used in: recommendation, node importance relative to a query, GNN aggregation (Approximated PPR in GNNs)

### HITS Algorithm (Hubs and Authorities)
- Two scores per node: **hub score** h_i and **authority score** a_i
- Authority: a node linked by many hubs is a good authority
- Hub: a node linking to many authorities is a good hub
- Iterative update:
  - a_i = Σ_{j→i} h_j
  - h_i = Σ_{i→j} a_j
- Normalize; converges to principal eigenvectors of A^T A and A A^T

### Applications
- Web search ranking
- Spam detection (TrustRank)
- Anomaly detection in financial networks
- Recommendation (SimRank, PPR-based methods)
- Graph neural networks (APPNP uses PPR)

## Key Papers
- [PageRank (Brin & Page, 1998)](http://ilpubs.stanford.edu:8090/422/) — Original PageRank paper
- [APPNP: Predict then Propagate (2019)](https://arxiv.org/pdf/1810.05997.pdf) — PageRank-inspired GNN

## Textbook Chapter Notes

Maps to **Chapter 4: Link Analysis & PageRank**.
- MicroSim idea: animate power iteration converging to stationary distribution on a small graph
- Code: compute PageRank with NetworkX; compare to manual power iteration
