# Lecture 9: Theory of Graph Neural Networks

## Overview

How expressive are GNNs? What can and cannot they distinguish? This lecture connects GNNs to the Weisfeiler-Lehman graph isomorphism test and characterizes their discriminative power.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/06-theory.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/09-theory.pdf

---

## Key Concepts

### Graph Isomorphism Problem
- Two graphs G₁ and G₂ are **isomorphic** if there exists a bijection between nodes that preserves edge structure
- No polynomial-time algorithm is known for general graph isomorphism
- **GNN power question:** can GNNs distinguish non-isomorphic graphs?

### Weisfeiler-Lehman (WL) Graph Isomorphism Test
**1-WL test (color refinement):**
1. Assign all nodes the same initial color (label)
2. For each node v: new_color(v) = HASH(color(v), multiset{color(u) : u ∈ N(v)})
3. Repeat until colors stabilize
4. If final color histograms differ → graphs are non-isomorphic (not the reverse!)

**Key fact:** The 1-WL test fails on certain graphs (e.g., two different regular graphs of the same degree).

### GNNs ≤ 1-WL Test (Xu et al., 2019)
**Theorem:** Any GNN with sum/mean/max aggregation is at most as powerful as the 1-WL test.

**Intuition:** Each GNN layer computes: h_v^(k+1) = f(h_v^(k), multiset{h_u^(k) : u ∈ N(v)})
This is exactly the 1-WL color refinement! So GNNs can distinguish exactly what 1-WL can.

### Maximum Power: GIN (Graph Isomorphism Network)
**Key insight:** For GNN to be as powerful as 1-WL, the aggregation must be injective on multisets.
- **Sum** over multisets is injective → SUM-based GNN is as powerful as 1-WL
- Mean/Max lose information → less powerful

**GIN update rule:**
h_v^(k+1) = MLP^(k)((1 + ε^(k)) · h_v^(k) + Σ_{u ∈ N(v)} h_u^(k))

**GIN theorem:** GIN with sum aggregation and sufficient MLP depth is as expressive as the 1-WL test.

### What 1-WL (and GINs) Cannot Distinguish
- Triangle counting: GNNs cannot distinguish graphs that differ only by triangle count
- Regular graphs of same degree: 3-regular graphs on 6 nodes that are non-isomorphic
- **Implication:** GNNs cannot detect: cycles of specific length, certain substructures, node positions

### Going Beyond 1-WL
- **k-WL test:** k-tuples of nodes (exponentially more powerful but expensive)
- **Position-aware GNNs (P-GNN):** add positional features based on anchor nodes
- **Identity-aware GNNs (ID-GNN):** inject node identity for context
- **Subgraph GNNs:** run GNN on subgraphs around each node
- **Higher-order GNNs:** operate on k-tuples directly

## Key Papers
- [How Powerful Are Graph Neural Networks? (Xu et al., 2019)](https://arxiv.org/pdf/1810.00826.pdf) — GIN paper
- [Identity-aware GNNs (You et al., 2021)](https://arxiv.org/pdf/2101.10320)
- [Graph Neural Networks are More Powerful Than We Think (Balcilar et al., 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10447704)
- [Counting Graph Substructures with GNNs](https://openreview.net/pdf?id=qaJxPhkYtD)

## Textbook Chapter Notes

Maps to **Chapter 9: Theory of GNNs**.
- Formal proof sketches for GIN theorem
- Code: implement GIN with PyG and show it outperforms mean/max GCN on graph classification
- MicroSim: visualize 1-WL color refinement on two graphs step by step
