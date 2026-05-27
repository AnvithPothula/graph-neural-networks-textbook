# Lecture 14: Reasoning over Knowledge Graphs

## Overview

Beyond embedding individual triples, KG reasoning involves answering complex multi-hop queries: "Find drugs that target proteins associated with Disease X." This lecture covers path-based reasoning, query embedding, and GNN-based KG reasoning.

**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/11-reasoning.pdf

---

## Key Concepts

### Multi-hop KG Queries
- **1-hop:** (e, r, ?) — which entities are related to e via r?
- **2-hop path:** What proteins are encoded by genes that regulate disease X?
- **Conjunctive queries:** Find entities satisfying multiple path conditions simultaneously
- **EPFO queries:** Existential Positive First-Order (AND, OR, NOT operations)

### Path-Based Reasoning

**Neural LP / DRUM:**
- Learns rule weights: P(h, r, t) if ∃ path h -r₁→ -r₂→ t
- Soft rules with differentiable learning

**TransE for multi-hop:**
- Sum relation vectors along a path: h + r₁ + r₂ + ... ≈ t
- Simple but effective for chain queries

### Query Embedding (Box Embeddings — Query2Box)
- Embed each query as a hyper-rectangle (box) in entity embedding space
- Answer entities are those inside the box
- **Logical operations:**
  - AND: box intersection (learnable)
  - OR: union of boxes (handled as disjunction)
  - Projection: shift + expand box via a relation
- Enables answering complex EPFO queries over incomplete KGs

### GNN-Based Reasoning (NBFNet)
- Neural Bellman-Ford Networks
- Generalizes Bellman-Ford shortest path algorithm to learned relational paths
- Treats multi-hop reasoning as a path-finding problem
- State-of-the-art on KG link prediction benchmarks

## Key Papers
- [Query2Box (Ren et al., 2020)](https://arxiv.org/abs/2002.05969)
- [NBFNet (Zhu et al., 2021)](https://arxiv.org/abs/2106.06935)
- [DRUM (Sadeghian et al., 2019)](https://arxiv.org/abs/1909.03834)

## Textbook Chapter Notes

Maps to **Chapter 14: Reasoning over Knowledge Graphs**.
- Diagram: visual representation of multi-hop query traversal
- MicroSim: interactive query graph traversal over a toy knowledge graph
