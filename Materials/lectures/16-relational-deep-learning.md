# Lecture 16: Relational Deep Learning

## Overview

Relational databases can be viewed as graphs. Relational Deep Learning (RDL) applies GNNs to relational databases for predictive tasks, unifying tabular ML with graph representation learning.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/12-RDL.pdf
**Slides (2025, advanced):** http://web.stanford.edu/class/cs224w/slides/13-Advanced_topics_RDL.pdf

---

## Key Concepts

### Relational Databases as Graphs
- A relational database = tables + foreign key relationships
- **Graph construction:**
  - Each row becomes a node
  - Foreign key links become edges
  - Column values become node/edge features
- This "entity graph" can be processed with standard GNN frameworks

### Motivation
- Most enterprise data lives in relational databases (not feature-store ML pipelines)
- Traditional ML: flatten tables → feature engineering → model
- RDL: directly leverage relational structure without manual feature engineering

### RelBench (Fey et al., 2024)
- Benchmark for deep learning on relational databases
- 7 real-world datasets (e-commerce, banking, medical)
- Tasks: node-level (customer churn, loan default), link-level (recommendation)
- Provides standard graph construction from SQL schema

### GNN Architectures for RDL

**RelGNN (2025):**
- Composite message passing for heterogeneous relational graphs
- Handles temporal attributes and many-to-many relationships
- Key innovation: separate encoders per entity type + relation-aware aggregation

**Relational Graph Transformer (2025):**
- Applies transformer-style attention over relational subgraphs
- Incorporates positional encodings derived from relational schema

### Temporal Graphs
- Relational data is often timestamped → temporal graphs
- **Temporal GNN:** only use edges/nodes before a certain time for prediction
- Key challenge: prevent data leakage across time
- Approaches: TGAT, TGN (Temporal Graph Network)

### Handling Heterogeneity in RDL
- Different entity types → different initial embeddings
- Different relation types → different message functions (like R-GCN)
- Schema-aware architecture: GNN structure mirrors ER diagram

## Key Papers
- [Relational Deep Learning (Fey et al., 2024)](https://arxiv.org/pdf/2312.04615)
- [RelBench (Robinson et al., 2024)](https://arxiv.org/pdf/2407.20060)
- [RelGNN (2025)](https://arxiv.org/abs/2502.06784)
- [Relational Graph Transformer (2025)](https://arxiv.org/abs/2505.10960)
- [TGN: Temporal Graph Networks (Rossi et al., 2020)](https://arxiv.org/abs/2006.10637)

## Textbook Chapter Notes

Maps to **Chapter 16: Relational Deep Learning**.
- Worked example: converting a simple e-commerce database to a graph
- Code: RelBench starter tutorial
