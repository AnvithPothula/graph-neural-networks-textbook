---
title: Learning Graph
description: Interactive concept dependency graph for the Graph Neural Networks textbook — 300 concepts, 626 edges, 15 taxonomy categories
---

# Learning Graph: Graph Neural Networks

[Open Learning Graph Viewer Fullscreen](../sims/graph-viewer/main.html){ .md-button .md-button--primary }

<iframe src="../sims/graph-viewer/main.html" width="100%" height="620px" frameborder="0"></iframe>

This section contains the learning graph for the **Graph Neural Networks** textbook. A learning graph is a Directed Acyclic Graph (DAG) of concepts where every edge represents a learning dependency: concept A must be understood before concept B.

The graph has **300 concepts** and **626 dependency edges** organized across **15 taxonomy categories** — from foundational math prerequisites at the left, through core GNN architectures in the middle, to frontier topics like LLM+GNN integration and graph foundation models at the right.

## How to Read the Graph

- **Left side (roots):** Foundational concepts with no prerequisites — matrix multiplication, gradient descent, the undirected graph.
- **Right side (leaves):** Advanced terminal concepts — ULTRA, OFA, RelGNN, DiGress.
- **Arrow direction:** An arrow from A to B means "understand A before B."
- **Node color:** Encodes taxonomy category (see legend below).

## Files in This Section

| File | Description |
|---|---|
| [Concept List](./concept-list.md) | Numbered list of all 300 concepts |
| [Learning Graph CSV](./learning-graph.csv) | Full dependency graph with taxonomy labels |
| [Learning Graph JSON](./learning-graph.json) | vis-network format for interactive viewer |
| [Concept Taxonomy](./concept-taxonomy.md) | 15-category taxonomy with color assignments |
| [Quality Metrics](./quality-metrics.md) | DAG validation, indegree analysis, chain lengths |
| [Taxonomy Distribution](./taxonomy-distribution.md) | Concept counts per category |
| [Course Description Assessment](./course-description-assessment.md) | 97/100 quality report |

## Taxonomy Color Legend

| Color | TaxonomyID | Category | Count |
|---|---|---|---|
| SteelBlue | PREREQ | Prerequisites | 20 |
| DodgerBlue | FOUND | Graph Fundamentals | 37 |
| Teal | ALGO | Classical Graph Algorithms | 33 |
| DarkSlateBlue | EMB | Node Embeddings | 17 |
| Indigo | GNN | GNN Architecture | 44 |
| MediumPurple | THEORY | GNN Theory | 17 |
| DarkOrchid | TRANS | Graph Transformers | 12 |
| Crimson | KG | Knowledge Graphs | 25 |
| DarkRed | HETERO | Heterogeneous Graphs | 8 |
| DarkGreen | APP | Applications | 23 |
| OliveDrab | SCALE | Scalability | 10 |
| DeepPink | GEN | Generative Models | 7 |
| DarkGoldenrod | ADV | Advanced Topics | 18 |
| Orange | TRAIN | Training & Optimization | 20 |
| DimGray | TOOLS | Tools & Frameworks | 9 |
