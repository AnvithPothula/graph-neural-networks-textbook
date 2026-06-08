---
title: Course Description for Graph Neural Networks and Machine Learning with Graphs
description: A detailed course description for Graph Neural Networks and Machine Learning with Graphs including overview, topics covered and learning objectives in the format of the 2001 Bloom Taxonomy
quality_score: 100
---

# Course Description: Machine Learning with Graphs

## Course Title

**Graph Neural Networks and Machine Learning with Graphs**

## Subtitle

*A Comprehensive Intelligent Textbook — From Graph Theory to Foundation Models*

## Inspiration & Standard

This textbook is built from the materials of Stanford CS224W (Jure Leskovec) but is designed to **exceed** that course in depth, accessibility, completeness, and interactivity. Where the course relies on lectures to fill in the gaps between terse slides, this textbook is fully self-contained. Where the course must choose between classical and modern topics due to time constraints, this textbook covers both.

---

## Overview

Complex data can be represented as a graph of relationships between objects. Such networks are a fundamental tool for modeling social, technological, and biological systems. This textbook provides a rigorous, intuition-first, and deeply interactive treatment of machine learning on graphs — from classical graph algorithms and shallow node embeddings, through modern Graph Neural Networks (GNNs), to frontier topics like Graph Transformers, Knowledge Graph Foundation Models, and LLM+GNN integration.

**Topics include:**
- Graph theory fundamentals and real-world network properties
- Classical graph algorithms (PageRank, label propagation, community detection)
- Shallow node embeddings (DeepWalk, node2vec, matrix factorization)
- Graph Neural Networks: GCN, GraphSAGE, GAT, GIN, and beyond
- GNN theory: expressiveness, WL test, over-smoothing, over-squashing
- Graph Transformers (Graphormer, GPS, SAN)
- Heterogeneous graphs and relational data
- Knowledge graph embeddings and multi-hop reasoning
- GNNs for recommender systems, drug discovery, and social networks
- Scalable GNN training on billion-node graphs
- Deep generative models for graphs
- LLMs + GNNs, agents on graphs, graph foundation models
- Frequent subgraph mining and motif analysis

---

## Target Audience

This textbook is designed for:
- **Graduate students** in CS, computational biology, data science, or any field with relational data
- **Advanced undergraduates** who have completed an [ML course](https://anvithpothula.github.io/machine-learning-textbook/) and want to go deeper on graphs
- **Practitioners** building recommendation engines, drug discovery pipelines, fraud detection systems, or knowledge bases
- **Researchers** entering graph ML who want a unified, up-to-date reference

---

## Prerequisites

### Required
- **Programming:** Python proficiency; NumPy/Pandas familiarity
- **Linear algebra:** Matrix multiplication, eigenvalues/eigenvectors, SVD
- **Probability & statistics:** Probability distributions, expectations, Bayes' theorem
- **ML fundamentals:** Supervised/unsupervised learning, gradient descent, backpropagation

### Helpful but taught in Chapter 0
- PyTorch and automatic differentiation
- Graph theory basics (we introduce from scratch)
- PyTorch Geometric (PyG)

---

## Chapter Structure (26 Chapters — Comprehensive Coverage)

The textbook is organized into 6 parts. It covers **all** topics from both the 2021 and 2025 CS224W offerings, plus dedicated prerequisite and application chapters not in the course.

### Part 0: Prerequisites (Chapters 0–1)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 0 | Math & Programming Prerequisites | ❌ Not covered in CS224W |
| 1 | Introduction to Graphs and Networks | ✅ Lecture 1 |

**Chapter 0 covers:** Linear algebra review (matrix operations, eigendecomposition, SVD), probability review, PyTorch basics, PyTorch Geometric installation and first graph, NetworkX tutorial. *This is the chapter that makes the textbook self-contained.*

### Part 1: Classical Graph Analysis (Chapters 2–5)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 2 | Graph Properties and Traditional ML Features | ✅ 2021 Lecture 2 / ❌ dropped from 2025 |
| 3 | Node Embeddings: DeepWalk and node2vec | ✅ Both years |
| 4 | Link Analysis and PageRank | ✅ 2021 Lecture 4 / ❌ dropped from 2025 |
| 5 | Label Propagation and Semi-Supervised Learning | ✅ 2021 Lecture 5 / ❌ dropped from 2025 |

### Part 2: Graph Neural Networks (Chapters 6–11)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 6 | GNN Foundations: Message Passing and GCN | ✅ Both years |
| 7 | GNN Design Space: GraphSAGE and GAT | ✅ Both years |
| 8 | GNN Training, Augmentation, and Practical Tips | ✅ 2025 Lecture 5 |
| 9 | Theory of GNNs: Expressiveness and the WL Test | ✅ Both years |
| 10 | Designing Powerful Encoders: GIN and Beyond | ✅ 2025 Lecture 7 |
| 11 | Graph Transformers | ✅ 2025 Lecture 8 |

### Part 3: Knowledge Graphs and Reasoning (Chapters 12–14)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 12 | Knowledge Graph Embeddings | ✅ Both years |
| 13 | Reasoning over Knowledge Graphs | ✅ 2021 Lecture 11 |
| 14 | Knowledge Graph Foundation Models | ✅ 2025 Lecture 15 |

### Part 4: Graphs in the Wild (Chapters 15–19)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 15 | Heterogeneous Graphs | ✅ 2025 Lecture 9 |
| 16 | GNNs for Recommender Systems | ✅ Both years |
| 17 | Relational Deep Learning | ✅ 2025 Lectures 12–13 |
| 18 | Community Structure in Networks | ✅ 2021 Lecture 14 / ❌ dropped from 2025 |
| 19 | Frequent Subgraph Mining | ✅ 2021 Lecture 12 / ❌ dropped from 2025 |

### Part 5: Scalability and Generation (Chapters 20–22)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 20 | Scaling GNNs to Billion-Node Graphs | ✅ 2021 Lecture 17 |
| 21 | Deep Generative Models for Graphs | ✅ Both years |
| 22 | Temporal and Dynamic Graphs | ❌ Not a dedicated lecture in CS224W |

### Part 6: Frontiers (Chapters 23–26)

| Chapter | Title | CS224W Coverage |
|---|---|---|
| 23 | Advanced GNN Topics: In-Context Learning and Uncertainty | ✅ 2025 Lecture 14 |
| 24 | LLMs + GNNs: Text-Attributed Graphs and Joint Training | ✅ 2025 Lecture 16 |
| 25 | Agents + Graphs | ✅ 2025 Lecture 17 |
| 26 | Conclusion: The GNN Design Space and Open Problems | ✅ Both years |

**Total: 27 chapters** (26 + prerequisites) vs. 19 lectures in CS224W.
**New material not in CS224W at all:** Chapters 0, 22 (temporal graphs), and expanded applications throughout.

---

## Learning Objectives

Using Bloom's Taxonomy (2001 revision), students who complete this textbook will be able to:

### Remember (Level 1)
- Recall the definitions of: graph, node, edge, adjacency matrix, degree, path, connected component, subgraph, clique, bipartite graph, heterogeneous graph
- List the four main graph ML tasks: node classification, link prediction, graph classification, graph generation
- Recall the names and key equations of: GCN, GraphSAGE, GAT, GIN, GraphRNN, TransE, RotatE, LightGCN, Graphormer
- State the Weisfeiler-Lehman graph isomorphism test procedure

### Understand (Level 2)
- Explain why message passing generalizes convolution from grids to irregular graphs
- Describe the difference between transductive and inductive learning with concrete examples
- Summarize why sum aggregation is strictly more expressive than mean or max for graph isomorphism
- Explain the power-law degree distribution and why it appears in real-world networks
- Describe how PageRank solves the spider-trap and dead-end problems via teleportation
- Explain the geometric intuition behind TransE, ComplEx, and RotatE for different relation patterns
- Describe what over-smoothing is, why it occurs, and three ways to mitigate it

### Apply (Level 3)
- Implement GCN, GraphSAGE, GAT, and GIN from scratch in PyTorch and using PyTorch Geometric
- Apply node2vec for unsupervised node embedding with the appropriate p/q settings
- Use the Open Graph Benchmark (OGB) pipeline to train and evaluate a GNN on a standard benchmark
- Apply knowledge graph embedding methods (TransE, RotatE) to link prediction tasks
- Build a LightGCN recommender system on a user-item interaction graph
- Train a scalable GNN using neighbor sampling or cluster-based mini-batching
- Generate molecular graphs conditioned on target properties using GCPN

### Analyze (Level 4)
- Analyze the expressive power of a GNN architecture and identify non-isomorphic graphs it fails to distinguish
- Compare GCN, GraphSAGE, GAT, and GIN architecturally and empirically across node and graph classification tasks
- Examine why certain relation patterns (symmetry, antisymmetry, inversion, composition) require different KG embedding geometry
- Decompose the trade-offs between accuracy and scalability for different GNN sampling strategies
- Evaluate the impact of graph structure vs. node features for a given dataset

### Evaluate (Level 5)
- Assess when graph structure adds value over feature-only baselines
- Critically evaluate a GNN paper's experimental setup for potential confounders (data leakage, unfair baselines)
- Judge which pooling strategy is appropriate for a graph classification task given graph size and structure
- Compare GNN-based approaches to classical graph algorithms for a specific problem
- Evaluate the appropriateness of different KG embedding methods given a dataset's relation patterns

### Create (Level 6)
- Design a novel end-to-end GNN system for a domain-specific task (drug interaction, fraud detection, traffic)
- Construct a knowledge graph from raw data and develop a reasoning pipeline
- Build interactive MicroSim visualizations of key GNN concepts using p5.js
- Develop and tune a GNN that achieves competitive performance on an OGB benchmark
- Write a well-structured literature review connecting classical graph algorithms to modern GNN approaches

---

## Pedagogical Design

Every chapter follows this structure:
1. **Motivation** — real-world problem that this chapter's methods solve
2. **Intuition first** — visual explanation before any math
3. **Mathematical formalism** — full derivations with LaTeX
4. **Algorithm / pseudocode** — step-by-step procedure
5. **PyTorch implementation** — complete, runnable code
6. **MicroSim** — interactive p5.js visualization of the key concept
7. **Empirical results** — benchmark performance with up-to-date numbers
8. **Common pitfalls** — what goes wrong and how to fix it
9. **Research extensions** — open problems and recent work
10. **Chapter summary** — key equations and concepts
11. **Exercises** — Bloom's taxonomy-aligned questions at all 6 levels
12. **Further reading** — 5–8 key papers with annotated descriptions

---

## What Makes This Textbook Better Than CS224W

| Dimension | CS224W Course | This Textbook |
|---|---|---|
| Topic coverage | 19 lectures (2025 drops 5 classical topics) | 26 chapters (covers all 2021 + 2025 topics + more) |
| Prerequisite material | Assumed, not taught | Full Chapter 0: math + PyTorch + PyG |
| Explanations | Slide bullets + live lecture | Full prose derivations, worked examples |
| Intuition | Conveyed verbally in lecture | Written out before every formalism |
| Interactivity | None (slides + colabs) | MicroSims for every key concept |
| Code | 6 Colabs, partial | Complete runnable examples in every chapter |
| Exercises | 3 HWs + 5 Colabs | 6-level Bloom's exercises per chapter |
| Temporal graphs | Brief mention | Dedicated Chapter 22 |
| Benchmark results | 2021-era numbers | Up-to-date 2024–2025 results |
| Accessibility | Requires lecture audio | Fully self-contained |

---

## Style and Technology

- **Format:** MkDocs with Material theme
- **Math:** LaTeX via MathJax; all major results stated as numbered theorems/propositions
- **Code:** Python with PyTorch Geometric; every snippet is self-contained and runnable
- **Interactivity:** p5.js MicroSims for: message passing, random walk, WL color refinement, PageRank power iteration, KG geometry, graph generation
- **Learning graph:** 250+ concepts with full dependency DAG
- **Glossary:** ISO 11179-compliant definitions for all key terms
- **Reading level:** Graduate-level rigor; Flesch-Kincaid Grade ~15

---

## Benchmark References

Primary evaluation datasets:
- **OGB (ogbn-arxiv, ogbn-products, ogbg-molhiv, ogbg-molpcba, ogbl-collab)** — standard node/graph/link benchmarks
- **Cora, CiteSeer, PubMed** — classic citation networks
- **TUDatasets** — graph classification benchmarks
- **FB15k-237, WN18RR** — knowledge graph completion
- **MovieLens-1M** — recommendation
- **ZINC** — molecular property prediction
- **RelBench** — relational databases

---

## Explicitly Out of Scope

The following topics are intentionally **not covered** in this textbook:

- **Reinforcement learning on graphs** — graph-structured state/action spaces, GNN-based policy networks, and graph-based reward shaping are excluded; learners should consult dedicated RL textbooks for this intersection.
- **Graph signal processing (beyond GCN spectral derivation)** — the full GSP literature (filterbanks, graph wavelets, spectral clustering via Laplacian eigenvectors) is covered only insofar as it motivates the GCN derivation in Chapter 6; deeper GSP theory is out of scope.
- **Streaming and dynamic graph algorithms** — temporal graphs (Chapter 22) covers event-based and snapshot models for GNNs, but classical streaming algorithms (e.g., triangle counting under memory constraints) and fully online graph processing are not covered.
- **Hardware-level GNN optimization** — GPU kernel design, sparse tensor formats for graphs, and chip-level inference optimization are not addressed; the focus stays at the software/model level.
- **Survey of GNN libraries beyond PyTorch Geometric** — DGL, Spektral, GraphNets, and JAX-based frameworks are not covered; all code examples use PyTorch Geometric exclusively.
- **Continuous-time differential equations on graphs** — neural ODEs on graphs and diffusion-PDE interpretations of GNNs are touched on briefly in the theory chapter but not developed in depth.
- **Quantum computing on graphs** — quantum graph algorithms and quantum GNNs are entirely out of scope.

---

## Recommended Companion Resources

- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — Hamilton (free) — closest textbook companion
- [Networks, Crowds, and Markets](http://www.cs.cornell.edu/home/kleinber/networks-book/) — Easley & Kleinberg (free) — social network background
- [Network Science](http://networksciencebook.com) — Barabási (free) — complex network properties
- [CS224W Course Notes PDF](https://archives.leni.sh/stanford/CS224w.pdf) — compact formula reference
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/) — primary coding reference
